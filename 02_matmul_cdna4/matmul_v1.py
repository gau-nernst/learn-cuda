import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import T, rocdl
from torch import Tensor


def local_tile(x, tiler, coord):
    return fx.zipped_divide(x, tiler)[(None,) * len(tiler), coord]


def build_matmul_v1():
    BLOCK_M = 256
    BLOCK_N = 128
    BLOCK_K = 64
    num_wave_m = 2
    num_wave_n = 2

    WAVE_M = BLOCK_M // num_wave_m
    WAVE_N = BLOCK_N // num_wave_n
    MFMA_M = 16
    MFMA_N = 16
    MFMA_K = 32

    num_waves = num_wave_m * num_wave_n
    block_size = num_waves * 64

    @fx.struct
    class SharedStorage:
        # 16 is alignment
        a: fx.Array[fx.BFloat16, BLOCK_M * BLOCK_K, 16]
        b: fx.Array[fx.BFloat16, BLOCK_N * BLOCK_K, 16]

    @flyc.kernel(name="matmul_v1")
    def kernel(
        gA: fx.Tensor,
        gB: fx.Tensor,
        gC: fx.Tensor,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Constexpr[int],
    ):
        f32x4 = fx.Vector.make_type(4, fx.Float32)
        bf16x8 = fx.Vector.make_type(8, fx.BFloat16)

        tid = fx.thread_idx.x
        wave_id = rocdl.readfirstlane(T.i32, tid // 64)  # wave-uniform
        lane_id = fx.lane_id()

        wave_id_n = wave_id % num_wave_n
        wave_id_m = wave_id // num_wave_n

        bid_n = fx.block_idx.x
        bid_m = fx.block_idx.y

        gA_ptr = fx.get_iter(gA) + gA.layout(bid_m * BLOCK_M, 0)
        gB_ptr = fx.get_iter(gB) + gB.layout(0, bid_n * BLOCK_N)

        A_buf = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(gA_ptr, num_records_bytes=(M - bid_m * BLOCK_M) * K * 2))
        B_buf = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(gB_ptr, num_records_bytes=(N - bid_n * BLOCK_N) * K * 2))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        NUM_A_DMA = BLOCK_M * BLOCK_K * 2 // (block_size * 16)
        NUM_B_DMA = BLOCK_N * BLOCK_K * 2 // (block_size * 16)

        acc = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in fx.range_constexpr(WAVE_N // MFMA_N)]
            for _ in fx.range_constexpr(WAVE_M // MFMA_M)
        ]

        for iter_k in range(K // BLOCK_K):
            fx.gpu.barrier()  # everyone finishes

            for i in fx.range_constexpr(NUM_A_DMA):
                idx = (i * block_size + tid) * 8
                col = idx % BLOCK_K
                row = idx // BLOCK_K
                voffset = (row * K + iter_k * BLOCK_K + col) * 2

                # NOTE: gmem source contains lane-dependent offset, specified as voffset.
                # but LDS destination is a wave-uniform address, where per-lane offset
                # is added automatically.
                dst = lds.a.ptr + (i * block_size + wave_id * 64) * 8  # wave-uniform
                rocdl.buffer_load_to_lds(A_buf, dst.llvm_ptr, voffset, size_bytes=16)

            for i in fx.range_constexpr(NUM_B_DMA):
                idx = (i * block_size + tid) * 8
                col = idx % BLOCK_K
                row = idx // BLOCK_K
                voffset = (row * K + iter_k * BLOCK_K + col) * 2

                dst = lds.b.ptr + (i * block_size + wave_id * 64) * 8  # wave-uniform
                rocdl.buffer_load_to_lds(B_buf, dst.llvm_ptr, voffset, size_bytes=16)

            fx.gpu.barrier()  # cross-wave visibility

            # target 16x16x32 BF16 MFMA instruction
            # Input layout
            #   K_L = K / (64 / (M * B)) = 8 elems = 16B
            #   A[m,k] is held by lane (k/8) * 16 + m, item (k%8)-th
            #
            #     |--lane  0--||--lane 16--||--lane 32--||--lane 48--|
            #     |--lane  1--||--lane 17--||--lane 33--||--lane 49--|
            #     |--  ...  --||--  ...  --||--  ...  --||--  ...  --|
            #     |--lane 15--||--lane 31--||--lane 47--||--lane 63--|
            #
            # Output layout
            #   H = 4
            #   B_I = ceil(64 / (N * M / H)) = 1
            #   M_I = (64 / B_I) / N = 4
            #   G = M / (H * M_I) = 1
            #   D[m,n] is held by lane (m/4) * 16 + n, item (m%4)-th
            #
            #   |lane  0|lane  1| ... |lane 15|
            #   |lane  0|lane  1| ... |lane 15|x3
            #   |lane 16|lane 17| ... |lane 31|
            #   |lane 16|lane 17| ... |lane 31|x3
            #   |lane 32|lane 33| ... |lane 47|
            #   |lane 32|lane 33| ... |lane 47|x3
            #   |lane 48|lane 49| ... |lane 63|
            #   |lane 48|lane 49| ... |lane 63|x3

            for k in fx.range_constexpr(BLOCK_K // MFMA_K):
                off_am = wave_id_m * WAVE_M + (lane_id % 16)
                off_bn = wave_id_n * WAVE_N + (lane_id % 16)
                off_k = k * MFMA_K + (lane_id // 16) * 8

                # load A and B from LDS to registers
                rA = [
                    (lds.a.ptr + ((off_am + m * MFMA_M) * BLOCK_K + off_k)).load(bf16x8)
                    for m in fx.range_constexpr(WAVE_M // MFMA_M)
                ]
                rB = [
                    (lds.b.ptr + ((off_bn + n * MFMA_N) * BLOCK_K + off_k)).load(bf16x8)
                    for n in fx.range_constexpr(WAVE_N // MFMA_N)
                ]

                for m in fx.range_constexpr(WAVE_M // MFMA_M):
                    for n in fx.range_constexpr(WAVE_N // MFMA_N):
                        # swap A and B, so output is N-contiguous
                        acc[m][n] = rocdl.mfma_f32_16x16x32_bf16(f32x4, [rB[n], rA[m], acc[m][n]])

        for m in fx.range_constexpr(WAVE_M // MFMA_M):
            for n in fx.range_constexpr(WAVE_N // MFMA_N):
                off_m = bid_m * BLOCK_M + wave_id_m * WAVE_M + m * MFMA_M + (lane_id % 16)
                off_n = bid_n * BLOCK_N + wave_id_n * WAVE_N + n * MFMA_N + (lane_id // 16) * 4
                dst = fx.get_iter(gC) + gC.layout(off_m, off_n)
                dst.store(acc[m][n].to(fx.BFloat16))  # bf16x4

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Constexpr[int],
        stream: fx.Stream,
    ):
        kernel(A, B, C, M, N, K).launch(
            grid=(N // BLOCK_N, M // BLOCK_M, 1),
            block=(block_size, 1, 1),
            stream=stream,
        )

    return launch


_matmul_v1 = build_matmul_v1()


def matmul_v1(A: Tensor, B: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = A.new_empty(M, N)
    _matmul_v1(A, B, C, M, N, K, torch.cuda.current_stream(A.device))
    return C
