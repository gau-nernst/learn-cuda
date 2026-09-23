import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import T, rocdl
from torch import Tensor


def local_tile(x, tiler, coord):
    return fx.zipped_divide(x, tiler)[(None,) * len(tiler), coord]


def buffer_load_to_lds_async(rsrc, lds_ptr, voffset, size_bytes: int, soffset: int = 0, offset: int = 0):
    rocdl.raw_ptr_buffer_load_async_lds(
        rsrc,
        lds_ptr,
        fx.Int32(size_bytes).ir_value(),
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        fx.Int32(offset).ir_value(),
    )


def build_matmul_v3():
    BLOCK_M = 256
    BLOCK_N = 128
    BLOCK_K = 64
    num_wave_m = 2
    num_wave_n = 2
    num_stages = 2

    WAVE_M = BLOCK_M // num_wave_m
    WAVE_N = BLOCK_N // num_wave_n
    MFMA_M = 16
    MFMA_N = 16
    MFMA_K = 32

    A_SIZE = BLOCK_M * BLOCK_K
    B_SIZE = BLOCK_N * BLOCK_K

    num_waves = num_wave_m * num_wave_n
    block_size = num_waves * 64

    @fx.struct
    class SharedStorage:
        # 16 is alignment
        a: fx.Array[fx.BFloat16, A_SIZE * num_stages, 16]
        b: fx.Array[fx.BFloat16, B_SIZE * num_stages, 16]

    @flyc.kernel(name="matmul_v3")
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

        acc = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in fx.range_constexpr(WAVE_N // MFMA_N)]
            for _ in fx.range_constexpr(WAVE_M // MFMA_M)
        ]

        # cap by 64 banks
        swizzle_span_chunks = min(BLOCK_K * 2, 64 * 4) // 16

        def dma(iter_k, stage_id):
            NUM_A_DMA = A_SIZE * 2 // (block_size * 16)
            for i in fx.range_constexpr(NUM_A_DMA):
                idx = (i * block_size + tid) * 8
                col = idx % BLOCK_K
                row = idx // BLOCK_K

                col = ((col // 8) ^ (row % swizzle_span_chunks)) * 8

                voffset = (row * K + iter_k * BLOCK_K + col) * 2
                dst = lds.a.ptr + stage_id * A_SIZE + (i * block_size + wave_id * 64) * 8  # wave-uniform
                buffer_load_to_lds_async(A_buf, dst.llvm_ptr, voffset, size_bytes=16)

            NUM_B_DMA = B_SIZE * 2 // (block_size * 16)
            for i in fx.range_constexpr(NUM_B_DMA):
                idx = (i * block_size + tid) * 8
                col = idx % BLOCK_K
                row = idx // BLOCK_K

                col = ((col // 8) ^ (row % swizzle_span_chunks)) * 8

                voffset = (row * K + iter_k * BLOCK_K + col) * 2
                dst = lds.b.ptr + stage_id * B_SIZE + (i * block_size + wave_id * 64) * 8  # wave-uniform
                buffer_load_to_lds_async(B_buf, dst.llvm_ptr, voffset, size_bytes=16)

            # mark a group
            rocdl.asyncmark()

        def mfma(acc, mfma_stage):
            for k in fx.range_constexpr(BLOCK_K // MFMA_K):
                row = lane_id % 16
                off_am = wave_id_m * WAVE_M + row
                off_bn = wave_id_n * WAVE_N + row

                chunk = k * (MFMA_K // 8) + (lane_id // 16)  # 16-byte chunk
                off_k = (chunk ^ (row % swizzle_span_chunks)) * 8

                # load A and B from LDS to registers
                rA = [
                    (lds.a.ptr + mfma_stage * A_SIZE + ((off_am + m * MFMA_M) * BLOCK_K + off_k)).load(bf16x8)
                    for m in fx.range_constexpr(WAVE_M // MFMA_M)
                ]
                rB = [
                    (lds.b.ptr + mfma_stage * B_SIZE + ((off_bn + n * MFMA_N) * BLOCK_K + off_k)).load(bf16x8)
                    for n in fx.range_constexpr(WAVE_N // MFMA_N)
                ]

                for m in fx.range_constexpr(WAVE_M // MFMA_M):
                    for n in fx.range_constexpr(WAVE_N // MFMA_N):
                        # swap A and B, so output is N-contiguous
                        acc[m][n] = rocdl.mfma_f32_16x16x32_bf16(f32x4, [rB[n], rA[m], acc[m][n]])

            return acc

        # issue prefetch
        for iter_k in range(num_stages - 1):
            dma(iter_k, iter_k)

        dma_stage = num_stages - 1
        mfma_stage = 0

        for iter_k in range(K // BLOCK_K - (num_stages - 1)):
            fx.gpu.barrier()  # everyone finishes
            dma(iter_k + num_stages - 1, dma_stage)
            dma_stage = (dma_stage + 1) % num_stages

            rocdl.wait_asyncmark(num_stages - 1)  # drain 1 DMA stage
            fx.gpu.barrier()  # cross-wave visibility
            acc = mfma(acc, mfma_stage)
            mfma_stage = (mfma_stage + 1) % num_stages

        # peel off last compute iterations
        for i in fx.range_constexpr(num_stages - 1):
            rocdl.wait_asyncmark(num_stages - 2 - i)
            fx.gpu.barrier()  # cross-wave visibility
            acc = mfma(acc, mfma_stage)
            mfma_stage = (mfma_stage + 1) % num_stages

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


_matmul_v3 = build_matmul_v3()


def matmul_v3(A: Tensor, B: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = A.new_empty(M, N)
    _matmul_v3(A, B, C, M, N, K, torch.cuda.current_stream(A.device))
    return C
