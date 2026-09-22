import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import T, rocdl
from torch import Tensor


def local_tile(x, tiler, coord):
    return fx.zipped_divide(x, tiler)[(None,) * len(tiler), coord]


def build_matmul_v1():
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    num_wave_m = 2
    num_wave_n = 2

    WAVE_M = BLOCK_M // num_wave_m
    WAVE_N = BLOCK_N // num_wave_n

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
        tid = fx.thread_idx.x
        wave_id = rocdl.readfirstlane(T.i32, tid // 64)  # wave-uniform
        lane_id = fx.lane_id()

        bid_n = fx.block_idx.x
        bid_m = fx.block_idx.y

        gA_ptr = fx.get_iter(gA) + gA.layout(bid_m * BLOCK_M, 0)
        gB_ptr = fx.get_iter(gB) + gB.layout(0, bid_n * BLOCK_N)

        A_buf = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(gA_ptr, num_records_bytes=(M - bid_m * BLOCK_M) * K * 2))
        B_buf = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(gB_ptr, num_records_bytes=(N - bid_n * BLOCK_N) * K * 2))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        NUM_A_DMA = BLOCK_M * BLOCK_K * 2 // (block_size * 16)
        NUM_B_DMA = BLOCK_N * BLOCK_K * 2 // (block_size * 16)

        NUM_ACC_REGS = BLOCK_M * BLOCK_N // block_size
        acc = fx.make_rmem_tensor(NUM_ACC_REGS, fx.Float32)
        acc.fill(0.0)

        for iter_k in range(K // BLOCK_K):
            fx.gpu.barrier()  # everyone finishes

            for i in fx.range_constexpr(NUM_A_DMA):
                idx = (i * block_size + tid) * 8
                col = idx % BLOCK_K
                row = idx // BLOCK_K
                voffset = (row * K + iter_k * BLOCK_K + col) * 2

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

            for i in fx.range_constexpr(NUM_ACC_REGS):
                idx = i * block_size + tid
                n = idx % BLOCK_N
                m = idx // BLOCK_N

                for k in fx.range_constexpr(BLOCK_K):
                    rA = lds.a[m * BLOCK_K + k].to(fx.Float32)
                    rB = lds.b[n * BLOCK_K + k].to(fx.Float32)
                    acc[i] = fx.fma(rA, rB, acc[i])

        for i in fx.range_constexpr(NUM_ACC_REGS):
            idx = i * block_size + tid
            n = bid_n * BLOCK_N + idx % BLOCK_N
            m = bid_m * BLOCK_M + idx // BLOCK_N
            gC[m, n] = acc[i].to(fx.BFloat16)

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
