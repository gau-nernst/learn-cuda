import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import rocdl
from torch import Tensor


def build_matmul_v0():
    num_waves = 8
    block_size = num_waves * 64
    vec_a = 4
    vec_b = 8

    @flyc.kernel(name="matmul_v0")
    def kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Constexpr[int],
    ):
        wave_id = fx.thread_idx.x // 64
        lane_id = fx.lane_id()
        m = fx.block_idx.x

        # [M,K] -> [K,M]
        A = fx.make_view(A.iter, fx.select(A.layout, indices=[1, 0]))

        A_buf = rocdl.make_buffer_tensor(A)
        B_buf = rocdl.make_buffer_tensor(B)
        A_slices = fx.zipped_divide(A_buf, (8, vec_a))  # (8,vec),(K/8,M/vec)
        B_slices = fx.zipped_divide(B_buf, (8, vec_b))  # (8,vec),(K/8,N/vec)

        ldg_16B_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        for n in range(wave_id, N // vec_b, num_waves):
            acc = fx.make_rmem_tensor((vec_a, vec_b), fx.Float32)

            # the compiler may fail acc.fill(0.0) for degenerate cases
            # e.g. vec_a=1
            for a in fx.range_constexpr(vec_a):
                for b in fx.range_constexpr(vec_b):
                    acc[a, b] = 0.0

            rA = fx.make_rmem_tensor((8, vec_a), fx.BFloat16)
            rB = fx.make_rmem_tensor((8, vec_b), fx.BFloat16)

            for iter_k in range(K // (64 * 8)):
                k = iter_k * 64 + lane_id

                fx.copy(ldg_16B_atom, A_slices[None, (k, m)], rA)
                fx.copy(ldg_16B_atom, B_slices[None, (k, n)], rB)

                # FlyDSL doesn't codegen fma automatically
                for j in fx.range_constexpr(8):
                    for a in fx.range_constexpr(vec_a):
                        for b in fx.range_constexpr(vec_b):
                            acc[a, b] = fx.fma(rA[j, a].to(fx.Float32), rB[j, b].to(fx.Float32), acc[a, b])

            # warp reduction
            for i in fx.range_constexpr(int(math.log2(64))):
                offset = 1 << i
                for a in fx.range_constexpr(vec_a):
                    for b in fx.range_constexpr(vec_b):
                        acc[a, b] += acc[a, b].shuffle_xor(offset, 64)

            if lane_id == 0:
                for a in fx.range_constexpr(vec_a):
                    for b in fx.range_constexpr(vec_b):
                        C[m * vec_a + a, n * vec_b + b] = acc[a, b].to(fx.BFloat16)

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
            grid=(M // vec_a, 1, 1),
            block=(block_size, 1, 1),
            stream=stream,
        )

    return launch


_matmul_v0 = build_matmul_v0()


def matmul_v0(A: Tensor, B: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = A.new_empty(M, N)
    _matmul_v0(A, B, C, M, N, K, torch.cuda.current_stream(A.device))
    return C
