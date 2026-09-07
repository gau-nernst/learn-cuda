import os

os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_LINEINFO"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = "./cutedsl_dump"
os.environ["CUTE_DSL_NO_CACHE"] = "1"

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cute_utils import _tcgen05, simple_tma_g2s
from cutlass import BFloat16, Int32, Int64, cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import get_smem_capacity_in_bytes
from triton.testing import do_bench


class MatmulV1Kernel:
    def __init__(self, BN: int = 128):
        BM = 128
        BK = 64
        self.cta_tile = (BM, BN, BK)

        smem_bytes = get_smem_capacity_in_bytes()
        self.stage_size = (BM + BN) * BK * 2
        self.num_stages = smem_bytes // self.stage_size

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        # we must put num_stages as the last mode since a lot of CuteDSL functions assume that
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)

        # don't need to select 1 stage of s_layout, make_tiled_tma_atom() does it internally
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BM, BK))
        return tma_atom, tma_tensor, s_layout

    @cute.jit
    def __call__(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream: CUstream):
        BM, BN, BK = self.cta_tile
        A_args = self.prepare_AB(A, BM, BK)
        B_args = self.prepare_AB(B, BN, BK)

        M, _ = A.shape
        N, _ = B.shape
        grid_m = cute.ceil_div(M, BM)
        grid_n = cute.ceil_div(N, BN)
        self.kernel(A_args, B_args, C).launch(
            grid=(grid_m, grid_n, 1),
            block=(6 * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        A_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        B_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        C_tensor: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        BM, BN, BK = self.cta_tile

        A_tma_atom, A_tma_tensor, sA_layout = A_args
        B_tma_atom, B_tma_tensor, sB_layout = B_args

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(BFloat16, sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)
        tma_full_mbar = smem.allocate_array(Int64, self.num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, self.num_stages)
        tmem_full_mbar = smem.allocate(Int64, 8)
        taddr = smem.allocate(Int32, 4)

        M, K = A_tma_tensor.shape

        if warp_id == 0:
            for i in cutlass.range_constexpr(self.num_stages):
                cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
            cute.arch.mbarrier_init(tmem_full_mbar, 1)
            cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma_atom)
            cpasync.prefetch_descriptor(B_tma_atom)
        cute.arch.sync_threads()

        # TMA warp
        if warp_id == 5:
            # select gmem tile
            gA_tile = cute.local_tile(A_tma_tensor, (BM, BK), (bid_m, None))  # [BM, BK, K/BK]
            gB_tile = cute.local_tile(B_tma_tensor, (BN, BK), (bid_n, None))  # [BN, BK, K/BK]

            tma_stage = 0
            empty_phase = 1

            for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, empty_phase)

                mbar = tma_full_mbar + tma_stage
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, self.stage_size)
                simple_tma_g2s(A_tma_atom, gA_tile[None, None, iter_k], sA[None, None, tma_stage], mbar)
                simple_tma_g2s(B_tma_atom, gB_tile[None, None, iter_k], sB[None, None, tma_stage], mbar)

                tma_stage = (tma_stage + 1) % self.num_stages
                if tma_stage == 0:
                    empty_phase ^= 1

        # MMA warp
        elif warp_id == 4:
            _tcgen05.alloc(taddr)

            tma_stage = 0
            full_phase = 0

            # BF16 MMA
            idesc = cutlass.const_expr((1 << 4) | (1 << 7) | (1 << 10) | (BN >> 3 << 17) | (BM >> 4 << 24))
            # 128B swizzling
            sdesc = cutlass.const_expr(((8 * 128) >> 4 << 32) | (1 << 46) | (2 << 61))

            for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, full_phase)
                _tcgen05.fence_after_thread_sync()

                a_desc = sdesc | (sA[None, None, tma_stage].iterator.toint() >> 4)
                b_desc = sdesc | (sB[None, None, tma_stage].iterator.toint() >> 4)

                MMA_K = cutlass.const_expr(16)  # 32B

                with cute.arch.elect_one():
                    for k in cutlass.range_constexpr(BK // MMA_K):
                        _tcgen05.mma_f16(0, a_desc, b_desc, idesc, iter_k > 0 or k > 0)
                        a_desc += 32 >> 4
                        b_desc += 32 >> 4
                    _tcgen05.commit(tma_empty_mbar + tma_stage)

                tma_stage = (tma_stage + 1) % self.num_stages
                if tma_stage == 0:
                    full_phase ^= 1

            with cute.arch.elect_one():
                _tcgen05.commit(tmem_full_mbar)

        # epilogue warps
        else:
            # (M, (WIDTH, N/WIDTH))
            WIDTH = cutlass.const_expr(16)
            C_ = cute.logical_divide(C_tensor, tiler=(None, WIDTH))

            bf16x16_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                BFloat16,
                num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            if warp_id == 0:
                cute.arch.mbarrier_wait(tmem_full_mbar, 0)
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            _tcgen05.fence_after_thread_sync()

            for i in cutlass.range_constexpr(BN // WIDTH):
                tmem = ((warp_id * 32) << 16) | (i * WIDTH)
                regs = _tcgen05.ld(tmem, "32x32b", WIDTH)
                nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD)

                # regs is TensorSSA. .to(BFloat16) is still TensorSSA
                # cute.copy doesn't accept TensorSSA -> we need to copy to cute.Tensor
                tmp = cute.make_rmem_tensor(16, BFloat16)
                tmp.store(regs.to(BFloat16))

                # C_ shape: (M, (WIDTH, N/WIDTH))
                dst = C_[bid_m * BM + tid, (None, bid_n * (BN // WIDTH) + i)]
                cute.copy(bf16x16_atom, tmp, dst)

            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            if warp_id == 0:
                _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile(BN: int):
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()
        A = cute.runtime.make_fake_tensor(BFloat16, (M, K), (K, 1), assumed_align=8)
        B = cute.runtime.make_fake_tensor(BFloat16, (N, K), (K, 1), assumed_align=8)
        C = cute.runtime.make_fake_tensor(BFloat16, (M, N), (cute.sym_int(divisibility=16), 1), assumed_align=32)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = MatmulV1Kernel(BN)
        return cute.compile(kernel, A, B, C, stream, options="--enable-tvm-ffi")


def cutedsl_v1(A: torch.Tensor, B: torch.Tensor):
    C = A.new_empty(A.shape[0], B.shape[1])
    MatmulV1Kernel.compile(256)(A, B.T, C)
    return C


def main():
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    C_ref = A @ B.T
    C = cutedsl_v1(A, B.T)
    torch.cuda.synchronize()
    torch.testing.assert_close(C, C_ref)

    cublas_ms = do_bench(lambda: torch.mm(A, B.T))
    ours_ms = do_bench(lambda: cutedsl_v1(A, B.T))

    cublas_tflops = 2 * M * N * K / (cublas_ms * 1e-3) * 1e-12
    ours_tflops = 2 * M * N * K / (ours_ms * 1e-3) * 1e-12
    print(f"{cublas_tflops=}")
    print(f"{ours_tflops=}")


if __name__ == "__main__":
    main()
