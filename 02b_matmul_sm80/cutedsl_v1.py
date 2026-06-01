import math
from functools import cache
from typing import NamedTuple

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, cute
from cutlass.cute.nvgpu import CopyUniversalOp, cpasync, warp
from torch import Tensor

WARP_SIZE = 32


# all self's attributes are constexpr
class MatmulV1Kernel(NamedTuple):
    warp_layout: tuple[int, int] = (2, 2)
    cta_tile: tuple[int, int, int] = (128, 128, 32)
    num_stages: int = 3

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream: CUstream):
        BM, BN, BK = self.cta_tile
        num_threads = math.prod(self.warp_layout) * WARP_SIZE

        # how swizzling is done in cute
        # https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/
        # Swizzle<B, M, S> is applied on element index
        # - B bits will be XOR-ed with a mask
        # - these B bits start at bit-M
        # - the mask starts at bit-(M+S)

        # a layout atom is [8, BK], for BK up to 64 (i.e. 128B)
        # 8 is from ldmatrix. BK up to 64 is due to how CuteDSL does swizzling.
        major_size = min(BK, 64)
        swizzle_bits = int(math.log2(major_size * 2 // 16))
        AB_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),  # should it be 3,4,3?
            offset=0,
            outer=cute.make_layout((8, major_size), stride=(major_size, 1)),
        )
        # ((8,BM/8), (major_size,BK/major_size),(1,num_stages))
        sA_layout = cute.tile_to_shape(AB_atom, (BM, BK, self.num_stages), (0, 1, 2))
        sB_layout = cute.tile_to_shape(AB_atom, (BN, BK, self.num_stages), (0, 1, 2))

        # TODO: remove smem store
        swizzle_bits = min(int(math.log2(BN * 2 // 16)), 3)
        C_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4),
            offset=0,
            outer=cute.make_layout((8, BN), stride=(BN, 1)),
        )
        sC_layout = cute.tile_to_shape(C_atom, (BM, BN), order=(0, 1))

        # global->shared for AB: cp.async.cg 16
        op = cpasync.CopyG2SOp(cute.nvgpu.LoadCacheMode.GLOBAL)
        cpasync_atom = cute.make_copy_atom(op, BFloat16, num_bits_per_copy=128)

        # thread layout: mapping from coordinates to thread ID
        # each cp.async 16B copies 8 BF16 elems. hence, we need (BK/cp_elems) threads to cover BK.
        # the whole CTA covers (num_threads/shape_dim1, shape_dim1) tile of copy atoms.
        cp_elems = 8
        shape_dim1 = BK // cp_elems
        t_layout = cute.make_layout(
            (num_threads // shape_dim1, shape_dim1),
            stride=(shape_dim1, 1),
        )

        # value layout: mapping from coordinate to value IDs
        # each thread holds (1, cp_elems) tile
        v_layout = cute.make_layout((1, cp_elems))

        # thread-value layout: a 2D layout (T, V), where there are T threads and V values,
        # that maps to linear position of a data.
        # this tiled_copy represents a copy issued by partipating threads, covering
        # a particular tile - (num_threads * 8 / BK, BK) of BF16 elems in this case.
        # => CTA-level tiled copy
        tiled_copy_AB = cute.make_tiled_copy_tv(cpasync_atom, t_layout, v_layout)

        # standard 16B copy for C.
        # same TV layout as AB's cp.async, only different in op.
        atom_cp = cute.make_copy_atom(CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        tiled_copy_C = cute.make_tiled_copy_tv(atom_cp, t_layout, v_layout)

        # create MMA: mma.m16n8k16
        # similarly, this is CTA-level tiled MMA
        # permutation_mnk can be read as CTA-level tile. but it's actually 3 layouts,
        # one in each MNK mode. these layouts "permute" the ordering of elements along
        # that mode.
        atom_m, atom_n = self.warp_layout
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16)),
            atom_layout_mnk=(atom_m, atom_n, 1),  # tile along m first, then n
            permutation_mnk=(atom_m * 16, atom_n * 16, 16),
        )

        grid_m = cute.ceil_div(mC.shape[0], BM)
        grid_n = cute.ceil_div(mC.shape[1], BN)
        smem_size = max((BM + BN) * BK * self.num_stages, BM * BN) * 2

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_AB,
            tiled_copy_C,
            tiled_mma,
        ).launch(grid=(grid_m, grid_n), block=[num_threads, 1, 1], smem=smem_size, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_g2s_AB: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        K = mA.shape[1]
        BM, BN, BK = self.cta_tile
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()

        # select input/output tiles
        gA = cute.local_tile(mA, tiler=(BM, BK), coord=(bid_m, None))  # (BM, BK, K/BK)
        gB = cute.local_tile(mB, tiler=(BN, BK), coord=(bid_n, None))  # (BN, BK, K/BK)
        gC = cute.local_tile(mC, tiler=(BM, BN), coord=(bid_m, bid_n))  # (BM, BN)

        # allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, sA_layout, 16)
        sB = smem.allocate_tensor(BFloat16, sB_layout, 16)
        sC = cute.make_tensor(sA.iterator, sC_layout)  # overlap buffer

        # get a thread's part of this tiled copy
        thr_g2s_A = tiled_copy_g2s_AB.get_slice(tid)
        thr_g2s_B = tiled_copy_g2s_AB.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        # get a thread's view of Source and Destination data.
        # gA is (BM, BK, K/BK), while sA is (BM, BK, num_stages)
        # each tiled_copy only covers (xx, BK) -> there is tiling as well
        # each of these is a cute.Tensor
        tAgA = thr_g2s_A.partition_S(gA)  # (cp_atom.shape, restM, restK, k_tiles)
        tAsA = thr_g2s_A.partition_D(sA)
        tBgB = thr_g2s_B.partition_S(gB)
        tBsB = thr_g2s_B.partition_D(sB)

        tCsC_epilogue = thr_copy_C.partition_S(sC)
        tCgC_epilogue = thr_copy_C.partition_D(gC)

        # prefetch
        for k_tile in range(self.num_stages - 1):
            cute.copy(
                tiled_copy_g2s_AB,
                tAgA[None, None, None, k_tile],
                tAsA[None, None, None, k_tile],
            )
            cute.copy(
                tiled_copy_g2s_AB,
                tBgB[None, None, None, k_tile],
                tBsB[None, None, None, k_tile],
            )
            cute.arch.cp_async_commit_group()

        # get a thread's view of A, B, and C data.
        # similarly, there will be "tiling" effect:
        # (BM, BN, BK) / (32, 32, 16) -> (32, 32, 16, BM/32, BN/32, BK/16)
        # NOTE: feels kinda awkward. these partitions are done on smem,
        # which is only used to create rmem tensor in the next step.
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)

        # rmem for MMA
        # tCrA: (mma_atom.shape, rest_M, rest_K)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        # s2r tiled copy, created from tiled MMA
        ldmatrix_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            BFloat16,
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(ldmatrix_atom, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(ldmatrix_atom, tiled_mma)

        thr_s2r_A = tiled_copy_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_copy_s2r_B.get_slice(tid)

        # .retile() reinterprets an existing rmem tensor to match ldmatrix copy atom.
        tCsA_s2r = thr_s2r_A.partition_S(sA)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCsB_s2r = thr_s2r_B.partition_S(sB)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

        read_stage = 0
        write_stage = self.num_stages - 1

        # main loop
        for k_tile in range(K // BK):
            # prefetch
            prefetch_k_tile = k_tile + self.num_stages - 1
            if prefetch_k_tile < K // BK:
                cute.arch.sync_threads()
                cute.copy(
                    tiled_copy_g2s_AB,
                    tAgA[None, None, None, prefetch_k_tile],
                    tAsA[None, None, None, write_stage],
                )
                cute.copy(
                    tiled_copy_g2s_AB,
                    tBgB[None, None, None, prefetch_k_tile],
                    tBsB[None, None, None, write_stage],
                )
            cute.arch.cp_async_commit_group()

            # ldmatrix + MMA
            tCsA_p = tCsA_s2r[None, None, None, read_stage]
            tCsB_p = tCsB_s2r[None, None, None, read_stage]
            cute.arch.cp_async_wait_group(self.num_stages - 1)
            cute.arch.sync_threads()

            for k_block in cutlass.range(BK // 16, unroll_full=True):
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, k_block],
                    tCrA_copy_view[None, None, k_block],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, k_block],
                    tCrB_copy_view[None, None, k_block],
                )
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

            write_stage = read_stage
            read_stage = (read_stage + 1) % self.num_stages

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # r2s
        tCrD = cute.make_fragment_like(tCrC, BFloat16)
        tCrD.store(tCrC.load().to(BFloat16))
        cute.autovec_copy(tCrD, tCsC)
        cute.arch.sync_threads()

        # s2r2g
        tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
        cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)
        cute.copy(tiled_copy_C, tCrC_epilogue, tCgC_epilogue)

    @cache
    @staticmethod
    def compile():
        M = cute.sym_int()
        N = cute.sym_int(divisibility=8)
        K = cute.sym_int(divisibility=8)

        A = cute.runtime.make_fake_tensor(BFloat16, (M, K), (cute.sym_int64(divisibility=8), 1), assumed_align=16)
        B = cute.runtime.make_fake_tensor(BFloat16, (N, K), (cute.sym_int64(divisibility=8), 1), assumed_align=16)
        C = cute.runtime.make_fake_tensor(BFloat16, (M, N), (cute.sym_int64(divisibility=8), 1), assumed_align=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        kernel = MatmulV1Kernel()
        return cute.compile(kernel, A, B, C, stream, options="--enable-tvm-ffi")


def cutedsl_v1(A: Tensor, B: Tensor):
    C = A.new_empty(A.shape[0], B.shape[1])
    MatmulV1Kernel.compile()(A, B.T, C)
    return C


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    C = cutedsl_v1(A, B.T)
    out_ref = A @ B.T
    torch.testing.assert_close(C, out_ref)
