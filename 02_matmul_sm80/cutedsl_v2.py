import math
from functools import cache
from typing import NamedTuple

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Uint32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import dsl_user_op
from torch import Tensor


@dsl_user_op
def mma_bf16(a: cute.TensorSSA, b: cute.TensorSSA, c: cute.TensorSSA, *, loc=None, ip=None):
    if a.element_type == BFloat16:
        a = cute.recast_tensor(a, Uint32)
    if b.element_type == BFloat16:
        b = cute.recast_tensor(b, Uint32)

    mlir_ty = Float32.mlir_type
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, Float32)


class MatmulV2Kernel(NamedTuple):
    warp_layout: tuple[int, int] = (2, 2)
    cta_tile: tuple[int, int, int] = (128, 128, 32)
    num_stages: int = 3

    @cute.jit
    def make_AB_layout(self, BM: int, BK: int, num_stages: int):
        # only go up to 128B
        swizzle_width = min(BK, 64)

        # canonical 128B swizzling is (3,4,3) on raw smem address.
        # for BF16, each elem = 2 bytes -> 128B swizzling is (3, 3, 3).
        swizzle_bits = int(math.log2(swizzle_width * 2 // 16))
        swizzle = cute.make_swizzle(swizzle_bits, 3, 3)

        slayout = cute.make_layout(
            (BM, (swizzle_width, BK // swizzle_width), num_stages),
            stride=(swizzle_width, (1, BM * swizzle_width), BM * BK),
        )
        return cute.make_composed_layout(swizzle, 0, slayout)

    @cute.jit
    def __call__(self, gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, stream: CUstream):
        M, N = gC.shape
        BM, BN, BK = self.cta_tile

        sA_layout = self.make_AB_layout(BM, BK, self.num_stages)
        sB_layout = self.make_AB_layout(BN, BK, self.num_stages)

        grid = (cute.ceil_div(M, BM), cute.ceil_div(N, BN), 1)
        block = (math.prod(self.warp_layout) * 32, 1, 1)

        self.kernel(
            gA,
            gB,
            gC,
            sA_layout,
            sB_layout,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        _, K = gA.shape
        BM, BN, BK = self.cta_tile
        num_warp_m, num_warp_n = self.warp_layout
        num_stages = self.num_stages

        WM = BM // num_warp_m
        WN = BN // num_warp_n
        warp_id_m = warp_id // num_warp_n
        warp_id_n = warp_id % num_warp_n

        # select input/output tiles
        gA_tiles = cute.local_tile(gA, tiler=(BM, BK), coord=(bid_m, None))  # (BM, BK, K/BK)
        gB_tiles = cute.local_tile(gB, tiler=(BN, BK), coord=(bid_n, None))  # (BN, BK, K/BK)

        # allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, sA_layout, 16)
        sB = smem.allocate_tensor(BFloat16, sB_layout, 16)

        # warp partition
        # shape: (WM, BK, num_stages)
        # (well, the 2nd mode, is actually (width,BK/width). but to simplify the expresion...)
        sA_warp = cute.logical_divide(sA, (WM, None, None))[(None, warp_id_m), None, None]
        sB_warp = cute.logical_divide(sB, (WN, None, None))[(None, warp_id_n), None, None]

        # pre-compute ldmatrix address (16x16 tile)
        # bring ldsm elems in front
        # (8, (WM,BK/8,num_stages))
        sA_ldsm = cute.zipped_divide(sA_warp, (1, 8, 1))[(0, None, 0), None]
        sB_ldsm = cute.zipped_divide(sB_warp, (1, 8, 1))[(0, None, 0), None]

        # partition to (16,16) tiles (appear as (16,2) lane tiles)
        # (8, ((16,WM/16), (2,BK/16), num_stages))
        sA_ldsm = cute.logical_divide(sA_ldsm, (None, (16, 2, None)))
        sB_ldsm = cute.logical_divide(sB_ldsm, (None, (16, 2, None)))

        # select the address
        # (8, WM/16, BK/16, num_stages)
        sA_ldsm = sA_ldsm[None, ((lane_id % 16, None), (lane_id // 16, None), None)]
        sB_ldsm = sB_ldsm[None, (((lane_id // 16) * 8 + (lane_id % 8), None), ((lane_id // 8) % 2, None), None)]

        # ldmatrix.x4
        ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
        ldsm_atom = cute.make_copy_atom(ldsm_op, BFloat16)

        # registers
        # let ptxas decides register reuse for rA and rB
        rA = cute.make_rmem_tensor((8, WM // 16, BK // 16), BFloat16)
        rB = cute.make_rmem_tensor(((4, 2), WN // 16, BK // 16), BFloat16)
        rC = cute.make_rmem_tensor((4, WN // 8, WM // 16), Float32)
        rC.fill(0.0)

        # prefetch
        for i in cutlass.range_constexpr(num_stages - 1):
            self.load_g2s(gA_tiles[None, None, i], sA[None, None, i], tid)
            self.load_g2s(gB_tiles[None, None, i], sB[None, None, i], tid)
            cute.arch.cp_async_commit_group()

        load_stage = num_stages - 1
        load_k = num_stages - 1
        compute_stage = 0

        num_iters = cute.ceil_div(K, BK)
        for iter_k in range(num_iters):
            load_k = iter_k + (num_stages - 1)
            if load_k < num_iters:
                cute.arch.sync_threads()
                self.load_g2s(gA_tiles[None, None, load_k], sA[None, None, load_stage], tid)
                self.load_g2s(gB_tiles[None, None, load_k], sB[None, None, load_stage], tid)
                load_stage = (load_stage + 1) % num_stages
            cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(num_stages - 1)
            cute.arch.sync_threads()

            for k in cutlass.range_constexpr(BK // 16):
                cute.copy(ldsm_atom, sA_ldsm[None, None, k, compute_stage], rA[None, None, k])
                cute.copy(ldsm_atom, sB_ldsm[None, None, k, compute_stage], rB[None, None, k])

                for m in cutlass.range_constexpr(WM // 16):
                    for n in cutlass.range_constexpr(WN // 16):
                        rC[None, n * 2 + 0, m] = mma_bf16(rA[None, m, k], rB[(None, 0), n, k], rC[None, n * 2 + 0, m])
                        rC[None, n * 2 + 1, m] = mma_bf16(rA[None, m, k], rB[(None, 1), n, k], rC[None, n * 2 + 1, m])

            compute_stage = (compute_stage + 1) % num_stages

        # epilogue
        cp_op = nvgpu.CopyUniversalOp()
        cp4B_atom = cute.make_copy_atom(cp_op, BFloat16, num_bits_per_copy=32)

        # create view into C gmem
        gC_cta = cute.local_tile(gC, tiler=(BM, BN), coord=(bid_m, bid_n))
        gC_warp = cute.local_tile(gC_cta, tiler=(WM, WN), coord=(warp_id_m, warp_id_n))

        # (2, (WM, WN/2))
        gC_warp = cute.zipped_divide(gC_warp, (1, 2))[(0, None), None]

        # (2, (((8,2), WM/16), (4,WN/8)))
        gC_view = cute.logical_divide(gC_warp, (None, (cute.make_layout((8, 2)), 4)))

        # (2, 2, WM/16, WN/8)
        gC_thr = gC_view[None, (((lane_id // 4, None), None), (lane_id % 4, None))]

        # explicit for loops to interleave cvt with st.global
        for m in cutlass.range_constexpr(WM // 16):
            for n in cutlass.range_constexpr(WN // 8):
                rC_bf16 = cute.make_rmem_tensor((2, 2), BFloat16)
                rC_bf16.store(rC[None, n, m].load().to(BFloat16))
                cute.copy(cp4B_atom, rC_bf16, gC_thr[None, None, m, n])

    @cute.jit
    def load_g2s(self, gA: cute.Tensor, sA: cute.Tensor, tid: Int32):
        # cp.async.cg
        op = cpasync.CopyG2SOp(nvgpu.LoadCacheMode.GLOBAL)
        atom = cute.make_copy_atom(op, BFloat16, num_bits_per_copy=128)

        tb_size = math.prod(self.warp_layout) * 32
        _, _, BK = self.cta_tile

        num_cols = BK // 8
        num_rows = tb_size // num_cols

        # (num_rows, 8, BM/num_rows)
        gA_view = cute.local_tile(gA, (num_rows, 8), (None, tid % num_cols))
        sA_view = cute.local_tile(sA, (num_rows, 8), (None, tid % num_cols))

        # (8, BM/num_rows)
        gA_view = gA_view[tid // num_cols, None, None]
        sA_view = sA_view[tid // num_cols, None, None]
        cute.copy(atom, gA_view, sA_view)

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

        kernel = MatmulV2Kernel()
        return cute.compile(kernel, A, B, C, stream, options="--enable-tvm-ffi")


def cutedsl_v2(A: Tensor, B: Tensor):
    C = A.new_empty(A.shape[0], B.shape[1])
    MatmulV2Kernel.compile()(A, B.T, C)
    return C
