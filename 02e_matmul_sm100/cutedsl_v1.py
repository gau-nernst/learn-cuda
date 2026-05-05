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
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import get_smem_capacity_in_bytes
from triton.testing import do_bench


@dsl_user_op
def tcgen05_mma_f16(
    d_tmem,
    a_desc,
    b_desc,
    idesc: cutlass.Constexpr,
    enable_input_d,
    *,
    loc=None,
    ip=None,
) -> None:
    nvvm.tcgen05_mma(
        nvvm.Tcgen05MMAKind.F16,
        nvvm.Tcgen05GroupKind.CTA_1,
        llvm.inttoptr(
            llvm.PointerType.get(cute.AddressSpace.tmem.value),
            Int32(d_tmem).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        ),
        Uint64(a_desc).ir_value(loc=loc, ip=ip),
        Uint64(b_desc).ir_value(loc=loc, ip=ip),
        Int32(idesc & 0xFFFF_FFFF).ir_value(loc=loc, ip=ip),
        Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_ld(taddr, shape, num, *, loc=None, ip=None):
    if shape == nvvm.Tcgen05LdStShape.SHAPE_32X32B:
        num_regs = num
    elif shape == nvvm.Tcgen05LdStShape.SHAPE_16X128B:
        num_regs = num * 2
    elif shape == nvvm.Tcgen05LdStShape.SHAPE_16X256B:
        num_regs = num * 4
    else:
        raise ValueError

    tmem_ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    tmem_ptr = llvm.inttoptr(tmem_ptr_ty, Int32(taddr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)

    if num_regs == 1:
        reg = nvvm.tcgen05_ld(Int32.mlir_type, shape, num, tmem_ptr, loc=loc, ip=ip)
        reg_f32 = llvm.bitcast(Float32.mlir_type, reg, loc=loc, ip=ip)
        return Float32(reg_f32)

    else:
        vec_i32_ty = ir.VectorType.get([num_regs], Int32.mlir_type, loc=loc)
        vec_f32_ty = ir.VectorType.get([num_regs], Float32.mlir_type, loc=loc)
        regs = nvvm.tcgen05_ld(vec_i32_ty, shape, num, tmem_ptr, loc=loc, ip=ip)
        regs_f32 = llvm.bitcast(vec_f32_ty, regs, loc=loc, ip=ip)
        return cute.TensorSSA(regs_f32, (num_regs,), Float32)


@dsl_user_op
def tcgen05_dealloc(*, loc=None, ip=None) -> None:
    tmem_ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    nvvm.tcgen05_dealloc(
        llvm.inttoptr(tmem_ptr_ty, Int32(0).ir_value(loc=loc, ip=ip), loc=loc, ip=ip),
        Int32(512).ir_value(loc=loc, ip=ip),
        group=nvvm.Tcgen05GroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def _stg_u32xN(
    tensor: cute.Tensor,
    coord: cute.Coord,
    values: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    modifier: cutlass.Constexpr[str] = "",
    *,
    loc=None,
    ip=None,
) -> None:
    base_ptr = (tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)).toint()
    value_operands = ", ".join(f"${i + 1}" for i in range(vec_size))
    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)]
        + [Uint32(values[i]).ir_value(loc=loc, ip=ip) for i in range(vec_size)],
        f"st.global{modifier}.v{vec_size}.u32 [$0], {{{value_operands}}};",
        ",".join(["l"] + ["r"] * vec_size),
        has_side_effects=True,
        is_align_stack=False,
    )


class MatmulKernel:
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
        # we must put num_stages as the last mode since tma_partition() uses the 1st mode
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)

        one_stage = cute.slice_(s_layout, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, A, one_stage, (BM, BK))
        return tma_atom, tma_tensor, s_layout

    @cute.jit
    def __call__(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream: CUstream):
        BM, BN, BK = self.cta_tile
        A_args = self.prepare_AB(A, BM, BK)
        B_args = self.prepare_AB(B, BN, BK)

        M, K = A.shape
        N, _ = B.shape
        grid_m = cute.ceil_div(M, BM)
        grid_n = cute.ceil_div(N, BN)
        self.kernel(A_args, B_args, C).launch(
            grid=(grid_m, grid_n),
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

        # select gmem tile
        gA_tile = cute.local_tile(A_tma_tensor, (BM, BK), (bid_m, None))  # [BM, BK, K/BK]
        gB_tile = cute.local_tile(B_tma_tensor, (BN, BK), (bid_n, None))  # [BM, BK, K/BK]
        tAsA, tAgA = cpasync.tma_partition(
            A_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_tile, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            B_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_tile, 0, 2),
        )

        if warp_id == 0:
            for i in cutlass.range_constexpr(self.num_stages):
                cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
            cute.arch.mbarrier_init(tmem_full_mbar, 1)
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma_atom)
            cpasync.prefetch_descriptor(B_tma_atom)
        cute.arch.sync_threads()

        # TMA warp
        if warp_id == 5:
            tma_stage = 0
            empty_phase = 1

            for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, empty_phase)

                mbar = tma_full_mbar + tma_stage
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, self.stage_size)
                cute.copy(A_tma_atom, tAgA[None, iter_k], tAsA[None, tma_stage], tma_bar_ptr=mbar)
                cute.copy(B_tma_atom, tBgB[None, iter_k], tBsB[None, tma_stage], tma_bar_ptr=mbar)

                tma_stage = (tma_stage + 1) % self.num_stages
                if tma_stage == 0:
                    empty_phase ^= 1

        # MMA warp
        elif warp_id == 4:
            cute.arch.alloc_tmem(512, taddr)

            tma_stage = 0
            full_phase = 0

            # BF16 MMA
            idesc = cutlass.const_expr((1 << 4) | (1 << 7) | (1 << 10) | (BN >> 3 << 17) | (BM >> 4 << 24))
            # 128B swizzling
            sdesc = cutlass.const_expr(((8 * 128) >> 4 << 32) | (1 << 46) | (2 << 61))

            for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, full_phase)
                nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)

                a_desc = sdesc | (sA[None, None, tma_stage].iterator.toint() >> 4)
                b_desc = sdesc | (sB[None, None, tma_stage].iterator.toint() >> 4)

                MMA_K = cutlass.const_expr(16)  # 32B

                with cute.arch.elect_one():
                    for k in cutlass.range_constexpr(BK // MMA_K):
                        tcgen05_mma_f16(0, a_desc, b_desc, idesc, iter_k > 0 or k > 0)
                        a_desc += 32 >> 4
                        b_desc += 32 >> 4
                    tcgen05.commit(tma_empty_mbar + tma_stage)

                tma_stage = (tma_stage + 1) % self.num_stages
                if tma_stage == 0:
                    full_phase ^= 1

            with cute.arch.elect_one():
                tcgen05.commit(tmem_full_mbar)

        # epilogue warps
        else:
            if warp_id == 0:
                cute.arch.mbarrier_wait(tmem_full_mbar, 0)
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)

            WIDTH = cutlass.const_expr(min(BN, 128))
            for i in cutlass.range_constexpr(BN // WIDTH):
                tmem = ((warp_id * 32) << 16) | (i * WIDTH)
                regs = tcgen05_ld(tmem, nvvm.Tcgen05LdStShape.SHAPE_32X32B, WIDTH)
                nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD)

                for j in cutlass.range_constexpr(WIDTH // 16):
                    tmp = cute.make_rmem_tensor(8, Uint32)
                    for k in cutlass.range_constexpr(8):
                        tmp[k] = _fp32x2_to_bf16x2(regs[j * 16 + k * 2], regs[j * 16 + k * 2 + 1])

                    coord = (bid_m * BM + tid, bid_n * BN + i * WIDTH + j * 16)
                    _stg_u32xN(C_tensor, coord, tmp, 8, ".relaxed.cta.L1::no_allocate")

            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            if warp_id == 0:
                tcgen05_dealloc()

    @cache
    @staticmethod
    def compile(BN: int):
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()
        A = cute.runtime.make_fake_tensor(BFloat16, (M, K), (K, 1), assumed_align=8)
        B = cute.runtime.make_fake_tensor(BFloat16, (N, K), (K, 1), assumed_align=8)
        C = cute.runtime.make_fake_tensor(BFloat16, (M, N), (N, 1), assumed_align=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = MatmulKernel(BN)
        return cute.compile(kernel, A, B, C, stream, options="--enable-tvm-ffi")


def cutedsl_v1(A: torch.Tensor, B: torch.Tensor):
    C = A.new_empty(A.shape[0], B.shape[1])
    MatmulKernel.compile(256)(A, B.T, C)
    return C


def main():
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    C_ref = A @ B.T

    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    kernel = MatmulKernel.compile(256)
    kernel(A, B, C)
    torch.cuda.synchronize()

    torch.testing.assert_close(C, C_ref)

    cublas_ms = do_bench(lambda: torch.mm(A, B.T))
    ours_ms = do_bench(lambda: kernel(A, B, C))

    cublas_tflops = 2 * M * N * K / (cublas_ms * 1e-3) * 1e-12
    ours_tflops = 2 * M * N * K / (ours_ms * 1e-3) * 1e-12
    print(f"{cublas_tflops=}")
    print(f"{ours_tflops=}")


if __name__ == "__main__":
    main()
