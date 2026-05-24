import cutlass
from cutlass import Boolean, Float32, Int32, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op

NVVM_CTA_GROUP_MAP = [
    None,
    nvvm.Tcgen05GroupKind.CTA_1,
    nvvm.Tcgen05GroupKind.CTA_2,
]


def simple_tma_g2s(atom, src, dst, mbar):
    """A simple helper that wraps group_modes() and tma_partition()
    NOTE: this should be called WITHOUT cute.elect_one()
    """
    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(dst, 0),
        cute.group_modes(src, 0),
    )
    cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar)


def _make_tmem_llvm_ptr(taddr, *, loc=None, ip=None):
    tmem_ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    return llvm.inttoptr(tmem_ptr_ty, Int32(taddr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


# name this _tcgen05 to avoid name collision with cutlass.cute.nvgpu.tcgen05
class _tcgen05:
    @dsl_user_op
    @staticmethod
    def alloc(
        taddr: cute.Pointer,
        cta_group: int = 1,
        *,
        loc=None,
        ip=None,
    ) -> None:
        nvvm.tcgen05_alloc(
            taddr.to_llvm_ptr(loc=loc, ip=ip),
            Uint32(512).ir_value(loc=loc, ip=ip),
            group=NVVM_CTA_GROUP_MAP[cta_group],
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    @staticmethod
    def dealloc(
        cta_group: int = 1,
        *,
        loc=None,
        ip=None,
    ) -> None:
        nvvm.tcgen05_dealloc(
            _make_tmem_llvm_ptr(0, loc=loc, ip=ip),
            Int32(512).ir_value(loc=loc, ip=ip),
            group=NVVM_CTA_GROUP_MAP[cta_group],
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    @staticmethod
    def mma_f16(
        d_tmem,
        a_desc,
        b_desc,
        idesc,
        enable_input_d,
        cta_group: int = 1,
        *,
        loc=None,
        ip=None,
    ) -> None:
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            NVVM_CTA_GROUP_MAP[cta_group],
            _make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    @staticmethod
    def commit(mbar, cta_mask=None, cta_group: int = 1, *, loc=None, ip=None):
        mbar_llvm = mbar.to_llvm_ptr(loc=loc, ip=ip)
        group = NVVM_CTA_GROUP_MAP[cta_group]
        if cutlass.const_expr(cta_mask is not None):
            nvvm.tcgen05_commit_arrive(
                mbar_llvm,
                multicast_mask=cta_mask.ir_value(loc=loc, ip=ip),
                group=group,
                loc=loc,
                ip=ip,
            )
        else:
            nvvm.tcgen05_commit_arrive(mbar_llvm, group=group, loc=loc, ip=ip)

    @dsl_user_op
    @staticmethod
    def ld(taddr, shape: str, num: int, *, loc=None, ip=None):
        if shape == "32x32b":
            nvvm_shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
            num_regs = num
        elif shape == "16x128b":
            nvvm_shape = nvvm.Tcgen05LdStShape.SHAPE_16X128B
            num_regs = num * 2
        elif shape == "16x256b":
            nvvm_shape = nvvm.Tcgen05LdStShape.SHAPE_16X256B
            num_regs = num * 4
        else:
            raise ValueError

        tmem_ptr = _make_tmem_llvm_ptr(taddr, loc=loc, ip=ip)

        if num_regs == 1:
            reg = nvvm.tcgen05_ld(Int32.mlir_type, nvvm_shape, num, tmem_ptr, loc=loc, ip=ip)
            reg_f32 = llvm.bitcast(Float32.mlir_type, reg, loc=loc, ip=ip)
            return Float32(reg_f32)

        else:
            vec_i32_ty = ir.VectorType.get([num_regs], Int32.mlir_type, loc=loc)
            vec_f32_ty = ir.VectorType.get([num_regs], Float32.mlir_type, loc=loc)
            regs = nvvm.tcgen05_ld(vec_i32_ty, nvvm_shape, num, tmem_ptr, loc=loc, ip=ip)
            regs_f32 = llvm.bitcast(vec_f32_ty, regs, loc=loc, ip=ip)
            return cute.TensorSSA(regs_f32, (num_regs,), Float32)

    @dsl_user_op
    @staticmethod
    def fence_after_thread_sync(*, loc=None, ip=None):
        nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)

    @dsl_user_op
    @staticmethod
    def fence_before_thread_sync(*, loc=None, ip=None):
        nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)
