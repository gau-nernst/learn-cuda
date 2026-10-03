from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import dsl_user_op

TORCH_TO_CUTE_DTYPE = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
    torch.int8: cutlass.Int8,
    torch.uint8: cutlass.Uint8,
    torch.int32: cutlass.Int32,
    torch.uint32: cutlass.Uint32,
}

CUTE_TO_PTX_DTYPE = {
    cutlass.Float32: "f32",
    cutlass.BFloat16: "bf16",
    cutlass.Float16: "f16",
    cutlass.Float8E4M3FN: "e4m3",
    cutlass.Float8E5M2: "e5m2",
    cutlass.Int8: "s8",
    cutlass.Uint8: "u8",
    cutlass.Int32: "s32",
    cutlass.Uint32: "u32",
}


def simple_tma_g2s(atom, src, dst, mbar, cache_policy=None):
    """A simple helper that wraps group_modes() and tma_partition()
    NOTE: this should be called WITHOUT cute.elect_one()
    """
    dst = cute.group_modes(dst, 0)
    src = cute.group_modes(src, 0)
    s_part, g_part = cpasync.tma_partition(atom, 0, cute.make_layout(1), dst, src)
    cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar, cache_policy=cache_policy)


@dsl_user_op
def permute(x: cute.Tensor, dims: tuple[int, ...], *, loc=None, ip=None):
    layout = cute.select(x.layout, mode=dims, loc=loc, ip=ip)
    return cute.make_tensor(x.iterator, layout, loc=loc, ip=ip)


@dsl_user_op
def mma_sync(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, *, loc=None, ip=None):
    # CuteDSL has nvvm.mma_sync(), but it doesn't export nvvm's MMA shape.
    # hence we have to use PTX.
    a_ty = CUTE_TO_PTX_DTYPE[a.element_type]
    b_ty = CUTE_TO_PTX_DTYPE[b.element_type]
    c_ty = CUTE_TO_PTX_DTYPE[c.element_type]
    mlir_ty = c.element_type.mlir_type
    K = 256 // a.element_type.width  # 32B

    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)

    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        f"mma.sync.aligned.m16n8k{K}.row.col.{c_ty}.{a_ty}.{b_ty}.{c_ty} "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
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
    return cute.TensorSSA(vec, 4, c.element_type)


class Sm120Attn:
    DIM: int = 128
    BQ: int = 64
    BK: int = 64
    num_stages: int = 3

    def __init__(self, num_heads: int):
        self.num_heads = num_heads

    @cute.jit
    def prepare_tma(
        self,
        x: cute.Tensor,
        BLOCK: cutlass.Constexpr[int],
        num_stages: cutlass.Constexpr[int],
        tma_op: cpasync.TmaCopyOp,
    ):
        # x: [B, L, H, D]
        swizzle = cute.make_swizzle(3, 4, 3)  # 128B
        s_layout = cute.make_layout(
            (1, BLOCK, 1, (64, self.DIM // 64), num_stages),
            stride=(0, 64, 0, (1, BLOCK * 64), BLOCK * self.DIM),
        )
        s_layout = cute.make_composed_layout(swizzle, 0, s_layout)
        return cpasync.make_tiled_tma_atom(tma_op, x, s_layout, (1, BLOCK, 1, self.DIM))

    @cute.jit
    def __call__(self, gQ: cute.Tensor, gK: cute.Tensor, gV: cute.Tensor, gO: cute.Tensor, stream: CUstream):
        B, Lq, _, _ = gQ.shape

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        Q_tma = self.prepare_tma(gQ, self.BQ, self.num_stages, tma_g2s)
        K_tma = self.prepare_tma(gK, self.BK, self.num_stages, tma_g2s)
        V_tma = self.prepare_tma(gV, self.BK, 1, tma_g2s)

        grid = (cute.ceil_div(Lq, self.BQ), self.num_heads, B)
        block = (5 * 32, 1, 1)
        self.kernel(Q_tma, K_tma, V_tma, gO).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(self, Q_tma: cpasync.TmaInfo, K_tma: cpasync.TmaInfo, V_tma: cpasync.TmaInfo, gO: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        q_tile_id, head_id, batch_id = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        Lk = K_tma.tma_tensor.shape[1]
        BQ = self.BQ
        BK = self.BK
        DIM = self.DIM
        WQ = BQ // 4
        num_stages = self.num_stages

        # allocate smem
        def allocate_smem(smem, s_layout):
            return smem.allocate_tensor(BFloat16, s_layout.outer, byte_alignment=128, swizzle=s_layout.inner)

        # K and V share the same smem slots
        smem = cutlass.utils.SmemAllocator()
        sK = allocate_smem(smem, K_tma.smem_layout)[0, None, 0, None, None]
        sV = cute.make_tensor(sK.iterator, V_tma.smem_layout.outer)[0, None, 0, None, None]

        # alias
        sQ = cute.make_tensor(sK.iterator, Q_tma.smem_layout.outer)[0, None, 0, None, 0]

        tma_q_full_mbar = smem.allocate_array(Int64, 1)
        tma_q_empty_mbar = smem.allocate_array(Int64, 1)
        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        BAR_MMA = 1

        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tma_q_full_mbar, 1)
                cute.arch.mbarrier_init(tma_q_empty_mbar, 128)
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(Q_tma.atom)
            cpasync.prefetch_descriptor(K_tma.atom)
            cpasync.prefetch_descriptor(V_tma.atom)
        cute.arch.sync_threads()

        if warp_id == 4:
            # TMA warp
            # load Q
            gQ_tile = cute.local_tile(Q_tma.tma_tensor[batch_id, None, head_id, None], (BQ, DIM), (q_tile_id, 0))
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(tma_q_full_mbar, BQ * DIM * 2)
            simple_tma_g2s(Q_tma.atom, gQ_tile, sQ, tma_q_full_mbar)
            cute.arch.mbarrier_wait(tma_q_empty_mbar, 0)  # wait for Q to finish

            # for KV
            stage_id = 0
            parity = 1

            # [block, head_dim, L/block]
            gK_tiles = cute.local_tile(K_tma.tma_tensor[batch_id, None, head_id, None], (BK, DIM), (None, 0))
            gV_tiles = cute.local_tile(V_tma.tma_tensor[batch_id, None, head_id, None], (BK, DIM), (None, 0))
            k_size = cutlass.const_expr(self.BK * self.DIM * 2)

            for iter_l in range(cute.ceil_div(Lk, BK)):
                mbar = tma_full_mbar + stage_id
                cute.arch.mbarrier_wait(tma_empty_mbar + stage_id, parity)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, k_size)
                simple_tma_g2s(K_tma.atom, gK_tiles[None, None, iter_l], sK[None, None, stage_id], mbar)
                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

                mbar = tma_full_mbar + stage_id
                cute.arch.mbarrier_wait(tma_empty_mbar + stage_id, parity)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, k_size)
                simple_tma_g2s(V_tma.atom, gV_tiles[None, None, iter_l], sV[None, None, stage_id], mbar)
                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        else:
            # MMA warps
            # ldmatrix.x4
            ldsm_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), BFloat16)
            ldsm_trans_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16)

            rQ = cute.make_rmem_tensor((8, WQ // 16, DIM // 16), BFloat16)
            rK = cute.make_rmem_tensor(((4, 2), BK // 16, DIM // 16), BFloat16)
            rS = cute.make_rmem_tensor((4, BK // 8, WQ // 16), Float32)
            rM = cute.make_rmem_tensor((2, WQ // 16), Float32)
            sumexp = cute.make_rmem_tensor((2, WQ // 16), Float32)
            rM.fill(float("-inf"))
            sumexp.fill(0.0)

            # TODO: experiment with register layout
            rP = cute.make_rmem_tensor((8, BK // 16, WQ // 16), BFloat16)
            rV = cute.make_rmem_tensor(((4, 2), BK // 16, DIM // 16), BFloat16)
            rO = cute.make_rmem_tensor((4, DIM // 8, WQ // 16), Float32)
            rO.fill(0.0)

            sQ_warp = cute.local_tile(sQ, (WQ, DIM), (warp_id, 0))
            sQ_ldsm = cute.zipped_divide(sQ_warp, (16, cute.make_layout((8, 2))))  # ((16,(8,2)), (WQ/16,DIM/16))
            sQ_ldsm = sQ_ldsm[(lane_id % 16, (None, lane_id // 16)), (None, None)]  # (8, WQ/16, DIM/16)

            # wait for and load Q
            if warp_id == 0:
                cute.arch.mbarrier_wait(tma_q_full_mbar, 0)
            cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)
            cute.copy(ldsm_atom, sQ_ldsm, rQ)
            cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)
            cute.arch.mbarrier_arrive(tma_q_empty_mbar)

            # for KV
            stage_id = 0
            parity = 0

            sK_ldsm = cute.zipped_divide(sK, (16, cute.make_layout((8, 2))))  # ((16,(8,2)), (BK/16,DIM/16,num_stages))
            sK_ldsm = sK_ldsm[
                ((lane_id // 16) * 8 + (lane_id % 8), (None, (lane_id // 8) % 2)), None
            ]  # (8, (BK/16,DIM/16, num_stages))

            sV_ldsm = cute.zipped_divide(sV, (16, cute.make_layout((8, 2))))  # ((16,(8,2)), (BK/16,DIM/16,num_stages))
            sV_ldsm = sV_ldsm[(lane_id % 16, (None, lane_id // 16)), None]  # (8, (BK/16,DIM/16,num_stages))

            # sqrt(dim) / ln(2)
            sm_scale = cutlass.const_expr(DIM ** (-0.5) * 1.4426950408889634)

            for iter_l in range(cute.ceil_div(Lk, BK)):
                rS.fill(0.0)
                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + stage_id, parity)
                cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)

                # S = Q @ K.T
                for k in cutlass.range_constexpr(DIM // 16):
                    cute.copy(ldsm_atom, sK_ldsm[None, (None, k, stage_id)], rK[None, None, k])
                    for m in cutlass.range_constexpr(WQ // 16):
                        for n in cutlass.range_constexpr(BK // 8):
                            rS[None, n, m] = mma_sync(rQ[None, m, k], rK[(None, n % 2), n // 2, k], rS[None, n, m])

                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                cute.arch.mbarrier_arrive(tma_empty_mbar + stage_id)
                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

                # online softmax
                for m in cutlass.range_constexpr(WQ // 16):
                    for n in cutlass.range_constexpr(BK // 8):
                        for i in cutlass.range_constexpr(4):
                            rS[i, n, m] *= sm_scale

                    # new rowmax
                    rM_new0 = rM[0, m]
                    rM_new1 = rM[1, m]
                    for n in cutlass.range_constexpr(BK // 8):
                        rM_new0 = cute.arch.fmax(rM_new0, cute.arch.fmax(rS[0, n, m], rS[1, n, m]))
                        rM_new1 = cute.arch.fmax(rM_new1, cute.arch.fmax(rS[2, n, m], rS[3, n, m]))

                    # butterfly reduction within 4 threads
                    for i in cutlass.range_constexpr(2):
                        other0 = cute.arch.shuffle_sync_bfly(rM_new0, 1 << i)
                        other1 = cute.arch.shuffle_sync_bfly(rM_new1, 1 << i)
                        rM_new0 = cute.arch.fmax(rM_new0, other0)
                        rM_new1 = cute.arch.fmax(rM_new1, other1)

                    # rescale previous O
                    rescale0 = cute.exp2(rM[0, m] - rM_new0, fastmath=True)
                    rescale1 = cute.exp2(rM[1, m] - rM_new1, fastmath=True)
                    for n in cutlass.range_constexpr(DIM // 8):
                        rO[0, n, m] *= rescale0
                        rO[1, n, m] *= rescale0
                        rO[2, n, m] *= rescale1
                        rO[3, n, m] *= rescale1

                    # save the new rowmax
                    rM[0, m] = rM_new0
                    rM[1, m] = rM_new1

                    # rowsumexp
                    sumexp_new = cute.make_rmem_tensor(2, Float32)
                    sumexp_new.fill(0.0)

                    for n in cutlass.range_constexpr(BK // 8):
                        rS[0, n, m] = cute.exp2(rS[0, n, m] - rM[0, m], fastmath=True)
                        rS[1, n, m] = cute.exp2(rS[1, n, m] - rM[0, m], fastmath=True)
                        rS[2, n, m] = cute.exp2(rS[2, n, m] - rM[1, m], fastmath=True)
                        rS[3, n, m] = cute.exp2(rS[3, n, m] - rM[1, m], fastmath=True)

                        sumexp_new[0] += rS[0, n, m] + rS[1, n, m]
                        sumexp_new[1] += rS[2, n, m] + rS[3, n, m]

                    # pack to BF16 for P
                    rP[None, None, m].store(rS[None, None, m].load().to(BFloat16))

                    # butterfly reduction within 4 threads
                    for i in cutlass.range_constexpr(2):
                        sumexp_new[0] += cute.arch.shuffle_sync_bfly(sumexp_new[0], 1 << i)
                        sumexp_new[1] += cute.arch.shuffle_sync_bfly(sumexp_new[1], 1 << i)

                    sumexp[0, m] = sumexp[0, m] * rescale0 + sumexp_new[0]
                    sumexp[1, m] = sumexp[1, m] * rescale1 + sumexp_new[1]

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + stage_id, parity)
                cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)

                # O += P @ V
                for k in cutlass.range_constexpr(BK // 16):
                    cute.copy(ldsm_trans_atom, sV_ldsm[None, (k, None, stage_id)], rV[None, k, None])
                    for m in cutlass.range_constexpr(WQ // 16):
                        for n in cutlass.range_constexpr(DIM // 8):
                            rO[None, n, m] = mma_sync(rP[None, k, m], rV[(None, n % 2), k, n // 2], rO[None, n, m])

                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                cute.arch.mbarrier_arrive(tma_empty_mbar + stage_id)
                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

            gO_view = cute.local_tile(gO[batch_id, None, head_id, None], (WQ, DIM), (q_tile_id * 4 + warp_id, 0))
            gO_view = permute(gO_view, (1, 0))  # [DIM, WQ]
            gO_view = cute.zipped_divide(
                gO_view, (cute.make_layout((2, 4)), cute.make_layout((8, 2)))
            )  # (((2,4),(8,2)), (DIM/8,WQ/16))
            gO_view = gO_view[((None, lane_id % 4), (lane_id // 4, None)), None]  # (2, 2, (DIM/8,WQ/16))
            gO_view = cute.group_modes(gO_view, 0, 2)  # ((2,2), (DIM/8,WQ/16))

            st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32)

            for m in cutlass.range_constexpr(WQ // 16):
                sumexp[0, m] = cute.arch.rcp_approx(sumexp[0, m])
                sumexp[1, m] = cute.arch.rcp_approx(sumexp[1, m])

                for n in cutlass.range_constexpr(DIM // 8):
                    rO[0, n, m] *= sumexp[0, m]
                    rO[1, n, m] *= sumexp[0, m]
                    rO[2, n, m] *= sumexp[1, m]
                    rO[3, n, m] *= sumexp[1, m]

                    tmp = cute.make_rmem_tensor(4, BFloat16)
                    tmp.store(rO[None, n, m].load().to(BFloat16))
                    cute.copy(st_atom, tmp, gO_view[None, (n, m)])

    @cache
    @staticmethod
    def compile(num_heads: int):
        B = cute.sym_int()
        Lq = cute.sym_int()
        Lk = cute.sym_int()
        D = Sm120Attn.DIM

        def _make_tensor(L):
            shape = (B, L, num_heads, D)
            stride = (cute.sym_int64(16), cute.sym_int64(16), cute.sym_int64(16), 1)
            return make_fake_tensor(BFloat16, shape, stride, assumed_align=16)

        Q = _make_tensor(Lq)
        K = _make_tensor(Lk)
        V = _make_tensor(Lk)
        O = _make_tensor(Lq)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm120Attn(num_heads)
        return cute.compile(kernel, Q, K, V, O, stream, options="--enable-tvm-ffi")


def attn_v6(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    O = torch.empty_like(Q)
    Sm120Attn.compile(Q.shape[1])(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), O.transpose(1, 2))
    return O
