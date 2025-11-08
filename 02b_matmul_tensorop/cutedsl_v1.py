# https://github.com/NVIDIA/cutlass/blob/v4.2.1/examples/python/CuTeDSL/ampere/tensorop_gemm.py
# NOTE: nvidia-cutlass-dsl==4.2.1 (future version may break)

import math
from typing import NamedTuple

import cutlass
import torch
from cutlass import cute
from cutlass.cute.nvgpu import CopyUniversalOp, cpasync, warp
from torch import Tensor

WARP_SIZE = 32


def py_min(a, b):
    return min(a, b)


class Gemm(NamedTuple):
    ab_dtype: type[cutlass.Numeric]
    c_dtype: type[cutlass.Numeric]
    acc_dtype: type[cutlass.Numeric]
    atom_layout_mnk: tuple[int, int, int] = (2, 2, 1)  # warp tiling
    cta_tile: tuple[int, int, int] = (128, 128, 32)
    num_stages: int = 3

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        shape: tuple[int, int, int],
    ):
        # A: (M, K, L)
        # B: (N, K, L)
        # C: (M, N, L)
        M, N, K = shape
        K = cute.assume(K, 8)  # 128-bit alignment
        N = cute.assume(N, 8)
        mA = cute.make_tensor(a_ptr, cute.make_layout((M, K, 1), stride=(K, 1, M * K)))
        mB = cute.make_tensor(b_ptr, cute.make_layout((N, K, 1), stride=(K, 1, N * K)))
        mC = cute.make_tensor(c_ptr, cute.make_layout((M, N, 1), stride=(N, 1, M * N)))

        # TODO: understand how cutlass does swizzling
        BM, BN, BK = self.cta_tile
        num_threads = math.prod(self.atom_layout_mnk) * WARP_SIZE

        major_size = py_min(BK, 64)  # built-in min() will make major_size a dynamic value under @cute.jit() decorator
        swizzle_bits = py_min(int(math.log2(major_size * 16 // 128)), 3)
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            offset=0,
            outer=cute.make_layout((8, major_size), stride=(major_size, 1)),
        )
        sA_layout = cute.tile_to_shape(layout_atom, (BM, BK, self.num_stages), order=(0, 1, 2))
        sB_layout = cute.tile_to_shape(layout_atom, (BN, BK, self.num_stages), order=(0, 1, 2))

        swizzle_bits = py_min(int(math.log2(BN * self.c_dtype.width // 128)), 3)
        # TODO: what is this 3, 3 and 3, 4

        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4), offset=0, outer=cute.make_layout((8, BN), stride=(BN, 1))
        )
        sC_layout = cute.tile_to_shape(layout_atom, (BM, BN), order=(0, 1))

        # cp.async.cg 16
        cp_async_cg = cpasync.CopyG2SOp(cpasync.LoadCacheMode.GLOBAL)
        atom_cp_async = cute.make_copy_atom(cp_async_cg, self.ab_dtype, num_bits_per_copy=128)

        # thread-value layout: a 2D layout (T, V), where there are T threads and V values.
        # both T and V can be an n-D layout as well.
        # thread layout: formed from the 1st element of all threads
        # value layout: formed from all elements of each thread
        cp_elems = 128 // 16  # 8
        shape_dim1 = cute.size(BK) // cp_elems  # for bk=32, this is 4. TODO: why need cute.size()?
        # num_threads=128 in total. layout (32, 4) : (4, 1)
        thread_layout = cute.make_layout((num_threads // shape_dim1, shape_dim1), stride=(shape_dim1, 1))
        # each thread holds cp_elems contiguous elements
        value_layout = cute.make_layout((1, cp_elems))
        tiled_copy_AB = cute.make_tiled_copy_tv(atom_cp_async, thread_layout, value_layout)

        atom_cp = cute.make_copy_atom(CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128)
        cp_elems = 128 // self.c_dtype.width
        shape_dim1 = cute.size(BN) // cp_elems
        thread_layout = cute.make_layout((num_threads // shape_dim1, shape_dim1), stride=(shape_dim1, 1))
        value_layout = cute.make_layout((1, cp_elems))
        tiled_copy_C = cute.make_tiled_copy_tv(atom_cp, thread_layout, value_layout)

        # mma.m16n8k16
        atom_m, atom_n, _ = self.atom_layout_mnk
        op = warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, (16, 8, 16))
        tC = cute.make_layout(self.atom_layout_mnk)
        permutation_mnk = (atom_m * 16, atom_n * 16, 16)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk)

        grid_dim = cute.ceil_div(mC.shape, (self.cta_tile[0], self.cta_tile[1], 1))
        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        raster_remap_grid_dim = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            cute.size(grid_dim[2]),
        )

        kernel = self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_AB,
            tiled_copy_AB,
            tiled_copy_C,
            tiled_mma,
            raster_factor,
        )

        sA_size = cute.size_in_bytes(self.ab_dtype, sA_layout)
        sB_size = cute.size_in_bytes(self.ab_dtype, sB_layout)
        sC_size = cute.size_in_bytes(self.c_dtype, sC_layout)
        smem_size = max(sA_size + sB_size, sC_size)

        kernel.launch(grid=raster_remap_grid_dim, block=[num_threads, 1, 1], smem=smem_size)

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        raster_factor: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.cta_tile[0], self.cta_tile[1], 1))

        # remap bid
        offset_tile_x = bidx // raster_factor
        offset_tile_y = (bidx % raster_factor) + bidy * raster_factor

        # does this happen?
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass

        else:
            # select the output tile
            # TODO: understand this
            tile_coord = (offset_tile_x, offset_tile_y, None)
            gA = cute.local_tile(mA[None, None, bidz], tiler=self.cta_tile, coord=tile_coord, proj=(1, None, 1))
            gB = cute.local_tile(mB[None, None, bidz], tiler=self.cta_tile, coord=tile_coord, proj=(None, 1, 1))
            gC = cute.local_tile(mC[None, None, bidz], tiler=self.cta_tile, coord=tile_coord, proj=(1, 1, None))

            # check for the remainder of K % BLOCK_K. handle this tile first
            res_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.cta_tile[2]) * cute.size(gA, mode=[2])
            gA = cute.domain_offset((0, res_k, 0), gA)
            gB = cute.domain_offset((0, res_k, 0), gB)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            # TODO: what's this?
            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB.layout.shape)
            cA = cute.local_tile(mcA[None, None, bidz], tiler=self.cta_tile, coord=tile_coord, proj=(1, None, 1))
            cB = cute.local_tile(mcB[None, None, bidz], tiler=self.cta_tile, coord=tile_coord, proj=(None, 1, 1))
            cA = cute.domain_offset((0, res_k, 0), cA)
            cB = cute.domain_offset((0, res_k, 0), cB)

            # allocate shared memory
            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(self.ab_dtype, sA_layout, 16)
            sB = smem.allocate_tensor(self.ab_dtype, sB_layout, 16)
            sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout)  # overlap buffer

            thr_copy_A = tiled_copy_A.get_slice(tid)
            thr_copy_B = tiled_copy_B.get_slice(tid)
            thr_copy_C = tiled_copy_C.get_slice(tid)

            # TODO: what's this?
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)

            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)

            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            # predicate = mask?
            # https://github.com/NVIDIA/cutlass/blob/v4.2.1/media/docs/cpp/cute/0y_predication.md
            # TODO: why A uses tAgA but B uses tBsB?
            tApA = cute.make_fragment(
                cute.make_layout(
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                dtype=cutlass.Boolean,
            )
            tBpB = cute.make_fragment(
                cute.make_layout(
                    (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                dtype=cutlass.Boolean,
            )

            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])

            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])

            # zero these for predicated loads
            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()

            num_stages = cute.size(tAsA, mode=[3])
            k_tile_count = cute.size(tAgA, mode=[3])
            k_tile_index = cutlass.Int32(0)

            # prefetch the 1st possibly partial tile
            for k in range(tApA.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_A,
                        src=tAgA[None, None, k, k_tile_index],
                        dst=tAsA[None, None, k, 0],
                        pred=tApA[None, None, k],
                    )
            for k in range(tBpB.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_B,
                        src=tBgB[None, None, k, k_tile_index],
                        dst=tBsB[None, None, k, 0],
                        pred=tBpB[None, None, k],
                    )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            # prefetch other tiles
            for k_tile in range(1, num_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(
                    tiled_copy_A,
                    src=tAgA[None, None, None, k_tile_index],
                    dst=tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    src=tBgB[None, None, None, k_tile_index],
                    dst=tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            # smem "owned" by this thread (I think?)
            thr_mma = tiled_mma.get_slice(tid)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCsC = thr_mma.partition_C(sC)
            tCgC = thr_mma.partition_C(gC)

            # rmem for MMA
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            ldmatrix_m8n8_x4_b16 = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
            atom_copy_s2r_AB = cute.make_copy_atom(ldmatrix_m8n8_x4_b16, mA.element_type)
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_AB, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_AB, tiled_mma)

            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tid)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tid)

            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            smem_pipe_read = 0
            smem_pipe_write = num_stages - 1

            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

            # prefetch smem->rmem pipeline
            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, src=tCsA_p[None, None, 0], dst=tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, src=tCsB_p[None, None, 0], dst=tCrB_copy_view[None, None, 0])

            # main loop
            # TODO: try remove rmem pipelining to make the logic simpler? it might not be necessary as
            # the compiler will re-arrange the instructions anyway...
            for k_tile in range(k_tile_count):
                # rmem pipeline
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_stages - 2)
                        cute.arch.sync_threads()

                    # prefetch AB smem->rmem for next rmem pipeline stage
                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(
                        tiled_copy_s2r_A,
                        src=tCsA_p[None, None, k_block_next],
                        dst=tCrA_copy_view[None, None, k_block_next],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        src=tCsB_p[None, None, k_block_next],
                        dst=tCrB_copy_view[None, None, k_block_next],
                    )

                    # prefetch A gmem->smem
                    if k_block == 0:
                        if k_tile + num_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_A,
                                src=tAgA[None, None, None, k_tile_index],
                                dst=tAsA[None, None, None, smem_pipe_write],
                                pred=tApA,
                            )

                    # MMA
                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                    # prefetch B gmem->smem
                    if k_block == 0:
                        if k_tile + num_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_B,
                                src=tBgB[None, None, None, k_tile_index],
                                dst=tBsB[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )

                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.sync_threads()

            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            tCrD[None] = tCrC.load().to(self.c_dtype)
            cute.autovec_copy(tCrD, tCsC)

            bm, bn, _ = self.cta_tile
            ceilM, ceilN, _ = cute.ceil_div(mC.shape, (bm, bn, 1))
            mcC = cute.make_identity_tensor((cute.size(ceilM) * bm, cute.size(ceilN) * bn, 1))
            cC = cute.local_tile(mcC[None, None, bidz], tiler=self.cta_tile, coord=tile_coord, proj=(1, 1, None))
            tCcC = thr_copy_C.partition_S(cC)

            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.arch.sync_threads()
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

            tCpC = cute.make_fragment(
                cute.make_layout(
                    (tCgC_epilogue.shape[0][1], cute.size(tCgC_epilogue, mode=[1]), cute.size(tCgC_epilogue, mode=[2])),
                    stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
                ),
                dtype=cutlass.Boolean,
            )
            for rest_v in range(tCpC.shape[0]):
                for m in range(tCpC.shape[1]):
                    tCpC[rest_v, m, 0] = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mC.shape[0])

            for rest_v in range(tCpC.shape[0]):
                for n in range(tCpC.shape[2]):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                        cute.copy(
                            tiled_copy_C,
                            src=tCrC_epilogue[None, None, n],
                            dst=tCgC_epilogue[None, None, n],
                            pred=tCpC[None, None, n],
                        )


def _compile_kernel():
    # cutedsl will throw error if CUDA context is not initialized in the current process...
    # torch.cuda.init() doesn't work...
    torch.randn(0, device="cuda")
    A_ptr, B_ptr, C_ptr = [
        cute.runtime.make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16) for _ in range(3)
    ]
    M, N, K = 1024, 1024, 1024
    gemm = Gemm(ab_dtype=cutlass.BFloat16, c_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32)
    return cute.compile(gemm, A_ptr, B_ptr, C_ptr, (M, N, K))


compiled_kernel = _compile_kernel()


def to_ptr(x: Tensor):
    return cute.runtime.make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def cutedsl_v1(A: Tensor, B: Tensor):
    M, K = A.shape
    _, N = B.shape
    assert A.stride() == (K, 1)
    assert B.stride() == (1, K)

    C = A.new_empty(M, N)
    compiled_kernel(to_ptr(A), to_ptr(B), to_ptr(C), (M, N, K))
    return C


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    C = cutedsl_v1(A, B.T)
    out_ref = A @ B.T
    torch.testing.assert_close(C, out_ref)
