#include "common.h"

#include <cuda_bf16.h>
#include <torch/library.h>
#include <ATen/ATen.h>

constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;

template <int BLOCK_N, int BLOCK_K>
__global__
__launch_bounds__(TB_SIZE)
void matmul_tmem_kernel(
  const nv_bfloat16 *A_ptr,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = __cvta_generic_to_shared(smem_ptr);
  constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);

  const int tma_mbar_addr = smem + B_size;
  const int mma_mbar_addr = tma_mbar_addr + 8;
  const int taddr         = mma_mbar_addr + 8;

  if (warp_id == 0 && elect_sync()) {
    // only 1 thread issue
    mbarrier_init(tma_mbar_addr, 1);
    mbarrier_init(mma_mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem for output
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(taddr), "r"(512));
  }
  __syncthreads();  // visible to all threads

  const int num_iters = K / BLOCK_K;
  int parity = 0;

  const int acc_tmem = 0;
  const int a_tmem_base = acc_tmem + BLOCK_N;

  for (int iter_k = 0; iter_k < num_iters; iter_k++) {
    const int off_k = iter_k * BLOCK_K;

    // issue TMA for B
    if (warp_id == 0 && elect_sync()) {
      tma_3d_g2s(smem, &B_tmap, 0, off_n, off_k / 64, tma_mbar_addr);
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(tma_mbar_addr), "r"(B_size) : "memory");
    }

    // load A from gmem->rmem->tmem
    // since A tile is [BLOCK_M, BLOCK_K], we will use BLOCK_K/2=32 tmem columns
    // each iteration loads 16 BF16 elems
    for (int i = 0; i < BLOCK_K / 16; i++) {
      auto *src_gmem = A_ptr + (off_m + tid) * K + (off_k + i * 16);
      int dst_tmem = ((warp_id * 32) << 16) | (a_tmem_base + i * 8);
      asm volatile(
        "{\n"
        ".reg .b32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
        "ld.global.v8.b32 {a0, a1, a2, a3, a4, a5, a6, a7}, [%1];\n"
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {a0, a1, a2, a3, a4, a5, a6, a7};"
        "}"
        :: "r"(dst_tmem), "l"(src_gmem)
      );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
    asm volatile("tcgen05.fence::before_thread_sync;");  // make tcgen05.st visible to tcgen05.mma
    __syncthreads();  // all threads finish loading A

    // issue MMA
    if (warp_id == 0 && elect_sync()) {
      mbarrier_wait(tma_mbar_addr, parity);
      asm volatile("tcgen05.fence::after_thread_sync;");  // make tcgen05.mma see TMA load and tcgen05.st

      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
      constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                                | (1U << 7U)   // atype=BF16
                                | (1U << 10U)  // btype=BF16
                                | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                                | ((uint32_t)BLOCK_M >> 4U << 24U)  // MMA_M
                                ;

      auto make_desc = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };

      // k1 selects the (BLOCK_M, 64) tile.
      // k2 selects the (BLOCK_M, 16) tile, whose rows are swizzled.
      for (int k1 = 0; k1 < BLOCK_K / 64; k1++)
        for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
          int a_tmem = a_tmem_base + k1 * 32 + k2 * 8;
          uint64_t b_desc = make_desc(smem + k1 * BLOCK_N * 128 + k2 * 32);
          int enable_input_d = (iter_k > 0) || (k1 > 0) || (k2 > 0);
          tcgen05_mma_f16_tmem(acc_tmem, a_tmem, b_desc, i_desc, enable_input_d);
        }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mma_mbar_addr) : "memory");
    }

    // wait MMA
    mbarrier_wait(mma_mbar_addr, parity);

    parity ^= 1;
  }

  // PTX doc says we need to add this before tcgen05.ld, after tcgen05.mma
  asm volatile("tcgen05.fence::after_thread_sync;");

  // load 8 columns from tmem at a time -> store 16 bytes per thread to smem
  // (still strided though)
  for (int n = 0; n < BLOCK_N / 8; n++) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d
    // Layout D
    float tmp[8];
    const int addr = ((warp_id * 32) << 16) | (acc_tmem + n * 8);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                  "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    nv_bfloat162 out[4];
    for (int i = 0; i < 4; i++)
      out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});

    // uncoalesced writes weeee
    nv_bfloat16 *out_ptr = C_ptr + (off_m + tid) * N + (off_n + n * 8);
    reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
  }
  __syncthreads();  // all threads finish reading data from tmem
  if (warp_id == 0)  // deallocate tmem
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
}

template <int BLOCK_N, int BLOCK_K>
void matmul_tmem_launch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  CUtensorMap B_tmap;
  init_tmap_3d_128B(&B_tmap, B_ptr, N, K, BLOCK_N, BLOCK_K);

  int grid = (M / BLOCK_M) * (N / BLOCK_N);
  int smem_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);
  smem_size += 2 * 8 + 4;  // mbars and taddr

  auto this_kernel = matmul_tmem_kernel<BLOCK_N, BLOCK_K>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  this_kernel<<<grid, TB_SIZE, smem_size>>>(A_ptr, B_tmap, C_ptr, M, N, K);
}

at::Tensor matmul_tmem(const at::Tensor& A, const at::Tensor& B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  auto C = at::empty({M, N}, A.options());
  // auto C = at::zeros({M, N}, A.options());  // for correctness check, use this

  auto *A_ptr = reinterpret_cast<nv_bfloat16 *>(A.data_ptr());
  auto *B_ptr = reinterpret_cast<nv_bfloat16 *>(B.data_ptr());
  auto *C_ptr = reinterpret_cast<nv_bfloat16 *>(C.data_ptr());

  matmul_tmem_launch<128, 64>(A_ptr, B_ptr, C_ptr, M, N, K);
  return C;
}

TORCH_LIBRARY(my_matmul, m) {
  m.def("matmul_tmem(Tensor A, Tensor B) -> Tensor", &matmul_tmem);
}
