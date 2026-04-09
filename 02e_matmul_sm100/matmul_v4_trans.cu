#include "common.h"

#include <cuda_bf16.h>
#include <torch/library.h>
#include <ATen/ATen.h>

constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__
__launch_bounds__(TB_SIZE)
void matmul_v4_trans_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
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
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);

  // set up mbarrier and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  1 mbar for epilgoue
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
  __shared__ int tmem_addr[1];  // tmem address is 32-bit
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  if (warp_id == 0 && elect_sync()) {
    // only 1 thread issue
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem for output
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(BLOCK_N));
  }

  __syncthreads();  // visible to all threads
  const int taddr = tmem_addr[0];  // this will be 0

  int phase = 0;

  const int num_iters = K / BLOCK_K;
  if (warp_id == 0 && elect_sync()) {
    // TMA warp
    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      // wait for MMA
      // the initial TMA phase is 0, and it is available. so we wait for 1 instead.
      const int stage_id = iter_k % NUM_STAGES;
      mbarrier_wait(mma_mbar_addr + stage_id * 8, phase ^ 1);

      // flip phase when we have cycled through all TMA buffers
      if (stage_id == NUM_STAGES - 1)
        phase ^= 1;

      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * (A_size + B_size);
      const int B_smem = A_smem + A_size;

      // original layout: [K, M]
      // permute:         [K/8, M/64, 8, 64]
      const int off_k = iter_k * BLOCK_K;
      tma_4d_g2s(A_smem, &A_tmap, 0, 0, off_m / 64, off_k / 8, mbar_addr);
      tma_4d_g2s(B_smem, &B_tmap, 0, 0, off_n / 64, off_k / 8, mbar_addr);
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(A_size + B_size) : "memory");
    }
  }
  else if (warp_id == 1 && elect_sync()) {
    // MMA warp
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                              | (1U << 7U)   // atype=BF16
                              | (1U << 10U)  // btype=BF16
                              | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                              | ((uint32_t)BLOCK_M >> 4U << 24U)  // MMA_M
                              | (1U << 15U)  // trans A
                              | (1U << 16U)  // trans B
                              ;

    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      // wait for TMA
      const int stage_id = iter_k % NUM_STAGES;
      mbarrier_wait(tma_mbar_addr + stage_id * 8, phase);
      asm volatile("tcgen05.fence::after_thread_sync;");  // (why) do we need this? from DeepGEMM

      // flip phase when we have cycled through all TMA buffers
      if (stage_id == NUM_STAGES - 1)
        phase ^= 1;

      const int A_smem = smem + stage_id * (A_size + B_size);
      const int B_smem = A_smem + A_size;

      // set up shared memory descriptors for A and B
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
      auto make_desc = [](int addr, int BLOCK_MN) -> uint64_t {
        const int LBO = 128 * 8;
        const int SBO = BLOCK_MN * sizeof(nv_bfloat16) * 8;
        return desc_encode(addr)
              | (desc_encode(LBO) << 16ULL)
              | (desc_encode(SBO) << 32ULL)
              | (1ULL << 46ULL)
              | (2ULL << 61ULL);
      };

      tcgen05_mma_f16(taddr, make_desc(A_smem, BLOCK_M), make_desc(B_smem, BLOCK_N), i_desc, iter_k);
      for (int k = 1; k < BLOCK_K / MMA_K; k++) {
        uint64_t a_desc = make_desc(A_smem + k * MMA_K * BLOCK_M * sizeof(nv_bfloat16), BLOCK_M);
        uint64_t b_desc = make_desc(B_smem + k * MMA_K * BLOCK_N * sizeof(nv_bfloat16), BLOCK_N);
        tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
      }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
    }

    // signal when tcgen05 finishes with the main loop
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mainloop_mbar_addr) : "memory");
  }
  __syncthreads();  // wait for all warps to reach here
  // wait for mainloop to finish
  mbarrier_wait(mainloop_mbar_addr, 0);

  // PTX doc says we need to add this before tcgen05.ld, after tcgen05.mma
  asm volatile("tcgen05.fence::after_thread_sync;");

  // load 8 columns from tmem at a time -> store 16 bytes per thread to smem
  // (still strided though)
  for (int n = 0; n < BLOCK_N / 8; n++) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d
    // Layout D
    float tmp[8];
    const int addr = taddr + ((warp_id * 32) << 16) + (n * 8);
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
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(BLOCK_N));
}

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES>
void matmul_v4_trans_launch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  CUtensorMap A_tmap, B_tmap;

  // original layout: [K, M] : [M, 1]
  // unflatten:       [K/8, 8, M/64, 64] : [M*8, M, 64, 1]
  // permute:         [K/8, M/64, 8, 64] : [M*8, 64, M, 1]
  auto init_tmap_AB = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr, uint64_t MN, uint32_t BLOCK_MN) {
    constexpr uint32_t rank = 4;
    uint64_t globalDim[rank]       = {64, 8, MN / 64, K / 8};
    uint64_t globalStrides[rank-1] = {MN * sizeof(nv_bfloat16), 128, MN * 8 * sizeof(nv_bfloat16)};  // in bytes
    uint32_t boxDim[rank]          = {64, 8, BLOCK_MN / 64, BLOCK_K / 8};
    uint32_t elementStrides[rank]  = {1, 1, 1, 1};

    auto err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      (void *)ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
  };
  init_tmap_AB(&A_tmap, A_ptr, M, BLOCK_M);
  init_tmap_AB(&B_tmap, B_ptr, N, BLOCK_N);

  int grid = (M / BLOCK_M) * (N / BLOCK_N);
  int size_AB = (BLOCK_M + BLOCK_N) * BLOCK_K * NUM_STAGES;
  int smem_size = size_AB * sizeof(nv_bfloat16);

  auto this_kernel = matmul_v4_trans_kernel<BLOCK_N, BLOCK_K, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  this_kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_ptr, M, N, K);
}

at::Tensor matmul_v4_trans(const at::Tensor& A, const at::Tensor& B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  auto C = at::empty({M, N}, A.options());
  // auto C = at::zeros({M, N}, A.options());  // for correctness check, use this

  auto *A_ptr = reinterpret_cast<nv_bfloat16 *>(A.data_ptr());
  auto *B_ptr = reinterpret_cast<nv_bfloat16 *>(B.data_ptr());
  auto *C_ptr = reinterpret_cast<nv_bfloat16 *>(C.data_ptr());

  matmul_v4_trans_launch<128, 128, 3>(A_ptr, B_ptr, C_ptr, M, N, K);
  return C;
}

TORCH_LIBRARY(my_matmul, m) {
  m.def("matmul_v4_trans(Tensor A, Tensor B) -> Tensor", &matmul_v4_trans);
}
