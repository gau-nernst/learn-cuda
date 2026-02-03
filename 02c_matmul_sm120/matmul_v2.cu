#include "common.h"
#include <assert.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__((NUM_WARP_M * NUM_WARP_N + 1) * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v2_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C,
  int M, int N, int K
) {
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;

  static_assert(BLOCK_M % NUM_WARP_M == 0);
  static_assert(BLOCK_N % NUM_WARP_N == 0);
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  // threadblock swizzling to improve L2 cache hit rate
  // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  constexpr int GROUP_M = 8;
  const int grid_m = cdiv(M, BLOCK_M);
  const int grid_n = cdiv(N, BLOCK_N);

  // each group is [GROUP_M, grid_n], tile from top (small M) to bottom (large M).
  // the last group might be shorter than GROUP_M if grid_m % GROUP_M != 0.
  const int group_size = GROUP_M * grid_n;
  const int group_id = bid / group_size;
  const int group_off_m = group_id * GROUP_M;
  const int group_m = std::min(grid_m - group_off_m, GROUP_M);  // actual group height

  const int bid_m = warp_uniform(group_off_m + ((bid % group_size) % group_m));
  const int bid_n = warp_uniform((bid % group_size) / group_m);

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int AB_size = A_size + B_size;

  // set up smem
  extern __shared__ __align__(1024) char smem[];
  const int smem_addr = static_cast<int>(__cvta_generic_to_shared(smem));
  const int A_smem = smem_addr;
  const int B_smem = A_smem + A_size;
  const int tma_mbar_addr = smem_addr + NUM_STAGES * AB_size;
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, 1);
      mbarrier_init(mma_mbar_addr + i * 8, NUM_WARP_M * NUM_WARP_N * WARP_SIZE);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }

  // make mbarrier visible
  __syncthreads();

  const int num_k_iters = cdiv(K, BLOCK_K);

  if (warp_id == NUM_WARP_M * NUM_WARP_N) {
    // TMA warp
    if (elect_sync()) {
      int stage_id = 0;
      int phase = 1;

      for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
        const int this_mbar_addr = tma_mbar_addr + stage_id * 8;
        const int off_k = iter_k * BLOCK_K;

        // wait MMA
        mbarrier_wait(mma_mbar_addr + stage_id * 8, phase);

        tma_2d_g2s(A_smem + stage_id * AB_size, &A_tmap, off_k, off_m, this_mbar_addr);
        tma_2d_g2s(B_smem + stage_id * AB_size, &B_tmap, off_k, off_n, this_mbar_addr);

        // signal TMA done
        mbarrier_arrive_expect_tx(this_mbar_addr, AB_size);

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          phase ^= 1;
      }
    }
  }
  else {
    // MMA warps
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;

    // set up rmem
    int A_rmem[WARP_M / MMA_M][BLOCK_K / MMA_K][4];
    int B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K][2];
    float acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

    // pre-compute address and swizzling used for ldmatrix
    const int A_smem_thread = A_smem + swizzle<BLOCK_K * sizeof(nv_bfloat16)>(warp_id_m * WARP_M + (lane_id % 16), lane_id / 16);
    const int B_smem_thread = B_smem + swizzle<BLOCK_K * sizeof(nv_bfloat16)>(warp_id_n * WARP_N + (lane_id % 8), lane_id / 8);

    int stage_id = 0;
    int phase = 0;

    for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
      // wait TMA
      if (warp_id == 0)
        mbarrier_wait(tma_mbar_addr + stage_id * 8, phase);
      asm volatile("bar.sync %0, %1;" :: "n"(1), "n"(NUM_WARP_M * NUM_WARP_N * WARP_SIZE));

      // A smem->rmem
      for (int m = 0; m < WARP_M / MMA_M; m++)
        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          int addr = A_smem_thread + stage_id * AB_size;
          addr += m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
          ldmatrix_x4(A_rmem[m][k], addr ^ (k * 32));
        }

      // B smem->rmem
      for (int n = 0; n < WARP_N / MMA_N; n++)
        for (int k = 0; k < BLOCK_K / MMA_K; k += 2) {
          int addr = B_smem_thread + stage_id * AB_size;
          addr += n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
          ldmatrix_x4(B_rmem[n][k], addr ^ (k * 32));
        }

      // signal finished using smem buffer
      mbarrier_arrive(mma_mbar_addr + stage_id * 8);

      // MMA
      for (int m = 0; m < WARP_M / MMA_M; m++)
        for (int n = 0; n < WARP_N / MMA_N; n++)
          for (int k = 0; k < BLOCK_K / MMA_K; k++)
            mma_m16n8k16(A_rmem[m][k], B_rmem[n][k], acc[m][n]);

      stage_id = (stage_id + 1) % NUM_STAGES;
      if (stage_id == 0)
        phase ^= 1;
    }

    // epilogue
    C += (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);
    for (int m = 0; m < WARP_M / MMA_M; m++)
      for (int n = 0; n < WARP_N / MMA_N; n++) {
        const int row = m * MMA_M + (lane_id / 4);
        const int col = n * MMA_N + (lane_id % 4) * 2;

        float *regs = acc[m][n];
        reinterpret_cast<nv_bfloat162 *>(C + (row + 0) * N + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
        reinterpret_cast<nv_bfloat162 *>(C + (row + 8) * N + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
      }
  }
}

static void init_tensor_map(
  CUtensorMap *tmap_ptr,
  const nv_bfloat16 *gmem_ptr,
  uint64_t gmem_height, uint64_t gmem_width,
  uint32_t smem_height, uint32_t smem_width
) {
  constexpr uint32_t rank = 2;
  uint64_t size[rank]        = {gmem_width, gmem_height};
  uint64_t stride[rank - 1]  = {gmem_width * sizeof(nv_bfloat16)};  // in bytes
  uint32_t box_size[rank]    = {smem_width, smem_height};
  uint32_t elem_stride[rank] = {1, 1};

  CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
  if (smem_width == 16)
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
  else if (smem_width == 32)
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
  else if (smem_width == 64)
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;

  auto res = cuTensorMapEncodeTiled(
    tmap_ptr,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    (void *)gmem_ptr,
    size,
    stride,
    box_size,
    elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    swizzle,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  if (res != CUDA_SUCCESS) {
    const char *error_msg_ptr;
    if (cuGetErrorString(res, &error_msg_ptr) != CUDA_SUCCESS)
      error_msg_ptr = "unable to get error string";
    std::cerr << "cuTensorMapEncodeTiled error: " << error_msg_ptr << std::endl;
  }
};

void matmul_v2(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // tuned for 5090
  const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 2;

  auto kernel = matmul_v2_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;

  const int TB_SIZE = (NUM_WARP_M * NUM_WARP_N + 1) * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES
                      + NUM_STAGES * 2 * 8;  // tma mbar and mma mbar

  CUtensorMap A_tmap, B_tmap;
  init_tensor_map(&A_tmap, A, M, K, BLOCK_M, BLOCK_K);
  init_tensor_map(&B_tmap, B, N, K, BLOCK_N, BLOCK_K);

  launch_kernel(kernel, grid_size, TB_SIZE, smem_size, A_tmap, B_tmap, C, M, N, K);
}
