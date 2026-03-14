// using custom SF layout
#include "common.h"
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

constexpr int BLOCK_K = 128;

template <int BLOCK_M, int BLOCK_N, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void mxfp8_mm_v2b_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  const __grid_constant__ CUtensorMap SFA_tmap,
  const __grid_constant__ CUtensorMap SFB_tmap,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  constexpr int MMA_K = 32;

  const int bid = blockIdx.x;
  const int bid_m = bid / cdiv(N, BLOCK_N);
  const int bid_n = bid % cdiv(N, BLOCK_N);

  const int tid = threadIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;
  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  C_ptr += (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

  // shared memory
  constexpr int STAGE_SIZE = (BLOCK_M + BLOCK_N) * (BLOCK_K + BLOCK_K / 32);

  extern __shared__ char smem_ptr[];
  const int smem_addr = __cvta_generic_to_shared(smem_ptr);
  const int A_smem = smem_addr;
  const int B_smem = A_smem + BLOCK_M * BLOCK_K;
  const int SFA_smem = B_smem + BLOCK_N * BLOCK_K;
  const int SFB_smem = SFA_smem + BLOCK_M * (BLOCK_K / 32);

  const int mbar_addr = smem_addr + NUM_STAGES * STAGE_SIZE;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++)
      mbarrier_init(mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  __syncthreads();  // mbarrier visible

  // pre-compute ldmatrix address
  const int A_smem_addr = A_smem + swizzle<BLOCK_K>((warp_id_m * WARP_M) + (lane_id % 16), lane_id / 16);
  const int B_smem_addr = B_smem + swizzle<BLOCK_K>((warp_id_n * WARP_N) + (lane_id % 8), lane_id / 8);

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
  // for scales, we repack [32,4] = [4,8,4] -> [8,4,4]
  // width is 16 -> we can use cp.async.cg + ldmatrix
  // since width is 16, we also don't need swizzling.
  int SFA_smem_addr = SFA_smem + (warp_id_m * WARP_M / 4 + lane_id) * 16;
  int SFB_smem_addr = SFB_smem + (warp_id_n * WARP_N / 4 + lane_id) * 16;

  // register memory
  int A_rmem[WARP_M / MMA_M][BLOCK_K / MMA_K][4];
  int B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K][2];
  int SFA_rmem[WARP_M / 32];
  int SFB_rmem[WARP_N / 32];
  float acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

  auto load = [&](int iter_k, int stage_id) {
    if (warp_id == 0 && elect_sync()) {
      const int this_mbar = mbar_addr + stage_id * 8;
      const int off_k = iter_k * BLOCK_K;
      tma_2d_g2s(A_smem + stage_id * STAGE_SIZE, &A_tmap, off_k, off_m, this_mbar);
      tma_2d_g2s(B_smem + stage_id * STAGE_SIZE, &B_tmap, off_k, off_n, this_mbar);
      tma_2d_g2s(SFA_smem + stage_id * STAGE_SIZE, &SFA_tmap, off_k / 8, off_m / 4, this_mbar);
      tma_2d_g2s(SFB_smem + stage_id * STAGE_SIZE, &SFB_tmap, off_k / 8, off_n / 4, this_mbar);
      mbarrier_arrive_expect_tx(this_mbar, STAGE_SIZE);
    }
  };

  auto compute = [&](int stage_id) {
    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int m = 0; m < WARP_M / MMA_M; m++) {
        int addr = (A_smem_addr + stage_id * STAGE_SIZE + m * MMA_M * BLOCK_K) ^ (k * 32);
        ldmatrix<4>(A_rmem[m][k], addr);
      }

    for (int k = 0; k < BLOCK_K / MMA_K; k += 2)
      for (int n = 0; n < WARP_N / MMA_N; n++) {
        int addr = (B_smem_addr + stage_id * STAGE_SIZE + n * MMA_N * BLOCK_K) ^ (k * 32);
        ldmatrix<4>(B_rmem[n][k], addr);
      }

    ldmatrix<WARP_M / 32>(SFA_rmem, SFA_smem_addr + stage_id * STAGE_SIZE);
    ldmatrix<WARP_N / 32>(SFB_rmem, SFB_smem_addr + stage_id * STAGE_SIZE);

    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int m = 0; m < WARP_M / MMA_M; m++)
        for (int n = 0; n < WARP_N / MMA_N; n++)
          mma_mxfp8(A_rmem[m][k], B_rmem[n][k], acc[m][n],
                    SFA_rmem[m / 2], k, m % 2,
                    SFB_rmem[n / 4], k, n % 4);
  };

  int num_k_iters = cdiv(K, BLOCK_K);

  for (int stage_id = 0; stage_id < NUM_STAGES - 1; stage_id++)
    load(stage_id, stage_id);

  int stage_id = 0;
  int phase = 0;

  for (int iter_k = 0; iter_k < num_k_iters - (NUM_STAGES - 1); iter_k++) {
    // issue prefetch
    __syncthreads();  // wait MMA
    const int prefetch_iter_k = iter_k + NUM_STAGES - 1;
    load(prefetch_iter_k, prefetch_iter_k % NUM_STAGES);

    // issue MMA
    if (warp_id == 1)  // warp0 issues prefetch TMA, warp1 waits current TMA
      mbarrier_wait(mbar_addr + stage_id * 8, phase);  // wait TMA
    __syncthreads();
    compute(stage_id);

    // increment stage_id and phase
    stage_id = (stage_id + 1) % NUM_STAGES;
    if (stage_id == 0)
      phase ^= 1;
  }

  for (int iter_k = num_k_iters - (NUM_STAGES - 1); iter_k < num_k_iters; iter_k++) {
    // issue MMA
    if (warp_id == 0)
      mbarrier_wait(mbar_addr + stage_id * 8, phase);  // wait TMA
    __syncthreads();
    compute(stage_id);

    // increment stage_id and phase
    stage_id = (stage_id + 1) % NUM_STAGES;
    if (stage_id == 0)
      phase ^= 1;
  }

  for (int m = 0; m < WARP_M / MMA_M; m++)
    for (int n = 0; n < WARP_N / MMA_N; n++) {
      const int row = (m * MMA_M) + (lane_id / 4);
      const int col = (n * MMA_N) + (lane_id % 4) * 2;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C_ptr + (row + 0) * N + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_ptr + (row + 8) * N + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

static void init_tensor_map(
  CUtensorMap *tmap_ptr,
  const char *gmem_ptr,
  uint64_t gmem_height, uint64_t gmem_width,
  uint32_t smem_height, uint32_t smem_width,
  CUtensorMapSwizzle swizzle
) {
  constexpr uint32_t rank = 2;
  uint64_t size[rank]        = {gmem_width, gmem_height};
  uint64_t stride[rank - 1]  = {gmem_width};  // in bytes
  uint32_t box_size[rank]    = {smem_width, smem_height};
  uint32_t elem_stride[rank] = {1, 1};

  auto res = cuTensorMapEncodeTiled(
    tmap_ptr,
    CU_TENSOR_MAP_DATA_TYPE_UINT8,
    rank, (void *)gmem_ptr, size, stride,
    box_size, elem_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    swizzle,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  if (res != CUDA_SUCCESS) {
    const char *error_msg_ptr;
    if (cuGetErrorString(res, &error_msg_ptr) != CUDA_SUCCESS)
      error_msg_ptr = "unable to get error string";
    std::cerr << "cuTensorMapEncodeTiled error: " << error_msg_ptr << std::endl;
  }
};

void mxfp8_mm_v2b(
  const char        *A_ptr,    // [M, K]
  const char        *B_ptr,    // [N, K]
  const char        *SFA_ptr,  // [M, K/32]
  const char        *SFB_ptr,  // [N, K/32]
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // TODO: currently the code is only correct for BLOCK_M=BLOCK_N=128
  // need to investigate why
  const int BLOCK_M = 128;
  const int BLOCK_N = 128;
  const int NUM_WARP_M = 2;
  const int NUM_WARP_N = 2;
  const int NUM_STAGES = 2;

  const int num_blocks = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  const int STAGE_SIZE = (BLOCK_M + BLOCK_N) * (BLOCK_K + BLOCK_K / 32);
  const int smem_size = STAGE_SIZE * NUM_STAGES
                      + NUM_STAGES * 8;  // mbar

  CUtensorMap A_tmap, B_tmap, SFA_tmap, SFB_tmap;
  init_tensor_map(&A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K, CU_TENSOR_MAP_SWIZZLE_128B);
  init_tensor_map(&B_tmap, B_ptr, N, K, BLOCK_N, BLOCK_K, CU_TENSOR_MAP_SWIZZLE_128B);
  init_tensor_map(&SFA_tmap, SFA_ptr, M / 4, K / 8, BLOCK_M / 4, BLOCK_K / 8, CU_TENSOR_MAP_SWIZZLE_NONE);
  init_tensor_map(&SFB_tmap, SFB_ptr, N / 4, K / 8, BLOCK_N / 4, BLOCK_K / 8, CU_TENSOR_MAP_SWIZZLE_NONE);

  auto kernel = mxfp8_mm_v2b_kernel<BLOCK_M, BLOCK_N, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
  launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, A_tmap, B_tmap, SFA_tmap, SFB_tmap, C_ptr, M, N, K);
}
