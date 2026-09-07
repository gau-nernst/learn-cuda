// using CuBLAS SF layout (WIP)
#include "common.h"
#include <cuda_bf16.h>

// fixed BLOCK_MNK
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 128;

constexpr int NUM_WARP_M = 2;
constexpr int NUM_WARP_N = 2;

template <int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void mxfp8_mm_v3_kernel(
  const char        *A_ptr,    // [M, K]
  const char        *B_ptr,    // [N, K]
  const char        *SFA_ptr,  // [M, K/32]
  const char        *SFB_ptr,  // [N, K/32]
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
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  // CuBLAS SF layout: from [M,K/32] to [M/128,K/128,32,4,4]
  // hence, each [BLOCK_MN,BLOCK_K]=[128,128] corresponds to 512B of SF data,
  // which corresponds to 1 warp issues 16B load per thread.
  A_ptr += off_m * K;
  B_ptr += off_n * K;
  SFA_ptr += bid_m * (K / 128) * 512;
  SFB_ptr += bid_n * (K / 128) * 512;

  // shared memory
  extern __shared__ char smem_ptr[];
  const int A_smem = __cvta_generic_to_shared(smem_ptr);
  const int B_smem = A_smem + BLOCK_M * BLOCK_K;
  const int SFA_smem = B_smem + BLOCK_N * BLOCK_K;
  const int SFB_smem = SFA_smem + BLOCK_M * (BLOCK_K / 32);

  constexpr int STAGE_SIZE = (BLOCK_M + BLOCK_N) * (BLOCK_K + BLOCK_K / 32);

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
  // we will use ldmatrix for SFA/SFB
  // recall CuBLAS SF layout
  //   [32x4] -> [32x4][32x4][32x4][32x4]
  //   [32x4]
  //   [32x4]
  //   [32x4]
  //
  // if warp0 loads the first ldmatrix tile [8x16B], warp0 holds the first 8 rows of each [32x4] tile
  //   i.e. row0-7, row32-39, row64-71, row96-103
  //
  // hence, we need to adjust the warp partition of A/B/C tile
  //   for A tile, this means that the 1st MMA tile consists of row00-07 + row32-039 from A
  //                                   2nd MMA tile             row64-71 + row96-103
  //
  //   for B tile, this means that the 1st MMA tile consists of row00-007 from B
  //                                   2nd MMA tile             row32-039
  //                                   3rd MMA tile             row64-071
  //                                   4th MMA tile             row96-103

  // pre-compute ldmatrix address
  const int A_smem_addr = A_smem + swizzle<BLOCK_K>((warp_id_m * 8) + (lane_id % 8) + (lane_id / 8) * 32, lane_id / 16);
  const int B_smem_addr = B_smem + swizzle<BLOCK_K>((warp_id_n * 8) + (lane_id % 8) + (lane_id / 16) * 32, (lane_id / 8) % 2);

  const int SFA_smem_addr = SFA_smem + (warp_id_m * 8 + (lane_id % 8) + (lane_id / 8) * (8 * NUM_WARP_M)) * 16;
  const int SFB_smem_addr = SFB_smem + (warp_id_n * 8 + (lane_id % 8) + (lane_id / 8) * (8 * NUM_WARP_N)) * 16;

  // register memory
  int A_rmem[BLOCK_K / MMA_K][WARP_M / MMA_M][4];
  int B_rmem[BLOCK_K / MMA_K][WARP_N / MMA_N][2];
  int SFA_rmem[WARP_M / 32];
  int SFB_rmem[WARP_N / 32];
  float acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

  auto load = [&](int stage_id) {
    gmem_to_smem<BLOCK_M, BLOCK_K, TB_SIZE>(A_smem + stage_id * STAGE_SIZE, A_ptr, K, tid);
    gmem_to_smem<BLOCK_N, BLOCK_K, TB_SIZE>(B_smem + stage_id * STAGE_SIZE, B_ptr, K, tid);
    if (warp_id == 0)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                  :: "r"(SFA_smem + stage_id * STAGE_SIZE + lane_id * 16), "l"(SFA_ptr + lane_id * 16));
    else if (warp_id == 1)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                  :: "r"(SFB_smem + stage_id * STAGE_SIZE + lane_id * 16), "l"(SFB_ptr + lane_id * 16));

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += 512;
    SFB_ptr += 512;

    asm volatile("cp.async.commit_group;");
  };

  auto compute = [&](int stage_id) {
    // notice the new row increment when we increment m
    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int m = 0; m < WARP_M / MMA_M; m++) {
        int addr = (A_smem_addr + stage_id * STAGE_SIZE + m * (8 * NUM_WARP_M) * BLOCK_K) ^ (k * 32);
        ldmatrix<4>(A_rmem[k][m], addr);
      }

    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int n = 0; n < WARP_N / MMA_N; n += 2) {
        int addr = (B_smem_addr + stage_id * STAGE_SIZE + n / 2 * (8 * NUM_WARP_N) * BLOCK_K) ^ (k * 32);
        ldmatrix<4>(B_rmem[k][n], addr);
      }

    ldmatrix<WARP_M / 32>(SFA_rmem, SFA_smem_addr + stage_id * STAGE_SIZE);
    ldmatrix<WARP_N / 32>(SFB_rmem, SFB_smem_addr + stage_id * STAGE_SIZE);

    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int m = 0; m < WARP_M / MMA_M; m++)
        for (int n = 0; n < WARP_N / MMA_N; n++)
          mma_mxfp8(A_rmem[k][m], B_rmem[k][n], acc[m][n],
                    SFA_rmem[m / 2], k, m % 2,
                    SFB_rmem[n / 4], k, n % 4);
  };

  int num_k_iters = cdiv(K, BLOCK_K);

  for (int stage_id = 0; stage_id < NUM_STAGES - 1; stage_id++)
    load(stage_id);

  for (int iter_k = 0; iter_k < num_k_iters - (NUM_STAGES - 1); iter_k++) {
    __syncthreads();  // wait MMA
    load((iter_k + NUM_STAGES - 1) % NUM_STAGES);  // issue prefetch

    asm volatile("cp.async.wait_group %0;" :: "n"(NUM_STAGES - 1));  // wait cp.async
    __syncthreads();
    compute(iter_k % NUM_STAGES);  // issue MMA
  }

  for (int iter_k = num_k_iters - (NUM_STAGES - 1); iter_k < num_k_iters; iter_k++) {
    asm volatile("cp.async.commit_group;");  // commit empty group
    asm volatile("cp.async.wait_group %0;" :: "n"(NUM_STAGES - 1));  // wait cp.async
    __syncthreads();
    compute(iter_k % NUM_STAGES);
  }

  // adjust output tile based on new layout
  C_ptr += (off_m + warp_id_m * 8) * N + (off_n + warp_id_n * 8);

  for (int m = 0; m < WARP_M / MMA_M; m++)
    for (int n = 0; n < WARP_N / MMA_N; n++) {
      const int row = (m * 8 * NUM_WARP_M) + ((lane_id / 4) % 2) + ;
      const int col = (n * 16) + (lane_id % 4) * 2;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C_ptr + (row + 0) * N + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_ptr + (row + 8) * N + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void mxfp8_mm_v3(
  const char        *A_ptr,    // [M, K]
  const char        *B_ptr,    // [N, K]
  const char        *SFA_ptr,  // [M, K/32]
  const char        *SFB_ptr,  // [N, K/32]
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  const int NUM_STAGES = 3;

  const int num_blocks = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  const int STAGE_SIZE = (BLOCK_M + BLOCK_N) * (BLOCK_K + BLOCK_K / 32);
  const int smem_size = STAGE_SIZE * NUM_STAGES;

  auto kernel = mxfp8_mm_v3_kernel<NUM_STAGES>;
  launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, M, N, K);
}
