#include "common.h"
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, typename input_type>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void row_scaled_mm_v1_kernel(const input_type *A,  // [M, K]
                             const input_type *B,  // [N, K]
                             const float *scale_A, // [M]
                             const float *scale_B, // [N]
                             nv_bfloat16 *C,       // [M, N]
                             int M, int N, int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 32;
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  const int bid = blockIdx.x;
  const int bid_m = bid / cdiv(N, BLOCK_N);
  const int bid_n = bid % cdiv(N, BLOCK_N);

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  // INT32 acc for INT8 input, otherwise FP32 acc
  using acc_type = std::conditional_t<std::is_same_v<input_type, int8_t>, int32_t, float>;

  // A is row-major, B is column-major, C is row-major
  A += (bid_m * BLOCK_M) * K;
  B += (bid_n * BLOCK_N) * K;
  scale_A += bid_m * BLOCK_M;
  scale_B += bid_n * BLOCK_N;
  C += (bid_m * BLOCK_M + warp_id_m * WARP_M) * N + (bid_n * BLOCK_N + warp_id_n * WARP_N);

  // convert shared memory address to 32-bit from the start
  extern __shared__ uint8_t smem[];
  const uint32_t A_smem = __cvta_generic_to_shared(smem);
  const uint32_t B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(input_type);

  uint32_t A_rmem[WARP_M / MMA_M][BLOCK_K / MMA_K][MMA_M * MMA_K * sizeof(input_type) / 4 / WARP_SIZE];
  uint32_t B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K][MMA_N * MMA_K * sizeof(input_type) / 4 / WARP_SIZE];
  acc_type acc[WARP_M / MMA_M][WARP_N / MMA_N][MMA_M * MMA_N / WARP_SIZE] = {};

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
  const int A_offk = (lane_id / 16) * 16;
  const uint32_t A_smem_thread = swizzle<BLOCK_K * sizeof(input_type)>(A_smem + (A_offm * BLOCK_K + A_offk) * sizeof(input_type));

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8);
  const int B_offk = (lane_id / 8) * 16;
  const uint32_t B_smem_thread = swizzle<BLOCK_K * sizeof(input_type)>(B_smem + (B_offn * BLOCK_K + B_offk) * sizeof(input_type));

  // pre-compute the address for each stage
  uint32_t A_buffers[NUM_STAGES];
  uint32_t B_buffers[NUM_STAGES];
  for (int stage = 0; stage < NUM_STAGES; stage++) {
    A_buffers[stage] = A_smem_thread + stage * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(input_type);
    B_buffers[stage] = B_smem_thread + stage * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(input_type);
  }

  const int num_k_iters = cdiv(K, BLOCK_K);

  auto load_data = [&](int k_iter) {
    if (k_iter < num_k_iters) {
      const uint32_t A_shared = A_smem + (k_iter % NUM_STAGES) * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(input_type);
      const uint32_t B_shared = B_smem + (k_iter % NUM_STAGES) * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(input_type);
      global_to_shared_swizzle<BLOCK_M, BLOCK_K, TB_SIZE>(A_shared, A, K, tid);
      global_to_shared_swizzle<BLOCK_N, BLOCK_K, TB_SIZE>(B_shared, B, K, tid);
      A += BLOCK_K;
      B += BLOCK_K;
    }
    asm volatile("cp.async.commit_group;");
  };

  // prefetch
  for (int stage = 0; stage < NUM_STAGES - 1; stage++)
    load_data(stage);

  for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    load_data(k_iter + NUM_STAGES - 1);  // prefetch

    asm volatile("cp.async.wait_group %0;" :: "n"(NUM_STAGES - 1));
    __syncthreads();

    // A shared->regs
    for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++) {
        uint32_t A_addr = A_buffers[k_iter % NUM_STAGES];
        A_addr += mma_id_m * MMA_M * BLOCK_K * sizeof(input_type);
        A_addr ^= mma_id_k * MMA_K * sizeof(input_type);
        ldmatrix<4>(A_rmem[mma_id_m][mma_id_k], A_addr);
      }

    // B shared->regs
    for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k += 2)
      for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
        uint32_t B_addr = B_buffers[k_iter % NUM_STAGES];
        B_addr += mma_id_n * MMA_N * BLOCK_K * sizeof(input_type);
        B_addr ^= mma_id_k * MMA_K * sizeof(input_type);
        ldmatrix<4>(B_rmem[mma_id_n][mma_id_k], B_addr);
      }

    // MMA
    for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
        for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++)
          mma_m16n8k32<input_type>(A_rmem[mma_id_m][mma_id_k],
                                   B_rmem[mma_id_n][mma_id_k],
                                   acc[mma_id_m][mma_id_n]);

    __syncthreads();
  }
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // not sure why cp.async doesn't work here
  for (int idx = tid; idx < BLOCK_M; idx += TB_SIZE)
    reinterpret_cast<float *>(smem)[idx] = scale_A[idx];
  for (int idx = tid; idx < BLOCK_N; idx += TB_SIZE) {
    reinterpret_cast<float *>(smem)[BLOCK_M + idx] = scale_B[idx];
  }
  __syncthreads();

  float scale_A_rmem[WARP_M / MMA_M][2];
  for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++) {
    float *addr = reinterpret_cast<float *>(smem) + (warp_id_m * WARP_M) + (mma_id_m * MMA_M) + (lane_id / 4);
    scale_A_rmem[mma_id_m][0] = addr[0];
    scale_A_rmem[mma_id_m][1] = addr[8];
  }
  float scale_B_rmem[WARP_N / MMA_N][2];
  for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
    float *addr = reinterpret_cast<float *>(smem) + BLOCK_M + (warp_id_n * WARP_N) + (mma_id_n * MMA_N) + (lane_id % 4) * 2;
    scale_B_rmem[mma_id_n][0] = addr[0];
    scale_B_rmem[mma_id_n][1] = addr[1];
  }

  for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 4);
      const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;

      // this will convert INT32->FP32 as necessary
      acc_type *regs = acc[mma_id_m][mma_id_n];
      float c0 = regs[0] * scale_A_rmem[mma_id_m][0] * scale_B_rmem[mma_id_n][0];
      float c1 = regs[1] * scale_A_rmem[mma_id_m][0] * scale_B_rmem[mma_id_n][1];
      float c2 = regs[2] * scale_A_rmem[mma_id_m][1] * scale_B_rmem[mma_id_n][0];
      float c3 = regs[3] * scale_A_rmem[mma_id_m][1] * scale_B_rmem[mma_id_n][1];

      reinterpret_cast<nv_bfloat162 *>(C + (row + 0) * N + col)[0] = __float22bfloat162_rn({c0, c1});
      reinterpret_cast<nv_bfloat162 *>(C + (row + 8) * N + col)[0] = __float22bfloat162_rn({c2, c3});
    }
}

void row_scaled_mm_v1(const int8_t *A,
                      const int8_t *B,
                      const float *scale_A,
                      const float *scale_B,
                      nv_bfloat16 *C,
                      int M, int N, int K) {
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 128;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 1;

  using input_type = int8_t;
  auto kernel = row_scaled_mm_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES, input_type>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(input_type) * NUM_STAGES;

  launch_kernel(kernel, grid_size, TB_SIZE, smem_size, A, B, scale_A, scale_B, C, M, N, K);
}

void row_scaled_mm_v1(const __nv_fp8_e4m3 *A,
                      const __nv_fp8_e4m3 *B,
                      const float *scale_A,
                      const float *scale_B,
                      nv_bfloat16 *C,
                      int M, int N, int K) {
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 128;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 1;

  using input_type = __nv_fp8_e4m3;
  auto kernel = row_scaled_mm_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES, input_type>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(input_type) * NUM_STAGES;

  launch_kernel(kernel, grid_size, TB_SIZE, smem_size, A, B, scale_A, scale_B, C, M, N, K);
}
