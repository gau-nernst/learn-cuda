#include "common.h"
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

template <int TB_SIZE, int HEIGHT, int WIDTH>
__device__ static
void global_to_shared_async(const nv_bfloat16 *in, int in_stride, uint32_t out, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    // NOTE: perhaps we can move swizzle out of this loop as well
    uint32_t dst_addr = out + swizzle_better<WIDTH * sizeof(nv_bfloat16)>(row, col / num_elems);
    cp_async(dst_addr, in + row * in_stride + col);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v7_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  static_assert(BLOCK_M % NUM_WARP_M == 0);
  static_assert(BLOCK_N % NUM_WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;
  constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // TODO: threadblock swizzling to improve L2 cache hit rate
  const int num_blocks_n = cdiv(N, BLOCK_N);
  const int bid_m = bid / num_blocks_n;
  const int bid_n = bid % num_blocks_n;
  const int offset_m = bid_m * BLOCK_M;
  const int offset_n = bid_n * BLOCK_N;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  // A is row-major, B is column-major, C is row-major
  A += offset_m * K;
  B += offset_n * K;
  C += (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);

  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int AB_size = A_size + B_size;

  // convert shared memory address to 32-bit from the start
  extern __shared__ nv_bfloat16 shm[];
  const uint32_t shm_u32 = cvta_shared(shm);
  const uint32_t A_shm = shm_u32;
  const uint32_t B_shm = A_shm + A_size;

  uint32_t A_regs[NUM_MMA_K][NUM_MMA_M][4];
  uint32_t B_regs[NUM_MMA_K][NUM_MMA_N][2];
  float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
  const uint32_t A_shm_thread = A_shm + swizzle_better<BLOCK_K * sizeof(nv_bfloat16)>(A_offm, lane_id / 16);

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
  const uint32_t B_shm_thread = B_shm + swizzle_better<BLOCK_K * sizeof(nv_bfloat16)>(B_offn, (lane_id % 16) / 8);

  const int num_k_iters = cdiv(K, BLOCK_K);

  auto load_AB = [&](int k_iter) {
    // select the correct shared memory buffer
    const int stage_id = k_iter % NUM_STAGES;
    global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shm + stage_id * AB_size, tid);
    global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shm + stage_id * AB_size, tid);

    // A/B pointer tracks position for global->shared load
    A += BLOCK_K;
    B += BLOCK_K;
    cp_async_commit_group();
  };

  auto compute = [&](int k_iter) {
    // A shared->regs
    for (int k = 0; k < NUM_MMA_K; k++)
      for (int m = 0; m < NUM_MMA_M; m++) {
        uint32_t A_addr = A_shm_thread + (k_iter % NUM_STAGES) * AB_size;
        A_addr += m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        A_addr ^= k * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_regs[k][m], A_addr);
      }

    // B shared->regs
    for (int k = 0; k < NUM_MMA_K; k++)
      for (int n = 0; n < NUM_MMA_N; n += 2) {
        uint32_t B_addr = B_shm_thread + (k_iter % NUM_STAGES) * AB_size;
        B_addr += n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        B_addr ^= k * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_regs[k][n], B_addr);
      }

    // do MMA
    for (int k = 0; k < NUM_MMA_K; k++)
      for (int m = 0; m < NUM_MMA_M; m++)
        for (int n = 0; n < NUM_MMA_N; n++)
          mma_m16n8k16(A_regs[k][m], B_regs[k][n], acc[m][n]);
  };

  // initiate NUM_STAGES-1 stages
  for (int stage = 0; stage < NUM_STAGES - 1; stage++)
    load_AB(stage);

  // loop invariance: there is always NUM_STAGES - 1 prefetch stages in-flight
  // thanks to pipelining, this loop now only has 1 __syncthreads()
  for (int k_iter = 0; k_iter < num_k_iters - (NUM_STAGES - 1); k_iter++) {
    // wait for previous MMA to finish using the shared buffer
    __syncthreads();

    // prefetch the next stage. add 1 more stage to the pipeline
    load_AB(k_iter + NUM_STAGES - 1);

    // wait for the 1st stage to finish. remove 1 stage from the pipeline
    // -> restore loop invariance
    cp_async_wait_group<NUM_STAGES - 1>();
    __syncthreads();

    // ldmatrix and mma
    compute(k_iter);
  }

  for (int k_iter = num_k_iters - (NUM_STAGES - 1); k_iter < num_k_iters; k_iter++) {
    // preserve invariance of cp.async commited groups
    cp_async_commit_group();

    // wait cp.async
    cp_async_wait_group<NUM_STAGES - 1>();
    __syncthreads();

    // ldmatrix and mma
    compute(k_iter);
  }

  for (int m = 0; m < NUM_MMA_M; m++)
    for (int n = 0; n < NUM_MMA_N; n++) {
      const int row = m * MMA_M + (lane_id / 4);
      const int col = n * MMA_N + (lane_id % 4) * 2;
      nv_bfloat16 *C_local = C + row * N + col;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C_local)[0]         = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_local + 8 * N)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v7(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 2;

  auto kernel = matmul_v7_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}
