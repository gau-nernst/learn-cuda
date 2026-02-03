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
    uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(out + (row * WIDTH + col) * sizeof(nv_bfloat16));
    cp_async(dst_addr, in + row * in_stride + col);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v6_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
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
  const int A_offk = (lane_id / 16) * 8;
  const uint32_t A_shm_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(A_shm + (A_offm * BLOCK_K + A_offk) * sizeof(nv_bfloat16));

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
  const int B_offk = ((lane_id % 16) / 8) * 8;
  const uint32_t B_shm_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(B_shm + (B_offn * BLOCK_K + B_offk) * sizeof(nv_bfloat16));

  const int num_k_iters = cdiv(K, BLOCK_K);

  auto load_AB = [&](int k_iter) {
    if (k_iter < num_k_iters) {
      // select the correct shared memory buffer
      const int stage_id = k_iter % NUM_STAGES;
      global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shm + stage_id * AB_size, tid);
      global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shm + stage_id * AB_size, tid);

      // A/B pointer tracks position for global->shared load
      A += BLOCK_K;
      B += BLOCK_K;
    }
    cp_async_commit_group();
  };

  // initiate NUM_STAGES-1 stages
  for (int stage = 0; stage < NUM_STAGES - 1; stage++)
    load_AB(stage);

  // loop invariance: there is always NUM_STAGES - 1 prefetch stages in-flight
  // thanks to pipelining, this loop now only has 1 __syncthreads()
  for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    // wait for previous MMA to finish using the shared buffer
    __syncthreads();

    // prefetch the next stage. add 1 more stage to the pipeline
    load_AB(k_iter + NUM_STAGES - 1);

    // wait for the 1st stage to finish. remove 1 stage from the pipeline
    // -> restore loop invariance
    cp_async_wait_group<NUM_STAGES - 1>();
    __syncthreads();

    // A shared->regs
    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        uint32_t A_addr = A_shm_thread + (k_iter % NUM_STAGES) * AB_size;
        A_addr += mma_id_m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        A_addr ^= mma_id_k * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_regs[mma_id_k][mma_id_m], A_addr);
      }

    // B shared->regs
    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++)
      for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n += 2) {
        uint32_t B_addr = B_shm_thread + (k_iter % NUM_STAGES) * AB_size;
        B_addr += mma_id_n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        B_addr ^= mma_id_k * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_regs[mma_id_k][mma_id_n], B_addr);
      }

    // do MMA. NUM_STAGES-1 prefetch stages are still on-going
    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++)
          mma_m16n8k16(A_regs[mma_id_k][mma_id_m],
                       B_regs[mma_id_k][mma_id_n],
                       acc[mma_id_m][mma_id_n]);
  }

  for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 4);
      const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;
      nv_bfloat16 *C_local = C + row * N + col;

      float *regs = acc[mma_id_m][mma_id_n];
      reinterpret_cast<nv_bfloat162 *>(C_local)[0]         = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_local + 8 * N)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v6(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 2;

  auto kernel = matmul_v6_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}
