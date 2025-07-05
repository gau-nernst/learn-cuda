#include "common.h"
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

template <int TB_SIZE, int HEIGHT, int WIDTH>
__device__
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
  constexpr int MMA_N = 16;
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

  // convert shared memory address to 32-bit from the start
  extern __shared__ nv_bfloat16 shm[];
  const uint32_t shm_u32 = cvta_shared(shm);
  const uint32_t A0_shm = shm_u32;
  const uint32_t B0_shm = A0_shm + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);

  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE; // 4
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE; // 4
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {};
  uint32_t A_regs[NUM_MMA_K][NUM_MMA_M][num_A_regs];
  uint32_t B_regs[NUM_MMA_K][NUM_MMA_N][num_B_regs];

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
  const int A_offk = (lane_id / 16) * 8;
  const uint32_t A0_shm_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(A0_shm + (A_offm * BLOCK_K + A_offk) * sizeof(nv_bfloat16));

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
  const int B_offk = ((lane_id % 16) / 8) * 8;
  const uint32_t B0_shm_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(B0_shm + (B_offn * BLOCK_K + B_offk) * sizeof(nv_bfloat16));

  // pre-compute the address for each stage
  uint32_t A_shm_thread[NUM_STAGES];
  uint32_t B_shm_thread[NUM_STAGES];
  for (int stage = 0; stage < NUM_STAGES; stage++) {
    A_shm_thread[stage] = A0_shm_thread + stage * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
    B_shm_thread[stage] = B0_shm_thread + stage * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
  }

  // initiate NUM_STAGES-1 async global->shared
  auto global_to_shared = [&](int stage) {
    const uint32_t A_shared = shm_u32 + stage * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16); // BLOCK_M, BLOCK_K
    const uint32_t B_shared = A_shared + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);                    // BLOCK_N, BLOCK_K
    global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, tid);
    global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared, tid);

    // mark this stage as a commit group
    cp_async_commit_group();

    // NOTE: A and B pointers now track position for global->shared load
    A += BLOCK_K;
    B += BLOCK_K;
  };

  // NOTE: we don't take care when num_k_iters < NUM_STAGES
  for (int stage = 0; stage < NUM_STAGES - 1; stage++)
    global_to_shared(stage);

  // loop invariance: there is always NUM_STAGES - 1 prefetch stages in-flight
  // thanks to pipelining, this loop now only has 1 __syncthreads()
  for (int k_iter = 0; k_iter < K / BLOCK_K; k_iter++) {
    if constexpr (NUM_STAGES > 1) {
      // wait for the 1st commit group to finish i.e. FIFO
      // this consumes 1 prefetch
      cp_async_wait_group<NUM_STAGES - 2>();
      __syncthreads(); // why can't we move this after prefetch?

      // prefetch the next stage. restore loop invariance
      // NOTE: to avoid branching here, we can do K / BLOCK_K - NUM_STAGES + 1 in the mainloop
      // and unroll the last NUM_STAGES-1 iterations.
      // NOTE: the location of prefetch in main loop is important.
      // imagine using 2 stages. if we don't issue prefetch immediately after wait_group above,
      // global->shared is not busy anymore. for 3 stages, maybe issue global->shared later is fine?
      const int prefetch_iter = k_iter + NUM_STAGES - 1;
      if (prefetch_iter < (K / BLOCK_K))
        global_to_shared(prefetch_iter % NUM_STAGES);
      else
        cp_async_commit_group();
    } else {
      // without pipelining
      __syncthreads();
      global_to_shared(0);
      cp_async_wait_all();
      __syncthreads();
    }

    const int stage = k_iter % NUM_STAGES;

    // shared->registers
    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++) {
      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        const uint32_t A_addr = A_shm_thread[stage] + mma_id_m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_regs[mma_id_k][mma_id_m], A_addr ^ (mma_id_k * MMA_K * sizeof(nv_bfloat16)));
      }
      for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
        const uint32_t B_addr = B_shm_thread[stage] + mma_id_n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_regs[mma_id_k][mma_id_n], B_addr ^ (mma_id_k * MMA_K * sizeof(nv_bfloat16)));
      }
    }

    // do MMA. NUM_STAGES-1 prefetch stages are still on-going
    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
          uint32_t *A_reg = A_regs[mma_id_k][mma_id_m];
          uint32_t *B_reg = B_regs[mma_id_k][mma_id_n];
          float *acc_reg = acc[mma_id_m][mma_id_n];
          mma_m16n8k16(A_reg, B_reg, acc_reg);
          mma_m16n8k16(A_reg, B_reg + 2, acc_reg + 4);
        }
  }

  const int a0_row = lane_id >> 2;
  const int a0_col = (lane_id % 4) * 2;
  C += a0_row * N + a0_col;

  for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
      nv_bfloat16 *C_local = C + (mma_id_m * MMA_M) * N + (mma_id_n * MMA_N);
      float *regs = acc[mma_id_m][mma_id_n];

      reinterpret_cast<nv_bfloat162 *>(C_local)[0]             = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_local + 8 * N)[0]     = __float22bfloat162_rn({regs[2], regs[3]});
      reinterpret_cast<nv_bfloat162 *>(C_local + 8)[0]         = __float22bfloat162_rn({regs[4], regs[5]});
      reinterpret_cast<nv_bfloat162 *>(C_local + 8 * N + 8)[0] = __float22bfloat162_rn({regs[6], regs[7]});
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
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}
