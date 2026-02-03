#include "common.h"
#include <assert.h>
#include <cuda_bf16.h>

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void gmem_to_smem(int dst, const T *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(T);
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    // NOTE: perhaps we can move swizzle out of this loop as well
    const int dst_addr = dst + swizzle<WIDTH * sizeof(T)>(row, col / num_elems);
    const T *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v0_kernel(
  const nv_bfloat16 *A,
  const nv_bfloat16 *B,
        nv_bfloat16 *C,
  int M, int N, int K
) {
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  static_assert(BLOCK_M % NUM_WARP_M == 0);
  static_assert(BLOCK_N % NUM_WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
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

  const int bid_m = group_off_m + ((bid % group_size) % group_m);
  const int bid_n = (bid % group_size) / group_m;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  // A is row-major, B is column-major, C is row-major
  A += off_m * K;
  B += off_n * K;
  C += (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int AB_size = A_size + B_size;

  // convert shared memory address to 32-bit from the start
  extern __shared__ char smem[];
  const int smem_addr = static_cast<int>(__cvta_generic_to_shared(smem));
  const int A_smem = smem_addr;
  const int B_smem = A_smem + A_size;

  int A_rmem[WARP_M / MMA_M][BLOCK_K / MMA_K][4];
  int B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K][2];
  float acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

  // pre-compute address and swizzling used for ldmatrix
  const int A_smem_thread = A_smem + swizzle<BLOCK_K * sizeof(nv_bfloat16)>(warp_id_m * WARP_M + (lane_id % 16), lane_id / 16);
  const int B_smem_thread = B_smem + swizzle<BLOCK_K * sizeof(nv_bfloat16)>(warp_id_n * WARP_N + (lane_id % 8), lane_id / 8);

  const int num_k_iters = cdiv(K, BLOCK_K);

  auto load_AB = [&](int stage_id) {
    gmem_to_smem<BLOCK_M, BLOCK_K, TB_SIZE>(A_smem + stage_id * AB_size, A, K, tid);
    gmem_to_smem<BLOCK_N, BLOCK_K, TB_SIZE>(B_smem + stage_id * AB_size, B, K, tid);
    A += BLOCK_K;
    B += BLOCK_K;
    asm volatile("cp.async.commit_group;");
  };

  auto compute = [&](int stage_id) {
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

    // MMA
    for (int m = 0; m < WARP_M / MMA_M; m++)
      for (int n = 0; n < WARP_N / MMA_N; n++)
        for (int k = 0; k < BLOCK_K / MMA_K; k++)
          mma_m16n8k16(A_rmem[m][k], B_rmem[n][k], acc[m][n]);
  };

  // initiate NUM_STAGES-1 cp.async stages
  for (int stage = 0; stage < NUM_STAGES - 1; stage++)
    load_AB(stage);

  // loop invariance: there are always NUM_STAGES stages
  for (int iter_k = 0; iter_k < num_k_iters - (NUM_STAGES - 1); iter_k++) {
    // wait for previous MMA (using the next buffer) to finish
    // -> NUM_STAGES-1 cp.async
    __syncthreads();

    // prefetch the next stage -> NUM_STAGES cp.async
    load_AB((iter_k + NUM_STAGES - 1) % NUM_STAGES);

    // wait for cp.async to finish -> NUM_STAGES-1 cp.async
    asm volatile("cp.async.wait_group %0;" :: "n"(NUM_STAGES - 1));
    __syncthreads();

    // initiate MMA -> NUM_STAGES-1 cp.async + 1 MMA
    compute(iter_k % NUM_STAGES);
  }

  for (int iter_k = num_k_iters - (NUM_STAGES - 1); iter_k < num_k_iters; iter_k++) {
    // wait for cp.async to finish
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group %0;" :: "n"(NUM_STAGES - 1));
    __syncthreads();

    // initiate MMA. don't need to wait for MMA
    compute(iter_k % NUM_STAGES);
  }

  for (int m = 0; m < WARP_M / MMA_M; m++)
    for (int n = 0; n < WARP_N / MMA_N; n++) {
      const int row = m * MMA_M + (lane_id / 4);
      const int col = n * MMA_N + (lane_id % 4) * 2;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C + (row + 0) * N + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C + (row + 8) * N + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v0(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // tuned for 5090
  const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 2;

  // tuned for PRO 6000
  // const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
  // const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  // const int NUM_STAGES = 3;

  auto kernel = matmul_v0_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

  launch_kernel(kernel, grid_size, TB_SIZE, smem_size, A, B, C, M, N, K);
}
