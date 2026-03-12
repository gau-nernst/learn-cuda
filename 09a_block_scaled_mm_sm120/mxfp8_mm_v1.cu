#include "common.h"
#include <cuda_bf16.h>

constexpr int BLOCK_K = 128;

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void load_scales(int dst, const T *src, int src_stride, int tid) {
  constexpr int cp_size = WIDTH * sizeof(T);
  static_assert(cp_size < 16);  // if WIDTH * sizeof(T) >= 16, use global_to_shared_swizzle()

  // each cp.async loads 1 row
  auto load_row = [&](int row) {
    const int dst_addr = dst + row * WIDTH * sizeof(T);
    const T *src_addr = src + row * src_stride;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(dst_addr), "l"(src_addr), "n"(cp_size));
  };

  for (int iter = 0; iter < HEIGHT / TB_SIZE; iter++)
    load_row(iter * TB_SIZE + tid);

  // HEIGHT might not be divisible for TB_SIZE
  // handle the remaining rows
  if constexpr (HEIGHT % TB_SIZE != 0) {
    const int row = HEIGHT / TB_SIZE * TB_SIZE + tid;
    if (row < HEIGHT)
      load_row(row);
  }
}

template <int BLOCK_M, int BLOCK_N, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void mxfp8_mm_v1_kernel(
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

  A_ptr += off_m * K;
  B_ptr += off_n * K;
  SFA_ptr += off_m * (K / 32);
  SFB_ptr += off_n * (K / 32);
  C_ptr += (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

  // shared memory
  extern __shared__ char smem_ptr[];
  const int A_smem = __cvta_generic_to_shared(smem_ptr);
  const int B_smem = A_smem + BLOCK_M * BLOCK_K;
  const int SFA_smem = B_smem + BLOCK_N * BLOCK_K;
  const int SFB_smem = SFA_smem + BLOCK_M * (BLOCK_K / 32);

  constexpr int STAGE_SIZE = (BLOCK_M + BLOCK_N) * (BLOCK_K + BLOCK_K / 32);

  // pre-compute ldmatrix address
  const int A_smem_addr = A_smem + swizzle<BLOCK_K>((warp_id_m * WARP_M) + (lane_id % 16), lane_id / 16);
  const int B_smem_addr = B_smem + swizzle<BLOCK_K>((warp_id_n * WARP_N) + (lane_id % 8), lane_id / 8);

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
  int SFA_smem_addr = SFA_smem + ((warp_id_m * WARP_M) + (lane_id % 4) * 8 + (lane_id / 4)) * 4;
  int SFB_smem_addr = SFB_smem + ((warp_id_n * WARP_N) + (lane_id % 4) * 8 + (lane_id / 4)) * 4;

  // register memory
  int A_rmem[WARP_M / MMA_M][BLOCK_K / MMA_K][4];
  int B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K][2];
  int SFA_rmem[WARP_M / 32];
  int SFB_rmem[WARP_N / 32];
  float acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

  auto load = [&](int stage_id) {
    gmem_to_smem<BLOCK_M, BLOCK_K, TB_SIZE>(A_smem + stage_id * STAGE_SIZE, A_ptr, K, tid);
    gmem_to_smem<BLOCK_N, BLOCK_K, TB_SIZE>(B_smem + stage_id * STAGE_SIZE, B_ptr, K, tid);
    load_scales<BLOCK_M, BLOCK_K / 32, TB_SIZE>(SFA_smem + stage_id * STAGE_SIZE, SFA_ptr, K / 32, tid);
    load_scales<BLOCK_N, BLOCK_K / 32, TB_SIZE>(SFB_smem + stage_id * STAGE_SIZE, SFB_ptr, K / 32, tid);

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += BLOCK_K / 32;
    SFB_ptr += BLOCK_K / 32;

    asm volatile("cp.async.commit_group;");
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

    for (int reg_id = 0; reg_id < WARP_M / 32; reg_id++)
      asm volatile("ld.shared.u32 %0, [%1];"
                  : "=r"(SFA_rmem[reg_id])
                  : "r"(SFA_smem_addr + stage_id * STAGE_SIZE + reg_id * 32 * 4));

    for (int reg_id = 0; reg_id < WARP_N / 32; reg_id++)
      asm volatile("ld.shared.u32 %0, [%1];"
                  : "=r"(SFB_rmem[reg_id])
                  : "r"(SFB_smem_addr + stage_id * STAGE_SIZE + reg_id * 32 * 4));

    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int m = 0; m < WARP_M / MMA_M; m++)
        for (int n = 0; n < WARP_N / MMA_N; n++)
          mma_mxfp8(A_rmem[m][k], B_rmem[n][k], acc[m][n],
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

  for (int m = 0; m < WARP_M / MMA_M; m++)
    for (int n = 0; n < WARP_N / MMA_N; n++) {
      const int row = (m * MMA_M) + (lane_id / 4);
      const int col = (n * MMA_N) + (lane_id % 4) * 2;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C_ptr + (row + 0) * N + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_ptr + (row + 8) * N + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void mxfp8_mm_v1(
  const char        *A_ptr,    // [M, K]
  const char        *B_ptr,    // [N, K]
  const char        *SFA_ptr,  // [M, K/32]
  const char        *SFB_ptr,  // [N, K/32]
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  const int BLOCK_M = 128;
  const int BLOCK_N = 128;
  const int NUM_WARP_M = 2;
  const int NUM_WARP_N = 2;
  const int NUM_STAGES = 3;

  const int num_blocks = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  const int STAGE_SIZE = (BLOCK_M + BLOCK_N) * (BLOCK_K + BLOCK_K / 32);
  const int smem_size = STAGE_SIZE * NUM_STAGES;

  auto kernel = mxfp8_mm_v1_kernel<BLOCK_M, BLOCK_N, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
  launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, M, N, K);
}
