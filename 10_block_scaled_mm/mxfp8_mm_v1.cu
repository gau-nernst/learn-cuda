#include "common.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

constexpr int BLOCK_K = 128;

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void load_scales(uint32_t dst, const T *src, int src_stride, int tid) {
  constexpr int cp_size = WIDTH * sizeof(T);
  static_assert(cp_size < 16);  // if WIDTH * sizeof(T) >= 16, use global_to_shared_swizzle()

  // each cp.async loads 1 row
  auto load_row = [&](int row) {
    const uint32_t dst_addr = dst + row * WIDTH * sizeof(T);
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

template <int BLOCK_M, int BLOCK_N, int NUM_WARP_M, int NUM_WARP_N>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void mxfp8_mm_v1_kernel(const __nv_fp8_e4m3 *A,        // [M, K]
                        const __nv_fp8_e4m3 *B,        // [N, K]
                        const __nv_fp8_e8m0 *scale_A,  // [M, K/32]
                        const __nv_fp8_e8m0 *scale_B,  // [N, K/32]
                        nv_bfloat16 *C,
                        int M, int N, int K) {
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 32;

  const int bid = blockIdx.x;
  const int bid_m = bid / cdiv(N, BLOCK_N);
  const int bid_n = bid % cdiv(N, BLOCK_N);

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  A += (bid_m * BLOCK_M) * K;
  B += (bid_n * BLOCK_N) * K;
  scale_A += (bid_m * BLOCK_M) * (K / 32);
  scale_B += (bid_n * BLOCK_N) * (K / 32);
  C += (bid_m * BLOCK_M + warp_id_m * WARP_M) * N + (bid_n * BLOCK_N + warp_id_n * WARP_N);

  // shared memory
  extern __shared__ uint8_t smem[];
  const uint32_t A_smem = __cvta_generic_to_shared(smem);
  const uint32_t B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(A[0]);
  const uint32_t scale_A_smem = B_smem + BLOCK_N * BLOCK_K * sizeof(B[0]);
  const uint32_t scale_B_smem = scale_A_smem + BLOCK_M * (BLOCK_K / 32) * sizeof(scale_A[0]);

  // pre-compute ldmatrix address
  uint32_t A_smem_addr;
  {
    const int row = (warp_id_m * WARP_M) + (lane_id % 16);
    const int col = (lane_id / 16) * 16;
    A_smem_addr = swizzle<BLOCK_K * sizeof(A[0])>(A_smem + (row * BLOCK_K + col) * sizeof(A[0]));
  }

  uint32_t B_smem_addr;
  {
    const int row = (warp_id_n * WARP_N) + (lane_id % 8);
    const int col = (lane_id / 8) * 16;
    B_smem_addr = swizzle<BLOCK_K * sizeof(B[0])>(B_smem + (row * BLOCK_K + col) * sizeof(B[0]));
  }

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
  uint32_t scale_A_smem_addr = scale_A_smem + ((warp_id_m * WARP_M) + (lane_id % 4) * 8 + (lane_id / 4)) * 4;
  uint32_t scale_B_smem_addr = scale_B_smem + ((warp_id_n * WARP_N) + (lane_id % 4) * 8 + (lane_id / 4)) * 4;

  // register memory
  uint32_t A_rmem[WARP_M / MMA_M][BLOCK_K / MMA_K][MMA_M * MMA_K / WARP_SIZE * sizeof(A[0]) / 4];
  uint32_t B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K][MMA_N * MMA_K / WARP_SIZE * sizeof(B[0]) / 4];
  uint32_t scale_A_rmem[WARP_M / 32];
  uint32_t scale_B_rmem[WARP_N / 32];
  float acc[WARP_M / MMA_M][WARP_N / MMA_N][MMA_M * MMA_N / WARP_SIZE] = {};

  int num_k_iters = cdiv(K, BLOCK_K);

  for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    global_to_shared_swizzle<BLOCK_M, BLOCK_K, TB_SIZE>(A_smem, A, K, tid);
    global_to_shared_swizzle<BLOCK_N, BLOCK_K, TB_SIZE>(B_smem, B, K, tid);
    load_scales<BLOCK_M, BLOCK_K / 32, TB_SIZE>(scale_A_smem, scale_A, K / 32, tid);
    load_scales<BLOCK_N, BLOCK_K / 32, TB_SIZE>(scale_B_smem, scale_B, K / 32, tid);

    A += BLOCK_K;
    B += BLOCK_K;
    scale_A += BLOCK_K / 32;
    scale_B += BLOCK_K / 32;

    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++) {
        const uint32_t row = mma_id_m * MMA_M * sizeof(A[0]);
        const uint32_t col = mma_id_k * MMA_K * sizeof(A[0]);
        ldmatrix<4>(A_rmem[mma_id_m][mma_id_k], (A_smem_addr + row * BLOCK_K) ^ col);
      }

    for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k += 2)
      for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
        const uint32_t row = mma_id_n * MMA_N * sizeof(B[0]);
        const uint32_t col = mma_id_k * MMA_K * sizeof(B[0]);
        ldmatrix<4>(B_rmem[mma_id_n][mma_id_k], (B_smem_addr + row * BLOCK_K) ^ col);
      }

    for (int reg_id = 0; reg_id < WARP_M / 32; reg_id++)
      asm volatile("ld.shared.u32 %0, [%1];"
                  : "=r"(scale_A_rmem[reg_id])
                  : "r"(scale_A_smem_addr + reg_id * 32 * 4));

    for (int reg_id = 0; reg_id < WARP_N / 32; reg_id++)
      asm volatile("ld.shared.u32 %0, [%1];"
                  : "=r"(scale_B_rmem[reg_id])
                  : "r"(scale_B_smem_addr + reg_id * 32 * 4));

    for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
        for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++)
          mma_m16n8k32_mxfp8(A_rmem[mma_id_m][mma_id_k],
                             B_rmem[mma_id_n][mma_id_k],
                             acc[mma_id_m][mma_id_n],
                             scale_A_rmem[mma_id_m / 2], mma_id_k, mma_id_m % 2,
                             scale_B_rmem[mma_id_n / 4], mma_id_k, mma_id_n % 4);

    __syncthreads();
  }

  for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
      const int row = (mma_id_m * MMA_M) + (lane_id / 4);
      const int col = (mma_id_n * MMA_N) + (lane_id % 4) * 2;

      float *regs = acc[mma_id_m][mma_id_n];
      reinterpret_cast<nv_bfloat162 *>(C + row * N + col)[0]       = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C + (row + 8) * N + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void mxfp8_mm_v1(const __nv_fp8_e4m3 *A,        // [M, K]
                 const __nv_fp8_e4m3 *B,        // [N, K]
                 const __nv_fp8_e8m0 *scale_A,  // [M, K/32]
                 const __nv_fp8_e8m0 *scale_B,  // [N, K/32]
                 nv_bfloat16 *C,
                 int M, int N, int K) {

  const int BLOCK_M = 128;
  const int BLOCK_N = 128;
  const int NUM_WARP_M = 2;
  const int NUM_WARP_N = 2;

  const int num_blocks = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  // 33/32 = (1 + 1/32), where 1/32 is the amount of each scale_A/B
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K / 32 * 33;

  auto kernel = mxfp8_mm_v1_kernel<BLOCK_M, BLOCK_N, NUM_WARP_M, NUM_WARP_N>;
  launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, A, B, scale_A, scale_B, C, M, N, K);
}
