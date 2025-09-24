#include "common.h"
#include <hip/hip_bf16.h>

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void gmem_to_smem(T *dst, const T *src, int src_stride, int tid) {
  using load_type = float4;
  constexpr int multiplier = sizeof(load_type) / sizeof(T);
  static_assert((HEIGHT * WIDTH) % (TB_SIZE * multiplier) == 0);
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * multiplier);

  for (int i = 0; i < num_iters; i++) {
    const int idx = (i * TB_SIZE + tid) * multiplier;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const load_type data = reinterpret_cast<const load_type *>(src + row * src_stride + col)[0];

    const int swizzled_col = swizzle<WIDTH>(row, col);
    reinterpret_cast<load_type *>(dst + row * WIDTH + swizzled_col)[0] = data;
  }
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int GROUP_M, int NUM_WARP_M, int NUM_WARP_N>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_v2_kernel(
  const __hip_bfloat16 *A_gmem,
  const __hip_bfloat16 *B_gmem,
        __hip_bfloat16 *C_gmem,
  int M, int N, int K
) {
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  const int bid = blockIdx.x;
  const int grid_m = cdiv(M, BLOCK_M);
  const int grid_n = cdiv(N, BLOCK_N);

  int bid_m, bid_n;
  if constexpr (GROUP_M == 1) {
    bid_m = bid / grid_n;
    bid_n = bid % grid_n;
  } else {
    // threadblock swizzling, from triton
    // improve L2 reuse when M is large.
    const int group_size = GROUP_M * grid_n;
    const int group_id = bid / group_size;
    const int first_bid_m = group_id * GROUP_M;
    const int group_size_m = min(grid_m - first_bid_m, GROUP_M);
    bid_m = first_bid_m + ((bid % group_size) % group_size_m);
    bid_n = (bid % group_size) / group_size_m;
  }

  const int offset_m = bid_m * BLOCK_M;
  const int offset_n = bid_n * BLOCK_N;
  A_gmem += offset_m * K;
  B_gmem += offset_n * K;
  C_gmem += (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);

  // shared memory
  extern __shared__ __hip_bfloat16 smem[];
  __hip_bfloat16 *A_smem = smem;
  __hip_bfloat16 *B_smem = A_smem + BLOCK_M * BLOCK_K;

  // register memory
  // do it this way to use mfma intrinsic
  s16x4 A_rmem[WARP_M / MMA_M][BLOCK_K / MMA_K];
  s16x4 B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K];
  fp32x4 C_rmem[WARP_M / MMA_M][WARP_N / MMA_N] = {};

  const int num_k_iters = cdiv(K, BLOCK_K);
  for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
    // gmem->smem
    __syncthreads();
    gmem_to_smem<BLOCK_M, BLOCK_K, TB_SIZE>(A_smem, A_gmem, K, tid);
    gmem_to_smem<BLOCK_N, BLOCK_K, TB_SIZE>(B_smem, B_gmem, K, tid);
    A_gmem += BLOCK_K;
    B_gmem += BLOCK_K;

    // smem->rmem
    // TODO: use wider load?
    // NOTE: for some reasons, factoring out swizzle out of the main loop is a bit slower.
    __syncthreads();
    for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
      for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++) {
        const int row = (warp_id_m * WARP_M) + (mma_id_m * MMA_M) + (lane_id % 16);
        const int col = (mma_id_k * MMA_K) + (lane_id / 16) * 4;
        const int swizzled_col = swizzle<BLOCK_K>(row, col);
        __hip_bfloat16 *addr = A_smem + (row * BLOCK_K + swizzled_col);
        A_rmem[mma_id_m][mma_id_k] = reinterpret_cast<s16x4 *>(addr)[0];
      }
    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++)
      for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++) {
        const int row = (warp_id_n * WARP_N) + (mma_id_n * MMA_N) + (lane_id % 16);
        const int col = (mma_id_k * MMA_K) + (lane_id / 16) * 4;
        const int swizzled_col = swizzle<BLOCK_K>(row, col);
        __hip_bfloat16 *addr = B_smem + (row * BLOCK_K + swizzled_col);
        B_rmem[mma_id_n][mma_id_k] = reinterpret_cast<s16x4 *>(addr)[0];
      }

    // mma
    // https://github.com/ROCm/composable_kernel/blob/rocm-7.0.1/include/ck/utility/amd_xdlops.hpp
    // https://github.com/tile-ai/tilelang/blob/v0.1.6.post1/src/tl_templates/hip/gemm.h
    // TODO: swap A and B like in tilelang for better C layout
    for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
      for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++)
        for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++)
          C_rmem[mma_id_m][mma_id_n] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A_rmem[mma_id_m][mma_id_k],
                                                                                 B_rmem[mma_id_n][mma_id_k],
                                                                                 C_rmem[mma_id_m][mma_id_n],
                                                                                 0, 0, 0);
  }

  __syncthreads();
  for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 16) * 4;
      const int col = mma_id_n * MMA_N + (lane_id % 16);

      fp32x4 data = C_rmem[mma_id_m][mma_id_n];
      C_gmem[(row + 0) * N + col] = __float2bfloat16(data[0]);
      C_gmem[(row + 1) * N + col] = __float2bfloat16(data[1]);
      C_gmem[(row + 2) * N + col] = __float2bfloat16(data[2]);
      C_gmem[(row + 3) * N + col] = __float2bfloat16(data[3]);
    }
}

void matmul_v2(
  const __hip_bfloat16 *A,
  const __hip_bfloat16 *B,
        __hip_bfloat16 *C,
  int M, int N, int K,
  hipStream_t stream
) {
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_N = 128;
  constexpr int BLOCK_K = 64;
  constexpr int GROUP_M = 1;

  constexpr int NUM_WARP_M = 2;
  constexpr int NUM_WARP_N = 2;

  const int grid_m = cdiv(M, BLOCK_M);
  const int grid_n = cdiv(N, BLOCK_N);
  const int grid_size = grid_m * grid_n;

  const int tb_size = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(__hip_bfloat16);

  matmul_v2_kernel<BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_WARP_M, NUM_WARP_N>
    <<<grid_size, tb_size, smem_size, stream>>>(A, B, C, M, N, K);
}
