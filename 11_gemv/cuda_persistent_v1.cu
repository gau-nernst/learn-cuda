#include <cuda_bf16.h>
#include "common.h"

template <int K, int WARP_N, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE, 1)  // occupancy=1 -> encourage more registers
__global__
void cuda_persistent_v1_kernel(
  const nv_bfloat16 *x_ptr,     // [K]
  const nv_bfloat16 *w_ptr,     // [N, K]
        nv_bfloat16 *y_ptr,     // [N]
  int N
) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int bid = blockIdx.x;
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

#if __CUDA_ARCH__ >= 1000
    constexpr int VEC_SIZE = 8;
#else
    constexpr int VEC_SIZE = 4;
#endif

  // each warp handle WARP_N rows per iteration
  for (int off_n = (bid * NUM_WARPS + warp_id) * WARP_N; off_n < N; off_n += gridDim.x * WARP_N) {
    float acc[WARP_N][2] = {};
    constexpr int NUM = VEC_SIZE * 2;

    for (int i = 0; i < K / (WARP_SIZE * NUM); i++) {
      const int col = (i * WARP_SIZE + lane_id) * NUM;
      int x[VEC_SIZE], w[WARP_N][VEC_SIZE];
      float x_f32[NUM], w_f32[WARP_N][NUM];

      ldg_b32<VEC_SIZE>(x, x_ptr + col);
      for (int n = 0; n < WARP_N; n++)
        ldg_b32_fast<VEC_SIZE>(w[n], w_ptr + ((off_n + n) * K + col));

      for (int j = 0; j < VEC_SIZE; j++) {
        bf16x2_to_fp32x2(x_f32 + j * 2, x[j]);

        for (int n = 0; n < WARP_N; n++) {
          bf16x2_to_fp32x2(w_f32[n] + j * 2, w[n][j]);

#if __CUDA_ARCH__ == 1000
          fma_f32x2(acc[n], x_f32 + j * 2, w_f32[n] + j * 2, acc[n]);
#else
          // some ILP. also makes it easier to implement f32x2
          acc[n][0] += x_f32[j * 2 + 0] * w_f32[n][j * 2 + 0];
          acc[n][1] += x_f32[j * 2 + 1] * w_f32[n][j * 2 + 1];
#endif
        }
      }
    }

    for (int n = 0; n < WARP_N; n++) {
      acc[n][0] += acc[n][1];

      // warp reduction
      for (int s = WARP_SIZE / 2; s > 0; s /= 2)
        acc[n][0] += __shfl_down_sync(0xFFFF'FFFF, acc[n][0], s);
    }

    static_assert(WARP_N % 2 == 0);
    if (lane_id == 0) {
      int tmp[WARP_N / 2];
      for (int i = 0; i < WARP_N / 2; i++)
        tmp[i] = fp32x2_to_bf16x2(acc[i * 2][0], acc[i * 2 + 1][0]);

      constexpr int NUM = std::min(WARP_N / 2, VEC_SIZE);
      for (int i = 0; i < WARP_N / 2 / NUM; i++)
        stg_b32_fast<NUM>(y_ptr + (off_n + i * NUM * 2), tmp + i * NUM);
    }
  }
}

void cuda_persistent_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int N, int K) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int num_sms = prop.multiProcessorCount;

  constexpr int NUM_WARPS = 4;
  constexpr int WARP_N = 4;

#define DISPATCH(K_) else if (K == K_) { \
  cuda_persistent_v1_kernel<K_, WARP_N, NUM_WARPS><<<num_sms, NUM_WARPS * WARP_SIZE>>>(A, B, C, N); \
}

  if (false) {}
  DISPATCH(1024)
  DISPATCH(2048)
  DISPATCH(2560)
  DISPATCH(4096)
  DISPATCH(5120)

#undef DISPATCH
}
