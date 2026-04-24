#include <cuda_bf16.h>
#include "common.h"

template <int K, int NUM_WARPS>
__block_size__((NUM_WARPS * WARP_SIZE, 1, 1))
__global__
void cuda_v1_kernel(
  const nv_bfloat16 *x_ptr,     // [K]
  const nv_bfloat16 *w_ptr,     // [N, K]
        nv_bfloat16 *y_ptr      // [N]
) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int bid = blockIdx.x;
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  // each warp handle 1 row in N
  // TODO: handle multiple rows?
  const int row = bid * NUM_WARPS + warp_id;

  float acc[2] = {};

#if __CUDA_ARCH__ >= 1000
  constexpr int NUM = 16;
#else
  constexpr int NUM = 8;
#endif

  for (int i = 0; i < K / (WARP_SIZE * NUM); i++) {
    const int col = (i * WARP_SIZE + lane_id) * NUM;
    int x[NUM / 2], w[NUM / 2];
    float x_f32[NUM], w_f32[NUM];

    ldg_b32_fast<NUM / 2>(w, w_ptr + (row * K + col));
    ldg_b32<NUM / 2>(x, x_ptr + col);

    for (int j = 0; j < NUM / 2; j++) {
      bf16x2_to_fp32x2(w_f32 + j * 2, w[j]);
      bf16x2_to_fp32x2(x_f32 + j * 2, x[j]);

#if __CUDA_ARCH__ == 1000
      fma_f32x2(acc, x_f32 + j * 2, w_f32 + j * 2, acc);
#else
      // some ILP. also makes it easier to implement f32x2
      acc[0] += x_f32[j * 2 + 0] * w_f32[j * 2 + 0];
      acc[1] += x_f32[j * 2 + 1] * w_f32[j * 2 + 1];
#endif
    }
  }

  acc[0] += acc[1];

  // warp reduction
  for (int s = WARP_SIZE / 2; s > 0; s /= 2)
    acc[0] += __shfl_down_sync(0xFFFF'FFFF, acc[0], s);

  if (lane_id == 0)
    y_ptr[row] = __float2bfloat16_rn(acc[0]);
}

void cuda_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int N, int K) {
  constexpr int NUM_WARPS = 4;

#define DISPATCH(K_) else if (K == K_) { \
  cuda_v1_kernel<K_, NUM_WARPS><<<N / NUM_WARPS, NUM_WARPS * WARP_SIZE>>>(A, B, C); \
}

  if (false) {}
  DISPATCH(1024)
  DISPATCH(2048)
  DISPATCH(2560)
  DISPATCH(4096)
  DISPATCH(5120)

#undef DISPATCH
}
