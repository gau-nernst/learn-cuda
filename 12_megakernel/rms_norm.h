#pragma once

#include "common.h"
#include <cuda_bf16.h>

// output is stored in x_smem
template <int DIM, int NUM_WARPS>
__device__ inline
void rms_norm(
  const nv_bfloat16 *x_ptr,  // [DIM]
  const nv_bfloat16 *w_ptr,  // [DIM]
  int x_smem,                // [DIM], fp32
  int workspace_smem         // [NUM_WARPS], fp32
) {
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  float sum_sq = 0.0f;

  auto acc_one = [&](int idx) {
    int x[4];
    float x_f32[8];

    // TODO: cache x
    ldg_b32<4>(x, x_ptr + idx);
    for (int i = 0; i < 4; i++) {
      bf16x2_to_fp32x2(x_f32 + i * 2, x[i]);
      sum_sq += x_f32[i * 2 + 0] * x_f32[i * 2 + 0];
      sum_sq += x_f32[i * 2 + 1] * x_f32[i * 2 + 1];
    }
  };

  // TODO: how much perf is lost if DIM is not constexpr
  static_assert(DIM % 8 == 0);
  for (int i = 0; i < DIM / (TB_SIZE * 8); i++)
    acc_one((i * TB_SIZE + tid) * 8);

  if constexpr (DIM % (TB_SIZE * 8) != 0) {
    const int idx = (DIM / (TB_SIZE * 8) * TB_SIZE + tid) * 8;
    if (idx < DIM)
      acc_one(idx);
  }

  // warp sum
  for (int s = WARP_SIZE / 2; s > 0; s /= 2)
    sum_sq += __shfl_down_sync(0xFFFF'FFFF, sum_sq, s);

  // cross-warps sum
  // store to smem, then single warp does reduction
  if (lane_id == 0)
    sts_b32<1>(workspace_smem + warp_id * 4, &sum_sq);
  __syncthreads();

  static_assert(NUM_WARPS <= WARP_SIZE);
  if (warp_id == 0) {
    if (lane_id < NUM_WARPS)
      lds_b32<1>(&sum_sq, workspace_smem + lane_id * 4);

    for (int s = NUM_WARPS / 2; s > 0; s /= 2)
      sum_sq += __shfl_down_sync(0xFFFF'FFFF, sum_sq, s);

    if (lane_id == 0)
      sts_b32<1>(workspace_smem, &sum_sq);
  }
  __syncthreads();
  lds_b32<1>(&sum_sq, workspace_smem);

  float scale = __frsqrt_rn(sum_sq / DIM + 1e-6f);

  auto process_one = [&](int idx) {
    int x[4], w[4];
    float x_f32[8], w_f32[8];
    ldg_b32<4>(w, w_ptr + idx);
    ldg_b32<4>(x, x_ptr + idx);

    for (int i = 0; i < 4; i++) {
      bf16x2_to_fp32x2(w_f32 + i * 2, w[i]);
      bf16x2_to_fp32x2(x_f32 + i * 2, x[i]);
      x_f32[i * 2 + 0] *= w_f32[i * 2 + 0] * scale;
      x_f32[i * 2 + 1] *= w_f32[i * 2 + 1] * scale;
      x[i] = fp32x2_to_bf16x2(x_f32[i * 2], x_f32[i * 2 + 1]);
    }

    sts_b32<4>(x_smem + idx * 2, x);
  };

  for (int i = 0; i < DIM / (TB_SIZE * 8); i++)
    process_one((i * TB_SIZE + tid) * 8);

  if constexpr (DIM % (TB_SIZE * 8) != 0) {
    const int idx = (DIM / (TB_SIZE * 8) * TB_SIZE + tid) * 8;
    if (idx < DIM)
      process_one(idx);
  }
}
