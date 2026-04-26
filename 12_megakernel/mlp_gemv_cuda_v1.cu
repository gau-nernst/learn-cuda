#include "common.h"
#include "rms_norm.h"
#include <cuda_bf16.h>
#include <ATen/ATen.h>

template <int DIM, int MLP_DIM, int WARP_N13, int WARP_N2, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE, 1)  // occupancy=1 -> encourage more registers
__global__
void mlp_gemv_cuda_v1_kernel(
  const nv_bfloat16 *x_ptr,          // [dim]
  const nv_bfloat16 *norm_ptr,       // [dim]
  const nv_bfloat16 *w13_ptr,        // [mlp_dim * 2, dim]
  const nv_bfloat16 *w2_ptr,         // [dim, mlp_dim]
        nv_bfloat16 *o_ptr,          // [dim]
        nv_bfloat16 *workspace_ptr,  // [mlp_dim]
        int         *flag_ptr        // [2]
) {
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  __shared__ char smem_ptr[DIM * 2 + NUM_WARPS * 4];
  const int smem = __cvta_generic_to_shared(smem_ptr);

  const int x_smem         = smem;  // size: bf16[DIM]
  const int workspace_smem = x_smem + DIM * 2;  // size: fp32[NUM_WARPS]

  // all threadblocks perform input norm to avoid grid sync.
  rms_norm<DIM, NUM_WARPS>(x_ptr, norm_ptr, x_smem, workspace_smem);
  __syncthreads();

  // 1st matmul
  const nv_bfloat16 *w1_ptr = w13_ptr;
  const nv_bfloat16 *w3_ptr = w13_ptr + MLP_DIM * DIM;

  // each warp handle WARP_N13 rows
  static_assert(MLP_DIM % WARP_N13 == 0);
  for (int off_n = (bid * NUM_WARPS + warp_id) * WARP_N13; off_n < MLP_DIM; off_n += num_bids * NUM_WARPS * WARP_N13) {
    float gate[WARP_N13] = {}, up[WARP_N13] = {};

    static_assert(DIM % (WARP_SIZE * 8) == 0);
    for (int i = 0; i < DIM / (WARP_SIZE * 8); i++) {
      const int col = (i * WARP_SIZE + lane_id) * 8;
      int w1[WARP_N13][4], w3[WARP_N13][4], x[4];
      float w1_f32[WARP_N13][8], w3_f32[WARP_N13][8], x_f32[8];

      for (int n = 0; n < WARP_N13; n++) {
        ldg_b32_fast<4>(w1[n], w1_ptr + ((off_n + n) * DIM + col));
        ldg_b32_fast<4>(w3[n], w3_ptr + ((off_n + n) * DIM + col));
      }
      lds_b32<4>(x, x_smem + col * 2);

      for (int j = 0; j < 4; j++) {
        bf16x2_to_fp32x2(x_f32 + j * 2, x[j]);

        for (int n = 0; n < WARP_N13; n++) {
          bf16x2_to_fp32x2(w1_f32[n] + j * 2, w1[n][j]);
          gate[n] += w1_f32[n][j * 2 + 0] * x_f32[j * 2 + 0];
          gate[n] += w1_f32[n][j * 2 + 1] * x_f32[j * 2 + 1];

          bf16x2_to_fp32x2(w3_f32[n] + j * 2, w3[n][j]);
          up[n] += w3_f32[n][j * 2 + 0] * x_f32[j * 2 + 0];
          up[n] += w3_f32[n][j * 2 + 1] * x_f32[j * 2 + 1];
        }
      }
    }

    // warp reduction
    for (int s = WARP_SIZE / 2; s > 0; s /= 2)
      for (int n = 0; n < WARP_N13; n++) {
        gate[n] += __shfl_down_sync(0xFFFF'FFFF, gate[n], s);
        up[n] += __shfl_down_sync(0xFFFF'FFFF, up[n], s);
      }

    if (lane_id == 0) {
      float y[WARP_N13];
      for (int n = 0; n < WARP_N13; n++)
        y[n] = gate[n] * up[n] * __frcp_rn(1.0f + __expf(-gate[n]));

      if constexpr (WARP_N13 % 2 == 0) {
        int tmp[WARP_N13 / 2];
        for (int i = 0; i < WARP_N13 / 2; i++)
          tmp[i] = fp32x2_to_bf16x2(y[i * 2], y[i * 2 + 1]);

        constexpr int NUM = std::min(WARP_N13 / 2, 4);
        for (int i = 0; i < WARP_N13 / 2 / NUM; i++)
          stg_b32<NUM>(workspace_ptr + (off_n + i * NUM * 2), tmp + i * NUM);
      }
      else {
        for (int n = 0; n < WARP_N13; n++)
          workspace_ptr[off_n + n] = __float2bfloat16_rn(y[n]);
      }
    }
  }

  __syncthreads();  // all threadblocks finish
  if (tid == 0) {
    atomic_add_release_gpu(flag_ptr, 1);  // signal done
    while (load_acquire_gpu(flag_ptr) != num_bids) {}
  }
  __syncthreads();

  // 2nd matmul
  // each warp handle WARP_N2 rows
  static_assert(DIM % (WARP_N2) == 0);
  for (int off_n = (bid * NUM_WARPS + warp_id) * WARP_N2; off_n < DIM; off_n += num_bids * NUM_WARPS * WARP_N2) {
    float acc[WARP_N2] = {};

    static_assert(MLP_DIM % (WARP_SIZE * 8) == 0);
    for (int i = 0; i < MLP_DIM / (WARP_SIZE * 8); i++) {
      const int col = (i * WARP_SIZE + lane_id) * 8;
      int x[4], w[WARP_N2][4];
      float x_f32[8], w_f32[WARP_N2][8];

      for (int n = 0; n < WARP_N2; n++)
        ldg_b32_fast<4>(w[n], w2_ptr + ((off_n + n) * MLP_DIM + col));
      ldg_b32<4>(x, workspace_ptr + col);

      for (int j = 0; j < 4; j++) {
        bf16x2_to_fp32x2(x_f32 + j * 2, x[j]);

        for (int n = 0; n < WARP_N2; n++) {
          bf16x2_to_fp32x2(w_f32[n] + j * 2, w[n][j]);
          acc[n] += w_f32[n][j * 2 + 0] * x_f32[j * 2 + 0];
          acc[n] += w_f32[n][j * 2 + 1] * x_f32[j * 2 + 1];
        }
      }
    }

    // warp reduction
    for (int s = WARP_SIZE / 2; s > 0; s /= 2)
      for (int n = 0; n < WARP_N2; n++)
        acc[n] += __shfl_down_sync(0xFFFF'FFFF, acc[n], s);

    if (lane_id == 0) {
      if constexpr (WARP_N2 % 2 == 0) {
        int tmp[WARP_N2 / 2];
        for (int i = 0; i < WARP_N2 / 2; i++) {
          int x[1];
          float x_f32[2];
          ldg_b32<1>(x, x_ptr + (off_n + i * 2));
          bf16x2_to_fp32x2(x_f32, x[0]);
          x_f32[0] += acc[i * 2 + 0];
          x_f32[1] += acc[i * 2 + 1];

          tmp[i] = fp32x2_to_bf16x2(x_f32[0], x_f32[1]);
        }

        constexpr int NUM = std::min(WARP_N2 / 2, 4);
        for (int i = 0; i < WARP_N2 / 2 / NUM; i++)
          stg_b32<NUM>(o_ptr + (off_n + i * NUM * 2), tmp + i * NUM);
      }
      else {
        for (int n = 0; n < WARP_N2; n++) {
          float out = __bfloat162float(x_ptr[off_n + n]) + acc[n];
          o_ptr[off_n + n] = __float2bfloat16_rn(out);
        }
      }
    }
  }

  // reset
  __syncthreads();  // all threadblocks finish
  if (tid == 0) {
    if (atomic_add_release_gpu(flag_ptr + 1, 1) == num_bids - 1) {
      flag_ptr[0] = 0;
      flag_ptr[1] = 0;
    }
  }
}

int *flag_ptr = nullptr;

at::Tensor mlp_gemv_cuda_v1(
  const at::Tensor& x,
  const at::Tensor& norm,
  const at::Tensor& w13,
  const at::Tensor& w2
) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int num_sms = prop.multiProcessorCount;

  // allocate flag_ptr (not thread-safe...)
  if (flag_ptr == nullptr) {
    cudaMalloc(&flag_ptr, 2 * 4);
    cudaMemset(flag_ptr, 0, 2 * 4);
  }

  const int dim = w2.size(0);
  const int mlp_dim = w2.size(1);

  at::Tensor o         = at::empty_like(x);
  at::Tensor workspace = at::empty({mlp_dim}, x.options());

  auto x_ptr         = reinterpret_cast<const nv_bfloat16 *>(x.data_ptr());
  auto norm_ptr      = reinterpret_cast<const nv_bfloat16 *>(norm.data_ptr());
  auto w13_ptr       = reinterpret_cast<const nv_bfloat16 *>(w13.data_ptr());
  auto w2_ptr        = reinterpret_cast<const nv_bfloat16 *>(w2.data_ptr());
  auto o_ptr         = reinterpret_cast<      nv_bfloat16 *>(o.data_ptr());
  auto workspace_ptr = reinterpret_cast<      nv_bfloat16 *>(workspace.data_ptr());

  constexpr int NUM_WARPS = 8;
  constexpr int WARP_N13 = 1;  // WARP_N>1 is always worse?
  constexpr int WARP_N2 = 1;
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

#define DISPATCH(DIM, MLP_DIM) else if (dim == DIM && mlp_dim == MLP_DIM) { \
  mlp_gemv_cuda_v1_kernel<DIM, MLP_DIM, WARP_N13, WARP_N2, NUM_WARPS><<<num_sms, TB_SIZE>>>(  \
    x_ptr, norm_ptr, w13_ptr, w2_ptr, o_ptr, workspace_ptr, flag_ptr); \
}

  if (false) {}
  DISPATCH(1024, 3072)

#undef DISPATCH

  return o;
}
