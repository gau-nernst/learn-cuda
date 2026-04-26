// WIP

#include "common.h"
#include "rms_norm.h"
#include <cuda_bf16.h>
#include <cmath>

// NUM_THREADS threads execute this
template <int HEAD_DIM, int NUM_THREADS>
__device__ inline
void qk_norm_rope(
        nv_bfloat16 *x_ptr,     // [HEAD_DIM]
  const nv_bfloat16 *w_ptr,     // [HEAD_DIM]
  const float       *rope_ptr,  // [HEAD_DIM * 2]
  int tid
) {
  // (HEAD_DIM/2) because we load each half separately for RoPE later.
  // for example, with NUM_THREADS = WARP_SIZE, HEAD_DIM = 128 => NUM = 2
  constexpr int NUM = (HEAD_DIM / 2) / NUM_THREADS;
  static_assert(NUM % 2 == 0);  // to simplify for b32 loads

  int x_lo[NUM / 2], x_hi[NUM / 2], w_lo[NUM / 2], w_hi[NUM / 2];
  float x_lo_f32[NUM], x_hi_f32[NUM], w_lo_f32[NUM], w_hi_f32[NUM];
  float cos[NUM], sin[NUM];

  ldg_b32<NUM>(x_lo, x_ptr + (tid * NUM));
  ldg_b32<NUM>(x_hi, x_ptr + (tid * NUM + HEAD_DIM / 2));
  ldg_b32<NUM>(w_lo, w_ptr + (tid * NUM));
  ldg_b32<NUM>(w_hi, w_ptr + (tid * NUM + HEAD_DIM / 2));
  ldg_b32<NUM * 2>(cos, rope_ptr + (tid * NUM));
  ldg_b32<NUM * 2>(sin, rope_ptr + (tid * NUM + HEAD_DIM / 2));

  float sum_sq = 0.0f;
  for (int i = 0; i < NUM / 2; i++) {
    bf16x2_to_fp32x2(x_lo_f32 + i * 2, x_lo[i]);
    sum_sq += x_lo_f32[i * 2 + 0] * x_lo_f32[i * 2 + 0];
    sum_sq += x_lo_f32[i * 2 + 1] * x_lo_f32[i * 2 + 1];

    bf16x2_to_fp32x2(x_hi_f32 + i * 2, x_hi[i]);
    sum_sq += x_hi_f32[i * 2 + 0] * x_hi_f32[i * 2 + 0];
    sum_sq += x_hi_f32[i * 2 + 1] * x_hi_f32[i * 2 + 1];
  }

  static_assert(NUM_THREADS <= WARP_SIZE);  // so we don't need to do threadblock reduction
  for (int s = NUM_THREADS / 2; s > 0; s /= 2)
    sum_sq += __shfl_xor_sync(0xFFFF'FFFF, sum_sq, s);

  float scale = __frsqrt_rn(sum_sq / HEAD_DIM + 1e-6f);

  for (int i = 0; i < NUM / 2; i++) {
    // apply RMS norm
    bf16x2_to_fp32x2(w_lo_f32 + i * 2, w_lo[i]);
    x_lo_f32[i * 2 + 0] *= scale * w_lo_f32[i * 2 + 0];
    x_lo_f32[i * 2 + 1] *= scale * w_lo_f32[i * 2 + 1];

    bf16x2_to_fp32x2(w_hi_f32 + i * 2, w_hi[i]);
    x_hi_f32[i * 2 + 0] *= scale * w_hi_f32[i * 2 + 0];
    x_hi_f32[i * 2 + 1] *= scale * w_hi_f32[i * 2 + 1];

    // apply RoPE
    float x_lo_0 = x_lo_f32[i * 2 + 0] * cos[i * 2 + 0] - x_hi_f32[i * 2 + 0] * sin[i * 2 + 0];
    float x_hi_0 = x_lo_f32[i * 2 + 0] * sin[i * 2 + 0] + x_hi_f32[i * 2 + 0] * cos[i * 2 + 0];
    float x_lo_1 = x_lo_f32[i * 2 + 1] * cos[i * 2 + 1] - x_hi_f32[i * 2 + 1] * sin[i * 2 + 1];
    float x_hi_1 = x_lo_f32[i * 2 + 1] * sin[i * 2 + 1] + x_hi_f32[i * 2 + 1] * cos[i * 2 + 1];

    x_lo[i] = fp32x2_to_bf16x2(x_lo_0, x_lo_1);
    x_hi[i] = fp32x2_to_bf16x2(x_hi_0, x_hi_1);
  }

  stg_b32<NUM>(x_ptr + (tid * NUM)               , x_lo);
  stg_b32<NUM>(x_ptr + (tid * NUM + HEAD_DIM / 2), x_hi);
}

template <int DIM, int HEAD_DIM, int NUM_HEADS, int NUM_KV_HEADS, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE, 1)  // occupancy=1 -> encourage more registers
__global__
void attn_cuda_v1_kernel(
  const nv_bfloat16 *x_ptr,          // [DIM]
  const nv_bfloat16 *norm_ptr,       // [DIM]
        nv_bfloat16 *kv_cache_ptr,   // [2, max_context, num_kv_heads, HEAD_DIM]
  const nv_bfloat16 *wqkv_ptr,       // [qkv_dim, DIM]
  const nv_bfloat16 *q_norm_ptr,     // [HEAD_DIM]
  const nv_bfloat16 *k_norm_ptr,     // [HEAD_DIM]
  const float       *rope_ptr,       // [HEAD_DIM * 2]
  const nv_bfloat16 *wo_ptr,         // [DIM, Q_DIM]
        char        *workspace_ptr,  // BF16[q_dim]
        int         *flag_ptr,
  int position,
  int max_context
) {
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int Q_DIM = NUM_HEADS * HEAD_DIM;
  constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;
  const int global_warp_id = bid * NUM_WARPS + warp_id;

  const int num_bids_per_head = num_bids / NUM_HEADS;

  constexpr int SMEM_SIZE = DIM * 2 + NUM_WARPS * 4
                          + Q_DIM * 2;
  __shared__ char smem_ptr[SMEM_SIZE];
  const int smem = __cvta_generic_to_shared(smem_ptr);

  const int x_smem         = smem;              // size: bf16[DIM]
  const int workspace_smem = x_smem + DIM * 2;  // size: fp32[NUM_WARPS]

  nv_bfloat16 *q_tmp_ptr = reinterpret_cast<nv_bfloat16 *>(workspace_ptr);         // size: bf16[Q_DIM]
  float *attn_o_ptr      = reinterpret_cast<float *>(q_tmp_ptr + Q_DIM);           // size: fp32[num_bids_per_head][num_heads][HEAD_DIM]
  float *attn_other_ptr  = attn_o_ptr + num_bids_per_head * NUM_HEADS * HEAD_DIM;  // size: fp32[num_bids_per_head][num_heads][2]

  // all threadblocks perform input norm to avoid grid sync.
  rms_norm<DIM, NUM_WARPS>(x_ptr, norm_ptr, x_smem, workspace_smem);
  __syncthreads();

  // QKV = x @ w_qkv
  for (int off_n = global_warp_id; off_n < MLP_DIM; off_n += num_bids * NUM_WARPS) {
    float acc = 0.0f;

    static_assert(DIM % (WARP_SIZE * 8) == 0);
    for (int i = 0; i < MLP_DIM / (WARP_SIZE * 8); i++) {
      const int col = (i * WARP_SIZE + lane_id) * 8;
      int x[4], w[4];
      float x_f32[8], w_f32[8];

      ldg_b32_fast<4>(w, w2_ptr + (off_n * MLP_DIM + col));
      lds_b32<4>(x, workspace_smem + col * 2);

      for (int j = 0; j < 4; j++) {
        bf16x2_to_fp32x2(w_f32 + j * 2, w[j]);
        bf16x2_to_fp32x2(x_f32 + j * 2, x[j]);
        acc += w_f32[j * 2 + 0] * x_f32[j * 2 + 0];
        acc += w_f32[j * 2 + 1] * x_f32[j * 2 + 1];
      }
    }

    // warp reduction
    for (int s = WARP_SIZE / 2; s > 0; s /= 2)
      acc += __shfl_down_sync(0xFFFF'FFFF, acc, s);

    if (lane_id == 0) {
      nv_bfloat16 y = __float2bfloat16_rn(acc);

      // store Q to tmp buffer, and KV to KV cache
      if (off_n < Q_DIM)
        q_tmp_ptr[off_n] = y;
      else if (off_n < Q_DIM + KV_DIM)
        kv_cache_ptr[position * KV_DIM + (off_n - Q_DIM)] = y;
      else
        kv_cache_ptr[(max_context + position) * KV_DIM + (off_n - (Q_DIM + KV_DIM))] = y;
    }
  }

  __syncthreads();  // all threadblocks finish
  if (tid == 0) {
    atomic_add_release_gpu(flag_ptr, 1);  // signal done
    while (load_acquire_gpu(flag_ptr) != num_bids) {}
  }
  __syncthreads();

  // QK norm
  // all threadblocks normalize Q independently?
  // 1 warp normalizes 1 head
  {
    if (global_warp_id < NUM_HEADS) {
      const int head_id = global_warp_id;
      qk_norm_rope<HEAD_DIM, WARP_SIZE>(q_tmp_ptr + head_id * HEAD_DIM, q_norm_ptr, rope_ptr);
      __syncthreads();
      if (lane_id == 0)
        atomic_add_release_gpu(flag_ptr + 1, 1);  // signal done
    }
    else if (global_warp_id < NUM_HEADS + NUM_KV_HEADS) {
      const int k_head_id = global_warp_id - NUM_HEADS;
      qk_norm_rope<HEAD_DIM, WARP_SIZE>(kv_cache_ptr + (position * KV_DIM + k_head_id * HEAD_DIM), k_norm_ptr, rope_ptr);
      __syncthreads();
      if (lane_id == 0)
        atomic_add_release_gpu(flag_ptr + 1, 1);  // signal done
    }

  }
  __syncthreads();
  if (tid == 0) {
    while (load_acquire_gpu(flag_ptr + 1) != NUM_HEADS + NUM_KV_HEADS) {}
  }
  __syncthreads();

  // attention
  // each threadblock = 1 attention head
  if (bid < num_bids_per_head * NUM_HEADS) {
    const int head_id = bid % NUM_HEADS;
    const int kv_head_id = head_id / (NUM_HEADS / NUM_KV_HEADS);

    //       element count    |        thread count
    // Q:          [HEAD_DIM] |          [THREADS_PER_TOK]
    // K: [BLOCK_L, HEAD_DIM] | [BLOCK_L, THREADS_PER_TOK]
    // V: [BLOCK_L, HEAD_DIM] | [BLOCK_L, THREADS_PER_TOK]
    //
    // load 8 BF16 elements at a time (16B)
    // for HEAD_DIM = 8, NUM_WARPS = 8 => THREADS_PER_TOK = 16, BLOCK_L = 16
    constexpr int THREADS_PER_TOK = HEAD_DIM / 8;
    constexpr int BLOCK_L = TB_SIZE / THREADS_PER_TOK;
    static_assert(THREADS_PER_TOK <= WARP_SIZE);  // reduction along HEAD_DIM is within-warp only

    const int col = tid % THREADS_PER_TOK;
    const int row = tid / THREADS_PER_TOK;

    // offset from kv_cache_ptr
    const int k_offset = row * KV_DIM + kv_head_id * HEAD_DIM + col * 8;
    const int v_offset = k_cache_offset + max_context * KV_DIM;

    // load Q[HEAD_DIM]
    int q[4];
    int q_f32[8];
    ldg_b32<4>(q, (q_tmp_ptr + head_id * HEAD_DIM + col * 8));
    for (int i = 0; i < 4; i++) {
      bf16x2_to_fp32x2(q_f32 + i * 2, q[i]);

      // ex2 scaling and softmax scale
      constexpr float q_scale = 1.4426950408889634 * std::rsqrt(HEAD_DIM);
      q_f32[i * 2 + 0] *= q_scale;
      q_f32[i * 2 + 1] *= q_scale;
    }

    // initialize attention state
    float max_s = -1e9f;   // [BLOCK_L] - don't use (-inf) to avoid (-inf) - (-inf) = NaN
    float sum_exp = 0.0f;  // [BLOCK_L]
    float o[8] = {};   // [BLOCK_L, HEAD_DIM]

    // iterate over KV cache
    const int num_kv_blocks = cdiv(position + 1, BLOCK_L);
    for (int kv_block = bid / NUM_HEADS; kv_block < num_kv_blocks; kv_block += num_bids_per_head) {
      int k[4], v[4];
      int k_f32[8], v_f32[8];
      ldg_b32<4>(k, kv_cache_ptr + (k_offset + kv_block * BLOCK_L * KV_DIM));
      ldg_b32<4>(v, kv_cache_ptr + (v_offset + kv_block * BLOCK_L * KV_DIM));

      // S = Q @ K.T
      //   shape: [BLOCK_L]
      float s = 0.0f;
      if (kv_block * BLOCK_L + row < position + 1) {
        // thread-level reduction
        for (int i = 0; i < 4; i++) {
          bf16x2_to_fp32x2(k_f32 + i * 2, k[i]);
          s += q_f32[i * 2 + 0] * k_f32[i * 2 + 0];
          s += q_f32[i * 2 + 1] * k_f32[i * 2 + 1];
        }
      }
      else {
        s = -INFINITY;
      }
      // reduction within THREADS_PER_TOK threads
      for (int stride = THREADS_PER_TOK / 2; stride > 0; stride /= 2)
        s += __shfl_xor_sync(0xFFFF'FFFF, s, stride);

      // new_max_S = maximum(max_S, S)
      //   shape: [BLOCK_L]
      // NOTE: we DON'T perform reduction along BLOCK_L dim here to avoid cross-warp communication.
      float new_max_s = max(max_s, s);
      float rescale = __exp2f(max_s - new_max_s);  // rescale=1.0 when we are out-of-bounds
      max_s = new_max_s;

      // P = exp(S - max_S)
      //   shape: [BLOCK_L]
      float p = __exp2f(s - max_s);  // p=0.0 when we are out-of-bounds
      sum_exp = sum_exp * rescale + p;

      // O (unreduced) = P * V
      //   shape: [BLOCK_L, HEAD_DIM]
      // we will do reduction along [BLOCK_L] after the loop
      if (kv_block * BLOCK_L + row < position + 1) {
        for (int i = 0; i < 4; i++) {
          bf16x2_to_fp32x2(v_f32 + i * 2, v[i]);
          o[i * 2 + 0] = o[i * 2 + 0] * rescale + p * v_f32[i * 2 + 0];
          o[i * 2 + 1] = o[i * 2 + 1] * rescale + p * v_f32[i * 2 + 1];
        }
      }
    }

    // recap
    //   max_S:   [BLOCK_L]
    //   sum_exp: [BLOCK_L]
    //   O:       [BLOCK_L, HEAD_DIM]
    // reduction along BLOCK_L dim
    // reduction within a warp first
    if constexpr (THREADS_PER_TOK < WARP_SIZE) {
      float new_max_s = max_s;
      for (int stride = THREADS_PER_TOK; stride < WARP_SIZE; stride *= 2)
        new_max_s = max(new_max_s, __shfl_xor_sync(0xFFFF'FFFF, new_max_s, stride));

      float rescale = __exp2f(max_s - new_max_s);
      sum_exp *= rescale;
      for (int i = 0; i < 8; i++)
        o[i] *= rescale;

      for (int stride = THREADS_PER_TOK; stride < WARP_SIZE; stride *= 2) {
        sum_exp += __shfl_xor_sync(0xFFFF'FFFF, sum_exp, stride);
        for (int i = 0; i < 8; i++)
          o[i] += __shfl_xor_sync(0xFFFF'FFFF, o[i], stride);
      }
    }

    // now we have
    //   max_S:   [NUM_WARPS]
    //   sum_exp: [NUM_WARPS]
    //   O:       [NUM_WARPS, HEAD_DIM]
    // store to smem
    __shared__ float max_s_smem[NUM_WARPS];
    __shared__ float sum_exp_smem[NUM_WARPS];
    __shared__ float o_smem[NUM_WARPS * HEAD_DIM];
    if (lane_id == 0) {
      max_s_smem[warp_id] = max_s;
      sum_exp_smem[warp_id] = sum_exp;
    }
    if (lane_id < THREADS_PER_TOK) {
      reinterpret_cast<int4 *>(o_smem + (warp_id * HEAD_DIM + col * 8 + 0))[0] = reinterpret_cast<int4 *>(o_smem + 0)[0]
      reinterpret_cast<int4 *>(o_smem + (warp_id * HEAD_DIM + col * 8 + 4))[0] = reinterpret_cast<int4 *>(o_smem + 4)[0]
    }
    __syncthreads();

    // for the final reduction, we want to split the work of O to the whole threadblock
    // our "subwarp" now has size of NUM_WARPS
    static_assert(NUM_WARPS <= WARP_SIZE);

    // compute rescale factor on all subwarps
    max_s = max_s_smem[tid % NUM_WARPS];
    float new_max_s = max_s;
    for (int stride = NUM_WARPS / 2; stride > 0; stride /= 2)
      new_max_s = max(new_max_s, __shfl_xor_sync(0xFFFF'FFFF, new_max_s, stride));
    float rescale = __exp2f(max_s - new_max_s);

    // 1st subwarp compute final sum_exp (gated with warp_id since we use warp shuffle)
    if (warp_id == 0) {
      sum_exp = sum_exp_smem[tid % NUM_WARPS];
      sum_exp *= rescale;
      for (int stride = NUM_WARPS / 2; stride > 0; stride /= 2)
        sum_exp += __shfl_xor_sync(0xFFFF'FFFF, sum_exp, stride);

      if (lane_id == 0) {
        // fp32[num_bids_per_head][num_heads][2]
        attn_other_ptr[bid * 2 + 0] = new_max_s;
        attn_other_ptr[bid * 2 + 1] = sum_exp;
      }
    }

    // since our subwarp has NUM_WARPS threads, the whole threadblock has WARP_SIZE subwarps
    // divide by 4 since we load 4 elems at a time
    for (int i = 0; i < HEAD_DIM / (WARP_SIZE / 4); i++) {
      const int col = (i * WARP_SIZE + (tid / NUM_WARPS)) * 4;
      int4 o_tmp = reinterpret_cast<int4 *>(o_smem + ((tid % NUM_WARPS) * HEAD_DIM + col))[0];
      o_tmp.x *= rescale;
      o_tmp.y *= rescale;
      o_tmp.z *= rescale;
      o_tmp.w *= rescale;
      for (int stride = NUM_WARPS / 2; stride > 0; stride /= 2) {
        o_tmp.x += __shfl_xor_sync(0xFFFF'FFFF, o_tmp.x, stride);
        o_tmp.y += __shfl_xor_sync(0xFFFF'FFFF, o_tmp.y, stride);
        o_tmp.z += __shfl_xor_sync(0xFFFF'FFFF, o_tmp.z, stride);
        o_tmp.w += __shfl_xor_sync(0xFFFF'FFFF, o_tmp.w, stride);
      }

      // size: fp32[num_bids_per_head][num_heads][HEAD_DIM]
      if (tid % NUM_WARPS == 0)
        reinterpret_cast<int4 *>(attn_o_ptr + (bid * HEAD_DIM + col))[0] = o_tmp;
    }

    __syncthreads();
    if (tid == 0)
      atomic_add_release_gpu(flag_ptr + 2, 1);  // signal done
  }
  if (tid == 0) {
    while (load_acquire_gpu(flag_ptr + 2) != num_bids_per_head * NUM_HEADS) {}
  }
  __syncthreads();

  // combine attention states across threadblocks
  // max_S and sum_exp: [num_bids_per_head][num_heads][2]
  // O:                 [num_bids_per_head][num_heads][HEAD_DIM]
  if (bid < NUM_HEADS) {
    // what's the typical value for num_bids_per_head?
    // PRO6000 has 188 SMs (the highest)
    // Qwen3-0.6B has 16 heads, Qwen3.5-0.8B has 8 heads
    // -> num_bids_per_head max ~ 23
    // thus, it's unlikely that num_bids_per_head > WARP_SIZE

    __syncthreads();
    if (tid == 0)
      atomic_add_release_gpu(flag_ptr + 3, 1);  // signal done
  }
  if (tid == 0) {
    while (load_acquire_gpu(flag_ptr + 3) != NUM_HEADS) {}
  }
  __syncthreads();

  // O projection

}
