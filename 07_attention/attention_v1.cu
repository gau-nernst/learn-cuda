#include "common.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <float.h>
#include <iostream>

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attention_v1_kernel(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,        // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv) {

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // each threadblock handles 1 BLOCK_Q
  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int bs_id = bid / num_q_blocks;
  const int q_block_id = bid % num_q_blocks;

  Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;
  K += bs_id * len_kv * DIM;
  V += bs_id * len_kv * DIM;
  O += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;

  // we overlap Q_shm with (K_shm + V_shm), since we only need to load Q_shm once
  extern __shared__ nv_bfloat16 shm[];
  const uint32_t Q_shm = __cvta_generic_to_shared(shm);
  const uint32_t K_shm = Q_shm;
  const uint32_t V_shm = K_shm + BLOCK_KV * DIM * sizeof(nv_bfloat16);

  // FA2: shard BLOCK_Q among all warps
  // replicate K and V on all warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // mma.m16n8k16
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE;
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE;
  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;

  // set up registers
  uint32_t Q_regs[WARP_Q / MMA_M][DIM / MMA_K][num_A_regs];
  uint32_t K_regs[BLOCK_KV / MMA_N][DIM / MMA_K][num_B_regs];

  // let compiler decide register reuse?
  uint32_t P_regs[WARP_Q / MMA_M][BLOCK_KV / MMA_K][num_A_regs];
  uint32_t V_regs[BLOCK_KV / MMA_K][DIM / MMA_N][num_B_regs];

  // we use the same registers for O_regs and PV_regs
  // rescale O_regs once we obtain new rowmax, then accumulate to O_regs
  float O_regs[WARP_Q / MMA_M][DIM / MMA_N][num_acc_regs] = {};

  const float softmax_scale = rsqrtf(static_cast<float>(DIM));

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // load Q [BLOCK_Q, DIM]
  global_to_shared<BLOCK_Q, DIM, TB_SIZE>(Q_shm, Q, DIM, tid);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // shared -> registers
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id % 16);
      const int col = mma_id_d * MMA_K + (lane_id / 16 * 8);
      const uint32_t addr = Q_shm + (row * DIM + col) * sizeof(nv_bfloat16);
      ldmatrix_x4(Q_regs[mma_id_q][mma_id_d], addr);
    }
  // we need a syncthreads() here so that we don't load K global->shared
  // before finishing loading Q shared->reg
  __syncthreads();

  for (int off_kv = 0; off_kv < len_kv; off_kv += BLOCK_KV) {
    float QK_regs[WARP_Q / MMA_M][BLOCK_KV / MMA_N][num_acc_regs] = {};

    // load K [BLOCK_KV, DIM]
    global_to_shared<BLOCK_KV, DIM, TB_SIZE>(K_shm, K, DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // shared -> registers
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
        const int row = mma_id_kv * MMA_N + (lane_id % 8);
        const int col = mma_id_d * MMA_K + (lane_id / 8 * 8);
        const uint32_t addr = K_shm + (row * DIM + col) * sizeof(nv_bfloat16);
        ldmatrix_x2(K_regs[mma_id_kv][mma_id_d], addr);
      }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16(Q_regs[mma_id_q][mma_id_d],
                       K_regs[mma_id_kv][mma_id_d],
                       QK_regs[mma_id_q][mma_id_kv]);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // apply softmax scale
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int reg_id = 0; reg_id < num_acc_regs; reg_id++)
          QK_regs[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

      // rowmax
      float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = QK_regs[mma_id_q][mma_id_kv];
        this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));  // c0 and c1
        this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));  // c2 and c3
      }

      // butterfly reduction within 4 threads
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      // new rowmax
      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      // rescale for previous O
      float rescale[2];
      rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        O_regs[mma_id_q][mma_id_d][0] *= rescale[0];
        O_regs[mma_id_q][mma_id_d][1] *= rescale[0];
        O_regs[mma_id_q][mma_id_d][2] *= rescale[1];
        O_regs[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      // save new rowmax
      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      // rowsumexp
      float this_rowsumexp[2] = {};
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = QK_regs[mma_id_q][mma_id_kv];
        regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);  // c0
        regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);  // c1
        regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);  // c2
        regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);  // c3

        this_rowsumexp[0] += regs[0] + regs[1];
        this_rowsumexp[1] += regs[2] + regs[3];

        // pack to P registers for next MMA
        // we need to change from m16n8 to m16k16
        nv_bfloat162 *this_P_regs = reinterpret_cast<nv_bfloat162 *>(P_regs[mma_id_q][mma_id_kv / 2]);
        this_P_regs[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
        this_P_regs[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
      }

      // butterfly reduction within 4 threads
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // accumulate to total rowsumexp
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }

    // load V [BLOCK_KV, DIM]
    // NOTE: we can schedule to load V global->shared earlier
    global_to_shared<BLOCK_KV, DIM, TB_SIZE>(V_shm, V, DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // shared -> registers
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        const int row = mma_id_kv * MMA_K + (lane_id % 16);
        const int col = mma_id_d * MMA_N + (lane_id / 16) * 8;
        const uint32_t addr = V_shm + (row * DIM + col) * sizeof(nv_bfloat16);
        ldmatrix_x2_trans(V_regs[mma_id_kv][mma_id_d], addr);
      }

    // MMA P = S @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k16(P_regs[mma_id_q][mma_id_kv],
                       V_regs[mma_id_kv][mma_id_d],
                       O_regs[mma_id_q][mma_id_d]);

    K += BLOCK_KV * DIM;
    V += BLOCK_KV * DIM;
  }

  // write to O
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
      nv_bfloat16 *O_ptr = O + row * DIM + col;

      // divide by softmax denominator
      float *regs = O_regs[mma_id_q][mma_id_d];
      regs[0] /= rowsumexp[mma_id_q][0];
      regs[1] /= rowsumexp[mma_id_q][0];
      regs[2] /= rowsumexp[mma_id_q][1];
      regs[3] /= rowsumexp[mma_id_q][1];

      reinterpret_cast<nv_bfloat162 *>(O_ptr)[0]           = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(O_ptr + 8 * DIM)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void attention_v1(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,        // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv,
  int dim) {

  if (dim != 128) {
    std::cerr << "Unsupported dim=" << dim << std::endl;
    exit(1);
  }

  const int BLOCK_Q = 128;
  const int BLOCK_KV = 64;
  const int DIM = 128;
  const int NUM_WARPS = 4;

  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int shm_size = max(BLOCK_Q, BLOCK_KV * 2) * DIM * sizeof(nv_bfloat16);

  auto kernel = attention_v1_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;
  launch_kernel(kernel, num_blocks, TB_SIZE, shm_size, Q, K, V, O, bs, len_q, len_kv);
}
