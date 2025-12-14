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
  if (bid == 0 && tid == 0){
    printf("Attention v1 kernel launched: BLOCK_Q=%d, BLOCK_KV=%d, DIM=%d, NUM_WARPS=%d\n",
           BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS);
  }

  // each threadblock handles 1 BLOCK_Q
  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int bs_id = bid / num_q_blocks;
  const int q_block_id = bid % num_q_blocks;

  Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;
  K += bs_id * len_kv * DIM;
  V += bs_id * len_kv * DIM;
  O += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;

  // we overlap Q_smem with (K_smem + V_smem), since we only need to load Q_smem once
  extern __shared__ nv_bfloat16 smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;
  const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);

  // FA2: shard BLOCK_Q among all warps
  // replicate K and V on all warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // mma.m16n8k16
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  // set up registers
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

  // let compiler decide register reuse?
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

  // rescale O_rmem once we obtain new rowmax, then accumulate to O_rmem for P @ V
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

  const float softmax_scale = rsqrtf(static_cast<float>(DIM));

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // load Q [BLOCK_Q, DIM]
  global_to_shared<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();
  
  if(isthread0()){
    // Debug: print first 8 elements of Q_smem
    printf("Q_smem first 8 elements:\n");
    for(int q_row=0; q_row < BLOCK_Q; q_row+=1){
      printf("Row %d: ", q_row);
      for(int i=0; i < 32; i++){
        nv_bfloat16 val = smem[q_row * DIM + i];
        float fval = __bfloat162float(val); 
        // print with 2 decimal places
        printf("%.1f ", fval);
      }
      printf("\n");
    }
  }
  
  int num_m_per_warp = WARP_Q / MMA_M;
  int num_k_per_warp = DIM / MMA_K;
  
  if (isthread0()){
    printf("WARP_Q, %d, DIM, %d, num_m_per_warp, %d, num_k_per_warp, %d \n", 
        WARP_Q, DIM, num_m_per_warp, num_k_per_warp
    );
    // uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
    // uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

    // let compiler decide register reuse?
    // uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
    // uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

    printf("Q_rmem size: WARP_Q / MMA_M, %d, DIM/MMA_K: %d, 4 \n", WARP_Q / MMA_M, DIM / MMA_K);
    printf("K_rmem size: BLOCK_KV / MMA_N, %d, DIM / MMA_K, %d, 2 \n", BLOCK_KV / MMA_N, DIM / MMA_K);
    printf("P_rmem size: WARP_Q / MMA_M, %d, BLOCK_KV / MMA_K, %d, 4 \n", WARP_Q / MMA_M, BLOCK_KV / MMA_K);
    printf("V_rmem size: BLOCK_KV / MMA_K, %d, DIM / MMA_N, %d, 2 \n", BLOCK_KV / MMA_K, DIM / MMA_N);
  }
    __syncthreads();
  // shared -> registers
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
    //   const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id % 16);
    //   const int col = mma_id_d * MMA_K + (lane_id / 16 * 8);
      const int row_start = warp_id * WARP_Q + mma_id_q * MMA_M;
      const int col_start = mma_id_d * MMA_K;
      const int row = row_start + (lane_id % 16);
      const int col = col_start + (lane_id / 16 * 8);
    //   printf("tid, %d, lane_id, %d mma_id_q, %d, mma_id_d, %d, row_start, %d, col_start, %d, row, %d, col, %d \n", tid,
    //          lane_id, mma_id_q, mma_id_d, row_start, col_start, row, col);
      const uint32_t addr = Q_smem + (row * DIM + col) * sizeof(nv_bfloat16);
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
      __syncthreads();
      if (mma_id_q == 0 && mma_id_d == 0 && warp_id == 0) {
        auto cur_q_reg = Q_rmem[mma_id_q][mma_id_d];
        uint32_t _q0 = cur_q_reg[0];
        uint32_t _q1 = cur_q_reg[1];
        uint32_t _q2 = cur_q_reg[2];
        uint32_t _q3 = cur_q_reg[3];
        nv_bfloat162 all_q[4];
        all_q[0] = *reinterpret_cast<nv_bfloat162 *>(&_q0);
        all_q[1] = *reinterpret_cast<nv_bfloat162 *>(&_q1);
        all_q[2] = *reinterpret_cast<nv_bfloat162 *>(&_q2);
        all_q[3] = *reinterpret_cast<nv_bfloat162 *>(&_q3);

        // Optional: convert to float2 for readable printing
        float2 fq0 = __bfloat1622float2(all_q[0]);
        float2 fq1 = __bfloat1622float2(all_q[1]);
        float2 fq2 = __bfloat1622float2(all_q[2]);
        float2 fq3 = __bfloat1622float2(all_q[3]);

        printf("tid, %d, land_id, %d, cur_q_reg bf16 values, %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f\n",
               tid, lane_id, __bfloat162float(all_q[0].x), __bfloat162float(all_q[0].y), __bfloat162float(all_q[1].x),
               __bfloat162float(all_q[1].y), __bfloat162float(all_q[2].x), __bfloat162float(all_q[2].y),
               __bfloat162float(all_q[3].x), __bfloat162float(all_q[3].y));
      }
    }
  // we need a syncthreads() here so that we don't load K global->shared
  // before finishing loading Q shared->reg
  __syncthreads();
  int num_kv_blocks = len_kv / BLOCK_KV;
  int num_n_per_kv_block = BLOCK_KV /  MMA_N;
  int num_k_per_dim = DIM / MMA_K;
  
  if(isthread0()){
    printf("num_kv_blocks, %d, num_n_per_kv_block, %d, num_k_per_dim, %d \n", num_kv_blocks, num_n_per_kv_block,
           num_k_per_dim);
  }
  
  for (int off_kv = 0; off_kv < len_kv; off_kv += BLOCK_KV) {
    if (isthread0()){
        printf("Processing K/V block offset: %d \n", off_kv);
    }
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // load K [BLOCK_KV, DIM]
    global_to_shared<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    if(isthread0() && off_kv == 0){
        // printf("S_rmem size is ")
        // Debug: print first 8 elements of Q_smem
        printf("K_smem first 8 elements:\n");
        for(int k_row=0; k_row < BLOCK_KV; k_row+=1){
        printf("Row %d: ", k_row);
        for(int i=0; i < 32; i++){
            nv_bfloat16 val = smem[k_row * DIM + i];
            float fval = __bfloat162float(val); 
            // print with 2 decimal places
            printf("%.1f ", fval);
        }
        printf("\n");
        }
    }
    __syncthreads();

    // shared -> registers
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
        const int row = mma_id_kv * MMA_N + (lane_id % 8);
        const int col = mma_id_d * MMA_K + (lane_id / 8 * 8);
        const uint32_t addr = K_smem + (row * DIM + col) * sizeof(nv_bfloat16);
        ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
        if (mma_id_kv == 0 && mma_id_d == 0 && off_kv == 0) {
          auto cur_k_reg = K_rmem[mma_id_kv][mma_id_d];
          uint32_t _k0 = cur_k_reg[0];
          uint32_t _k1 = cur_k_reg[1];
          uint32_t _k2 = cur_k_reg[2];
          uint32_t _k3 = cur_k_reg[3];
          nv_bfloat162 all_k[2];
          all_k[0] = *reinterpret_cast<nv_bfloat162 *>(&_k0);
          all_k[1] = *reinterpret_cast<nv_bfloat162 *>(&_k1);

          // Optional: convert to float2 for readable printing
          float2 fq0 = __bfloat1622float2(all_k[0]);
          float2 fq1 = __bfloat1622float2(all_k[1]);

          printf("tid, %d, land_id, %d, all_k bf16 values, %0.2f %0.2f | %0.2f %0.2f \n", tid, lane_id,
                 __bfloat162float(all_k[0].x), __bfloat162float(all_k[0].y), __bfloat162float(all_k[1].x),
                 __bfloat162float(all_k[1].y));
        }
      }
  

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
          mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d], S_rmem[mma_id_q][mma_id_kv]);
          if (mma_id_q == 0 && mma_id_kv == 0 && mma_id_d == 0 && off_kv == 0) {
            nv_bfloat162 cur_q[4];
            nv_bfloat162 cur_k[2];
            cur_q[0] = *reinterpret_cast<nv_bfloat162 *>(&Q_rmem[mma_id_q][mma_id_d][0]);
            cur_q[1] = *reinterpret_cast<nv_bfloat162 *>(&Q_rmem[mma_id_q][mma_id_d][1]);
            cur_q[2] = *reinterpret_cast<nv_bfloat162 *>(&Q_rmem[mma_id_q][mma_id_d][2]);
            cur_q[3] = *reinterpret_cast<nv_bfloat162 *>(&Q_rmem[mma_id_q][mma_id_d][3]);
            cur_k[0] = *reinterpret_cast<nv_bfloat162 *>(&K_rmem[mma_id_kv][mma_id_d][0]);
            cur_k[1] = *reinterpret_cast<nv_bfloat162 *>(&K_rmem[mma_id_kv][mma_id_d][1]);
            float2 fq0 = __bfloat1622float2(cur_q[0]);
            float2 fq1 = __bfloat1622float2(cur_q[1]);
            float2 fq2 = __bfloat1622float2(cur_q[2]);
            float2 fq3 = __bfloat1622float2(cur_q[3]);
            float2 fk0 = __bfloat1622float2(cur_k[0]);
            float2 fk1 = __bfloat1622float2(cur_k[1]);
            printf("tid, %d, land_id, %d, cur_q bf16 values, %0.4f %0.4f | %0.4f %0.4f | %0.4f %0.4f | %0.4f %0.4f\n",
                   tid, lane_id, __bfloat162float(cur_q[0].x), __bfloat162float(cur_q[0].y),
                   __bfloat162float(cur_q[1].x), __bfloat162float(cur_q[1].y), __bfloat162float(cur_q[2].x),
                   __bfloat162float(cur_q[2].y), __bfloat162float(cur_q[3].x), __bfloat162float(cur_q[3].y));
            printf("tid, %d, land_id, %d, cur_k bf16 values, %0.4f %0.4f | %0.4f %0.4f\n", tid, lane_id, 
                   __bfloat162float(cur_k[0].x), __bfloat162float(cur_k[0].y), __bfloat162float(cur_k[1].x),
                   __bfloat162float(cur_k[1].y));
            
            
            
            auto res = S_rmem[mma_id_q][mma_id_kv];
            float all_res[4];
            all_res[0] = *reinterpret_cast<float *>(&res[0]);
            all_res[1] = *reinterpret_cast<float *>(&res[1]);
            all_res[2] = *reinterpret_cast<float *>(&res[2]);
            all_res[3] = *reinterpret_cast<float *>(&res[3]);
            printf("tid, %d, land_id, %d, all_k res values, %0.4f %0.4f %0.4f %0.4f \n", tid, lane_id, all_res[0],
                   all_res[1], all_res[2], all_res[3]);
          }
        }
      }
    }
    __syncthreads();

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // apply softmax scale
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int reg_id = 0; reg_id < 4; reg_id++)
          S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

      // rowmax
      float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
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
      if(off_kv == 32 && mma_id_q ==0){
        printf("tid: %d, mma_id_q: %d, before first K block, rowmax: %f, %f, this_rowmax: %f, %f \n", tid, mma_id_q,
               rowmax[mma_id_q][0], rowmax[mma_id_q][1], this_rowmax[0], this_rowmax[1]);
      }
      
      rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      // save new rowmax
      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      // rowsumexp
      float this_rowsumexp[2] = {};
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);  // c0
        regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);  // c1
        regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);  // c2
        regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);  // c3

        this_rowsumexp[0] += regs[0] + regs[1];
        this_rowsumexp[1] += regs[2] + regs[3];

        // pack to P registers for next MMA
        // we need to change from m16n8 to m16k16
        nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
        this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
        this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
      }

      // butterfly reduction within 4 threads
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // accumulate to total rowsumexp
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
      
      if (off_kv == 32 and mma_id_q == 0){
        printf("tid: %d, mma_id_q: %d, rowsumexp: %f, %f | this_rowsumexp: %f, %f \n", tid, mma_id_q,
               rowsumexp[mma_id_q][0], rowsumexp[mma_id_q][1], this_rowsumexp[0], this_rowsumexp[1]);
        // rescale
        printf("tid: %d, mma_id_q: %d, rescale: %f, %f \n", tid, mma_id_q,
               rescale[0], rescale[1]);
      }
      
    }
    
    // show tile p for debug
    // if ()
    
    // if(off_kv == 0 ){
    //     // row max
    //     for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
    //         printf("tid: %d, After first K block, rowmax[%d]: %f, %f \n", tid, mma_id_q, rowmax[mma_id_q][0], rowmax[mma_id_q][1]);
    //     }
        
    // }
    // print rowsumexp
    if (off_kv == 32){
        for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
            printf("tid: %d, After last K block, rowsumexp[%d]: %f, %f | \n", tid, mma_id_q, rowsumexp[mma_id_q][0], rowsumexp[mma_id_q][1]);
        }
    }
    // load V [BLOCK_KV, DIM]
    // NOTE: we can schedule to load V global->shared earlier
    global_to_shared<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid, true);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // shared -> registers
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        const int row = mma_id_kv * MMA_K + (lane_id % 16);
        const int col = mma_id_d * MMA_N + (lane_id / 16) * 8;
        const uint32_t addr = V_smem + (row * DIM + col) * sizeof(nv_bfloat16);
        ldmatrix_x2_trans(V_rmem[mma_id_kv][mma_id_d], addr);
        if (off_kv == 0 && mma_id_d == 0 && mma_id_kv == 0){
            auto cur_v_reg = V_rmem[mma_id_kv][mma_id_d];
            uint32_t _v0 = cur_v_reg[0];
            uint32_t _v1 = cur_v_reg[1];
            nv_bfloat162 all_v[2];
            all_v[0] = *reinterpret_cast<nv_bfloat162 *>(&_v0);
            all_v[1] = *reinterpret_cast<nv_bfloat162 *>(&_v1);
    
            // Optional: convert to float2 for readable printing
            float2 fv0 = __bfloat1622float2(all_v[0]);
            float2 fv1 = __bfloat1622float2(all_v[1]);
    
            printf("tid, %d, land_id, %d, all_v bf16 values, %0.2f %0.2f | %0.2f %0.2f \n", tid, lane_id,
                     __bfloat162float(all_v[0].x), __bfloat162float(all_v[0].y), __bfloat162float(all_v[1].x),
                     __bfloat162float(all_v[1].y));
        }
      }
    __syncthreads();
    // MMA O += P @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {

          mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d], O_rmem[mma_id_q][mma_id_d]);
          if (off_kv == BLOCK_KV && mma_id_q == 0 && mma_id_d == 0 && mma_id_kv == 0) {
            auto res = O_rmem[mma_id_q][mma_id_d];
            // float all_res[4];
            // all_res[0] = res[0];
            printf("tid, %d, land_id, %d, O_rmem bf16 values after first MMA, %0.4f %0.4f %0.4f %0.4f \n", tid, lane_id,
                   res[0], res[1], res[2], res[3]);
            // show p
            auto p_res = P_rmem[mma_id_q][mma_id_kv];
            nv_bfloat162 all_p[4];
            all_p[0] = *reinterpret_cast<nv_bfloat162 *>(&p_res[0]);
            all_p[1] = *reinterpret_cast<nv_bfloat162 *>(&p_res[1]);
            all_p[2] = *reinterpret_cast<nv_bfloat162 *>(&p_res[2]);
            all_p[3] = *reinterpret_cast<nv_bfloat162 *>(&p_res[3]);
            float2 fp0 = __bfloat1622float2(all_p[0]);
            float2 fp1 = __bfloat1622float2(all_p[1]);
            float2 fp2 = __bfloat1622float2(all_p[2]);
            float2 fp3 = __bfloat1622float2(all_p[3]);
            printf("tid, %d, land_id, %d, P_rmem bf16 values after first MMA, %0.4f %0.4f | %0.4f %0.4f | %0.4f %0.4f "
                   "| %0.4f %0.4f \n",
                   tid, lane_id, fp0.x, fp0.y, fp1.x, fp1.y, fp2.x, fp2.y, fp3.x, fp3.y);

            // show v
            auto v_res = V_rmem[mma_id_kv][mma_id_d];
            nv_bfloat162 all_v[2];
            all_v[0] = *reinterpret_cast<nv_bfloat162 *>(&v_res[0]);
            all_v[1] = *reinterpret_cast<nv_bfloat162 *>(&v_res[1]);
            float2 fv0 = __bfloat1622float2(all_v[0]);
            float2 fv1 = __bfloat1622float2(all_v[1]);
            printf("tid, %d, land_id, %d, V_rmem bf16 values after first MMA, %0.4f %0.4f | %0.4f %0.4f \n", tid,
                   lane_id, __bfloat162float(all_v[0].x), __bfloat162float(all_v[0].y), __bfloat162float(all_v[1].x),
                   __bfloat162float(all_v[1].y));
          }
        }
      }
    }
    K += BLOCK_KV * DIM;
    V += BLOCK_KV * DIM;
  }

  // write to O
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      // divide by softmax denominator
      float *regs = O_rmem[mma_id_q][mma_id_d];
      if( mma_id_q == 0 && mma_id_d == 0){
        printf("before: %d, After final denom, O_rmem[%d][%d]: %f, %f, %f, %f \n", tid, mma_id_q, mma_id_d,
               regs[0], regs[1], regs[2], regs[3]);
        //rowsumexp
        printf("tid: %d, Final rowsumexp[%d]: %f, %f \n", tid, mma_id_q,
               rowsumexp[mma_id_q][0], rowsumexp[mma_id_q][1]);
      }
      regs[0] /= rowsumexp[mma_id_q][0];
      regs[1] /= rowsumexp[mma_id_q][0];
      regs[2] /= rowsumexp[mma_id_q][1];
      regs[3] /= rowsumexp[mma_id_q][1];

      reinterpret_cast<nv_bfloat162 *>(O + (row + 0) * DIM + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * DIM + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
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

  if (dim != 64) {
    std::cerr << "Unsupported dim=" << dim << std::endl;
    exit(1);
  }

  const int BLOCK_Q = 64;
  const int BLOCK_KV = 32;
  const int DIM = 64;
  const int NUM_WARPS = 4;

  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int smem_size = max(BLOCK_Q, BLOCK_KV * 2) * DIM * sizeof(nv_bfloat16);

  auto kernel = attention_v1_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;
  launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, Q, K, V, O, bs, len_q, len_kv);
}



// template <typename T, typename... Args>
// void launch_kernel(
//   T *kernel,
//   int num_blocks,
//   int block_size,
//   int smem_size,
//   Args... args) {
//   if (smem_size > 48'000)
//     CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
//     printf("Launch kenrel with num_blocks: %d, block_size: %d \n", num_blocks, block_size);
//   kernel<<<num_blocks, block_size, smem_size>>>(args...);
//   CUDA_CHECK(cudaGetLastError());
// }

