#include "common.h"
#include <cuda_bf16.h>
#include <iostream>
#include <float.h>
// Impl a fp32 version?

/*
# Share memory: Q/K/V
# Reg mem:
#  query_rmem warp bf16 [query_rows_per_warp // MMAConfig.M][DIM      // MMAConfig.K][MMAConfig.M, MMAConfig.K]
#  key_rmem   tb   bf16 [BLOCK_KV            // MMAConfig.N][DIM      // MMAConfig.K][MMAConfig.N, MMAConfig.K]
#  s_rmem     warp fp32 [query_rows_per_warp // MMAConfig.M][BLOCK_KV // MMAConfig.N][MMAConfig.M, MMAConfig.N]
#  prob_rmem  warp bf16 [query_rows_per_warp // MMAConfig.M][BLOCK_KV // MMAConfig.K][MMAConfig.M, MMAConfig.K]
#  value_rmem tb   bf16 [BLOCK_KV            // MMAConfig.K][DIM      // MMAConfig.N][MMAConfig.K, MMAConfig.N]
#  out_rmem   warp fp32 [query_rows_per_warp // MMAConfig.M][DIM      // MMAConfig.N][MMAConfig.M, MMAConfig.N]
#  rowmax     warp fp32 [BLOCK_Q]
#  rowsumexp  warp fp32 [BLOCK_Q]

*/

// Load val from global mem to shared mem

template <int TB_SIZE, int smem_rows, int smem_cols>
__device__ void load_global_to_shared_mem(const nv_bfloat16 *src, uint32_t dst, int ld_src, int tid) {
  // assume src has moved to the correct block and current load position
  // dst is a block shared mem, [smem_rows][smem_cols]
  // load [:smem_rows, :smem_cols] from src to dst
  const int total_elems = smem_rows * smem_cols;
  int elems_per_thread_per_instruction = 16 / sizeof(nv_bfloat16); // 16 bytes per cp.async
  // TODO:check total_elems is divisible by TB_SIZE
  int total_elems_per_iter = elems_per_thread_per_instruction * TB_SIZE;
  int num_iters = cdiv(total_elems, total_elems_per_iter);
  for (int iter_id = 0; iter_id < num_iters; ++iter_id) {
    // if (isthread0()) {
    //   printf("Strat iter %d \n", iter_id);
    // }
    int cur_iter_base_id = iter_id * total_elems_per_iter;
    int cur_thread_base_id = cur_iter_base_id + tid * elems_per_thread_per_instruction;
    int cur_thread_row = cur_thread_base_id / smem_cols;
    int cur_thread_col = cur_thread_base_id % smem_cols;
    // printf("tid: %d, iter_id: %d, cur_thread_base_id: %d, cur_thread_row: %d, cur_thread_col: %d\n",
    //       tid, iter_id, cur_thread_base_id, cur_thread_row,  cur_thread_col);
    // dst postion
    const uint32_t dst_addr = dst + (cur_thread_row * smem_cols + cur_thread_col) * sizeof(nv_bfloat16);
    // srt postion
    const nv_bfloat16 *src_addr = src + (cur_thread_row * ld_src + cur_thread_col);
    // cp-size : 16 Bytes per instruction
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
    // cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size} [dst], [src], cp-size{, src-size}{, cache-policy} ;
    // cp.async.cg.shared.global.L2::128B 
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_addr), "l"(src_addr));
  }
}

// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
__device__ void load_matrix_x4(uint32_t dst_regs[4], uint32_t src_addr) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(dst_regs[0]), "=r"(dst_regs[1]), "=r"(dst_regs[2]), "=r"(dst_regs[3])
               : "r"(src_addr));
}
__device__ void load_matrix_x2(uint32_t dst_regs[2], uint32_t src_addr) {
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
               : "=r"(dst_regs[0]), "=r"(dst_regs[1])
               : "r"(src_addr));
}
__device__ void mma_m16n8k16_instruction(uint32_t A[4], uint32_t B[2], float C[4], float D[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS *WARP_SIZE)
__global__ void attention_v6_kernel(
        const nv_bfloat16 *Q,
        const nv_bfloat16 *K,
        const nv_bfloat16 *V,
        nv_bfloat16 *O,
        const int bs,
        const int len_q,
        const int len_kv
    ) {
  // impl
  // ** Load query from global mem to shared mem
  __shared__ nv_bfloat16 query_smem[BLOCK_Q * DIM];
  // convert query_smem to uint32_t ptr
  int tid = threadIdx.x;
  const uint32_t query_smem_ptr = __cvta_generic_to_shared(query_smem);
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  load_global_to_shared_mem<TB_SIZE, BLOCK_Q, DIM>(Q, query_smem_ptr, DIM, tid);

  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();
  if (isthread0()) {
    // debug print
    for (int i = 0; i < min(10, BLOCK_Q); ++i) {
      printf("query_smem[%d]: ", i);
      for (int j = 0; j < min(48, DIM); ++j) {
        printf("%0.1f, ", __bfloat162float(query_smem[i * DIM + j]));
      }
      printf("\n");
    }
  }

  // Let check how many mmas are required for the Q@K
  // m16n8k16 config
  constexpr int MMA_M = 16;
  constexpr int MMA_K = 16;
  constexpr int MMA_N = 8;

  constexpr int num_query_rows_per_tb = BLOCK_Q;
  constexpr int num_query_rows_per_warp = cdiv(num_query_rows_per_tb, NUM_WARPS);
  // query serve as A matrix, A_frag includes 8 bf16s, and save in 4 unit32 regs
  constexpr int first_mma_num_m = num_query_rows_per_warp / MMA_M;
  constexpr int first_mma_num_k = DIM / MMA_K;
  constexpr int first_mma_num_n = BLOCK_KV / MMA_M;
  uint32_t query_regs[first_mma_num_m][first_mma_num_k][4]; // each 4 unit32 regs store 8 bf16s
  // key server as B matrix, B_frag include 4 bf16s, and save in 2 unit32 regs
  uint32_t key_regs[first_mma_num_n][first_mma_num_k][2]; // each 2 unit32 regs store 4 bf16s
  // ** Load query from share mem to register
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  for (int mma_m_id = 0; mma_m_id < first_mma_num_m; ++mma_m_id) {
    for (int mma_k_id = 0; mma_k_id < first_mma_num_k; ++mma_k_id) {
      auto cur_a_frag_ptr = &query_regs[mma_m_id][mma_k_id][0];
      // the start point of current 16x16 sub-matrix
      auto cur_mma_query_regs = query_regs[mma_m_id][mma_k_id];
      int cur_lane_row_in_a_frag = lane_id % 16;
      int cur_lane_col_in_a_frag = lane_id / 16 * 8;
      int cur_lane_row = warp_id * num_query_rows_per_warp + mma_m_id * MMA_M + cur_lane_row_in_a_frag;
      int cur_lane_col = mma_k_id * MMA_K + cur_lane_col_in_a_frag;
      const uint32_t cur_lane_query_smem_addr = query_smem_ptr +
            (cur_lane_row * DIM + cur_lane_col) * sizeof(nv_bfloat16);
      load_matrix_x4(cur_mma_query_regs, cur_lane_query_smem_addr);
      __syncthreads();
      // show the first mma a frag
      if(mma_m_id == 0 && mma_k_id == 0) {
        // printf("query_regs[%d][%d]: ", mma_m_id, mma_k_id);
        auto cur_mma_query_regs = query_regs[mma_m_id][mma_k_id];
        nv_bfloat162 temp_query_regs[4];
        temp_query_regs[0] = *reinterpret_cast<nv_bfloat162 *>(&cur_mma_query_regs[0]);
        temp_query_regs[1] = *reinterpret_cast<nv_bfloat162 *>(&cur_mma_query_regs[1]);
        temp_query_regs[2] = *reinterpret_cast<nv_bfloat162 *>(&cur_mma_query_regs[2]);
        temp_query_regs[3] = *reinterpret_cast<nv_bfloat162 *>(&cur_mma_query_regs[3]);
        float2 f_temp[4];
        f_temp[0] = __bfloat1622float2(temp_query_regs[0]);
        f_temp[1] = __bfloat1622float2(temp_query_regs[1]);
        f_temp[2] = __bfloat1622float2(temp_query_regs[2]);
        f_temp[3] = __bfloat1622float2(temp_query_regs[3]);
        printf("tid: %d, query_regs: %0.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f\n", tid, f_temp[0].x, f_temp[0].y,
               f_temp[1].x, f_temp[1].y, f_temp[2].x, f_temp[2].y, f_temp[3].x, f_temp[3].y);
      }
    }
  } // end of load query to register
  
  // ** start iterate over key/value blocks
  int num_kv_iters = cdiv(len_kv, BLOCK_KV);
  for(int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++){
    // ** init Q@K.T output reg: s_reg, 4 fp32s, as zeros
    float s_regs[first_mma_num_m][first_mma_num_n][4] = {};
    // ** Load key from global mem to shared mem
    // currently, query were loaded to register, so we reuse the shared mem of query for key
    // key size: [BLOCK_KV, DIM]
    nv_bfloat16 *key_smem = query_smem;
    const uint32_t key_smem_ptr = __cvta_generic_to_shared(key_smem);
    load_global_to_shared_mem<TB_SIZE,BLOCK_KV, DIM>(K, key_smem_ptr, DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    
    if (isthread0()) {
        // debug print
        for (int i = 0; i < min(10, BLOCK_Q); ++i) {
        printf("key_smem[%d]: ", i);
        for (int j = 0; j < min(48, DIM); ++j) {
            printf("%0.1f, ", __bfloat162float(key_smem[i * DIM + j]));
        }
        printf("\n");
        }
    }
    
    // ** Load key from shared mem to register
    for(int mma_n_id = 0; mma_n_id < first_mma_num_n; mma_n_id++){
        for(int mma_k_id = 0; mma_k_id < first_mma_num_k; mma_k_id++){
            auto cur_mma_key_frag = key_regs[mma_n_id][mma_k_id];
            int cur_mma_b_frag_row = mma_n_id * MMA_N;
            int cur_mma_b_frag_col = mma_k_id * MMA_K;
            // why 8? 8x16? yes
            int cur_lane_b_frag_row_in_b_frag = lane_id % 8;
            int cur_lane_b_frag_col_in_b_frag = lane_id / 8 *8;
            int cur_lane_row = cur_mma_b_frag_row + cur_lane_b_frag_row_in_b_frag;
            int cur_lane_col = cur_mma_b_frag_col + cur_lane_b_frag_col_in_b_frag;
            auto cur_lane_key_smem_addr = key_smem_ptr + (cur_lane_row * DIM + cur_lane_col) * sizeof(nv_bfloat16);
            load_matrix_x2(key_regs[mma_n_id][mma_k_id], cur_lane_key_smem_addr);

          // show the first mma b frag
            if (mma_n_id == 0 && mma_k_id == 0) {
              auto cur_mma_key_regs = key_regs[mma_n_id][mma_k_id];
              nv_bfloat162 temp_key_regs[2];
              temp_key_regs[0] = *reinterpret_cast<nv_bfloat162 *>(&cur_mma_key_regs[0]);
              temp_key_regs[1] = *reinterpret_cast<nv_bfloat162 *>(&cur_mma_key_regs[1]);
              float2 f_temp[4];
              f_temp[0] = __bfloat1622float2(temp_key_regs[0]);
              f_temp[1] = __bfloat1622float2(temp_key_regs[1]);
              printf("tid: %d, key_regs: %0.1f, %.1f, %.1f, %.1f\n", tid, f_temp[0].x, f_temp[0].y, f_temp[1].x,
                     f_temp[1].y);
            }
        }
    } // end of load key to register
    
    // ** query and key are both in register, start compute Q@K.T
    for(int mma_m_id = 0; mma_m_id < first_mma_num_m; mma_m_id++){
        for(int mma_n_id = 0; mma_n_id < first_mma_num_n; mma_n_id++){
            for(int mma_k_id = 0; mma_k_id < first_mma_num_k; mma_k_id++){
                // 
                auto cur_a_frag = query_regs[mma_m_id][mma_k_id];
                auto cur_b_frag = key_regs[mma_n_id][mma_k_id];
                auto cur_d_frag = s_regs[mma_m_id][mma_n_id];
                mma_m16n8k16(query_regs[mma_m_id][mma_k_id],
                             key_regs[mma_n_id][mma_k_id],
                             s_regs[mma_m_id][mma_n_id]);
                // mma_m16n8k16(cur_a_frag, cur_b_frag, cur_d_frag);
                // mma_m16n8k16_instruction(cur_a_frag, cur_b_frag, cur_d_frag, cur_d_frag);
                // show the result of first mma
                if(mma_m_id == 0 && mma_n_id == 0 && mma_k_id == 0 && kv_iter == 0){
                    nv_bfloat162 temp_query_regs[4];
                    temp_query_regs[0] = *reinterpret_cast<nv_bfloat162 *>(&cur_a_frag[0]);
                    temp_query_regs[1] = *reinterpret_cast<nv_bfloat162 *>(&cur_a_frag[1]);
                    temp_query_regs[2] = *reinterpret_cast<nv_bfloat162 *>(&cur_a_frag[2]);
                    temp_query_regs[3] = *reinterpret_cast<nv_bfloat162 *>(&cur_a_frag[3]);
                    float2 f_temp[4];
                    f_temp[0] = __bfloat1622float2(temp_query_regs[0]);
                    f_temp[1] = __bfloat1622float2(temp_query_regs[1]);
                    f_temp[2] = __bfloat1622float2(temp_query_regs[2]);
                    f_temp[3] = __bfloat1622float2(temp_query_regs[3]);
                    printf("tid: %d, query_regs: %0.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f\n", tid, f_temp[0].x, f_temp[0].y,
                           f_temp[1].x, f_temp[1].y, f_temp[2].x, f_temp[2].y, f_temp[3].x, f_temp[3].y);
                    nv_bfloat162 temp_key_regs[2];
                    temp_key_regs[0] = *reinterpret_cast<nv_bfloat162 *>(&cur_b_frag[0]);
                    temp_key_regs[1] = *reinterpret_cast<nv_bfloat162 *>(&cur_b_frag[1]);
                    float2 f_key_temp[4];
                    f_key_temp[0] = __bfloat1622float2(temp_key_regs[0]);
                    f_key_temp[1] = __bfloat1622float2(temp_key_regs[1]);
                    printf("tid: %d, key_regs: %0.1f, %.1f, %.1f, %.1f\n", tid, f_key_temp[0].x, f_key_temp[0].y,
                           f_key_temp[1].x, f_key_temp[1].y);
                    
                    printf("tid: %d, s_regs: %0.1f, %0.1f, %0.1f, %0.1f\n", tid,
                    cur_d_frag[0],
                    cur_d_frag[1],
                    cur_d_frag[2],
                    cur_d_frag[3]);
                    
                        // s_regs[mma_m_id][mma_n_id][0],
                        // s_regs[mma_m_id][mma_n_id][1],
                        // s_regs[mma_m_id][mma_n_id][2],
                        // s_regs[mma_m_id][mma_n_id][3]);
                }
    }}} // end of Q@K compute
    
    // __syncthreads();
    
    // if (isthread0() && kv_iter == 0) {
    //     // debug print
    //     for (int i = 0; i < min(10, BLOCK_Q); ++i) {
    //     printf("key_smem[%d]: ", i);
    //     for (int j = 0; j < min(48, DIM); ++j) {
    //         printf("%0.1f, ", __bfloat162float(key_smem[i * DIM + j]));
    //     }
    //     printf("\n");
    //     }
    // }
    
  } // end of kv block iter


}

void attention_v6(const nv_bfloat16 *Q, // [bs, len_q, DIM]
                  const nv_bfloat16 *K, // [bs, len_kv, DIM]
                  const nv_bfloat16 *V, // [bs, len_kv, DIM]
                  nv_bfloat16 *O,       // [bs, len_q, DIM]
                  int bs, int len_q, int len_kv, int dim) {

  if (dim != 128 && dim != 64) {
    std::cerr << "Unsupported dim=" << dim << std::endl;
    exit(1);
  }

  const int BLOCK_Q = 128;
  const int BLOCK_KV = 64;
  const int DIM = 64;
  const int NUM_WARPS = 4;

  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;
//   const int smem_size = max(BLOCK_Q, BLOCK_KV * 2) * DIM * sizeof(nv_bfloat16);

  //   auto kernel = attention_v1_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;

  dim3 grid(num_blocks);
  dim3 block(TB_SIZE);
  attention_v6_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS><<<grid, block>>>(Q, K, V, O, bs, len_q, len_kv);

  //   launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, Q, K, V, O, bs, len_q, len_kv);
}
