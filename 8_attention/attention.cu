#include <cuda_bf16.h>
#include <cstdint>
#include <float.h>
#include <iostream>

#define CUDA_CHECK(x)                                                                                                  \
  {                                                                                                                    \
    auto error = x;                                                                                                    \
    if (error != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }

constexpr int WARP_SIZE = 32;
using KernelFn = void(*)(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,  // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv);

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

__device__
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__global__
void attention_kernel(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,  // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv) {

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // each threadblock handles 1 sequence
  // TODO: split along len_q as well
  Q += bid * len_q * DIM;
  K += bid * len_kv * DIM;
  V += bid * len_kv * DIM;
  O += bid * len_q * DIM;

  extern __shared__ nv_bfloat16 shm[];
  const uint32_t Q_shm = __cvta_generic_to_shared(shm);
  const uint32_t K_shm = Q_shm + BLOCK_Q * DIM * sizeof(nv_bfloat16);
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
  float QK_regs[WARP_Q / MMA_M][BLOCK_KV / MMA_N][num_acc_regs];

  // let compiler decide register reuse?
  uint32_t P_regs[WARP_Q / MMA_M][BLOCK_KV / MMA_K][num_A_regs];
  uint32_t V_regs[BLOCK_KV / MMA_K][DIM / MMA_N][num_B_regs];

  // we use the same registers for O_regs and PV_regs
  // rescale O_regs once we obtain new rowmax, then accumulate to O_regs
  float O_regs[WARP_Q / MMA_M][DIM / MMA_N][num_acc_regs];

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2];

  // const float softmax_scale = __frsqrt_rn(static_cast<float>(DIM));
  const float softmax_scale = 1.0f;

  for (int off_q = 0; off_q < len_q; off_q += BLOCK_Q) {
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      rowmax[mma_id_q][0] = -FLT_MAX;
      rowmax[mma_id_q][1] = -FLT_MAX;
      rowsumexp[mma_id_q][0] = 0.0f;
      rowsumexp[mma_id_q][1] = 0.0f;
    }

    // clear O accumulator
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int reg_id = 0; reg_id < num_acc_regs; reg_id++)
          O_regs[mma_id_q][mma_id_d][reg_id] = 0.0f;

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

    for (int off_kv = 0; off_kv < len_kv; off_kv += BLOCK_KV) {
      // clear QK accumulator
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
          for (int reg_id = 0; reg_id < num_acc_regs; reg_id++)
            QK_regs[mma_id_q][mma_id_kv][reg_id] = 0.0f;

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
        // rowmax
        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
          float *regs = QK_regs[mma_id_q][mma_id_kv];

          // apply softmax scale
          for (int reg_id = 0; reg_id < num_acc_regs; reg_id++)
            regs[reg_id] *= softmax_scale;

          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));  // c0 and c1
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));  // c2 and c3
        }

        // butterfly reduction within 4 threads
        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 0x1));
        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 0x10));
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 0x1));
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 0x10));

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
          this_P_regs[(mma_id_kv % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
          this_P_regs[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
        }

        // butterfly reduction within 4 threads
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 0x1);
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 0x10);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 0x1);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 0x10);

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

      __syncthreads();
      K += BLOCK_KV * DIM;
      V += BLOCK_KV * DIM;
    }
    K -= len_kv * DIM;
    V -= len_kv * DIM;

    // write to O
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
        nv_bfloat16 *O_ptr = O + row * DIM + col;

        // divide by softmax denominator
        float *this_O_regs = O_regs[mma_id_q][mma_id_d];
        this_O_regs[0] /= rowsumexp[mma_id_q][0];
        this_O_regs[1] /= rowsumexp[mma_id_q][0];
        this_O_regs[2] /= rowsumexp[mma_id_q][1];
        this_O_regs[3] /= rowsumexp[mma_id_q][1];

        reinterpret_cast<nv_bfloat162 *>(O_ptr)[0] = __float22bfloat162_rn({this_O_regs[0], this_O_regs[1]});
        reinterpret_cast<nv_bfloat162 *>(O_ptr + 8 * DIM)[0] = __float22bfloat162_rn({this_O_regs[2], this_O_regs[3]});
      }

    __syncthreads();
    Q += BLOCK_Q * DIM;
    O += BLOCK_Q * DIM;
  }
}

template<int DIM>
void attention_dispatch(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,  // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv) {

  const int BLOCK_Q = 128;
  const int BLOCK_KV = 64;
  const int NUM_WARPS = 4;

  const int num_tbs = bs;
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int shm_size = (BLOCK_Q + BLOCK_KV * 2) * DIM * sizeof(nv_bfloat16);

  KernelFn kernel = attention_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;

  if (shm_size > 48'000) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
  }
  kernel<<<num_tbs, TB_SIZE, shm_size>>>(Q, K, V, O, bs, len_q, len_kv);
  CUDA_CHECK(cudaGetLastError());
}

void attention(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,  // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv,
  int dim) {

  if (dim == 128)
    attention_dispatch<128>(Q, K, V, O, bs, len_q, len_kv);
}
