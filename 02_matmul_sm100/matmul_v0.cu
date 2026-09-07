#include "common.h"

#include <cuda_bf16.h>

// STRIDE in bytes, col in the units of 16-byte
template <int STRIDE>
__device__ static
int swizzle(int row, int col) {
  if constexpr (STRIDE >= 128)
    col ^= (row % 8) / std::max(128 / STRIDE, 1);
  return row * STRIDE + col * 16;
}

__device__
void ldmatrix_x4(int reg[4], int addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
              : "r"(addr));
}

__device__
void mma_m16n8k16(const int A[4], const int B[2], float C[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, " // D
               "{%4, %5, %6, %7}, " // A
               "{%8, %9}, "         // B
               "{%0, %1, %2, %3};"  // C
              : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]));
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ static
void gmem_to_smem(int dst, const T *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(T);
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    // NOTE: perhaps we can move swizzle out of this loop as well
    int dst_addr = dst + swizzle<WIDTH * sizeof(T)>(row, col / num_elems);
    const T *src_addr = src + row * src_stride + col;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v0_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // TODO: threadblock swizzling to improve L2 cache hit rate
  const int num_blocks_n = cdiv(N, BLOCK_N);
  const int bid_m = bid / num_blocks_n;
  const int bid_n = bid % num_blocks_n;
  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  // A is row-major, B is column-major, C is row-major
  A += off_m * K;
  B += off_n * K;
  C += (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

  // convert shared memory address to 32-bit from the start
  extern __shared__ nv_bfloat16 smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  const int A_smem = smem;
  const int B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int AB_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);

  int A_rmem[BLOCK_K / MMA_K][WARP_M / MMA_M][4];
  int B_rmem[BLOCK_K / MMA_K][WARP_N / MMA_N][2];
  float acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_smem_thr = A_smem + swizzle<BLOCK_K * sizeof(nv_bfloat16)>((warp_id_m * WARP_M) + (lane_id % 16), lane_id / 16);
  const int B_smem_thr = B_smem + swizzle<BLOCK_K * sizeof(nv_bfloat16)>((warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8, (lane_id % 16) / 8);

  const int num_k_iters = cdiv(K, BLOCK_K);

  auto load_AB = [&](int k_iter) {
    // select the correct shared memory buffer
    const int stage_id = k_iter % NUM_STAGES;
    gmem_to_smem<BLOCK_M, BLOCK_K, TB_SIZE>(A_smem + stage_id * AB_size, A, K, tid);
    gmem_to_smem<BLOCK_N, BLOCK_K, TB_SIZE>(B_smem + stage_id * AB_size, B, K, tid);

    // A/B pointer tracks position for gmem->smem load
    A += BLOCK_K;
    B += BLOCK_K;
    asm volatile("cp.async.commit_group;");
  };

  auto compute = [&](int k_iter) {
    const int stage_id = k_iter % NUM_STAGES;

    // A shared->regs
    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int m = 0; m < WARP_M / MMA_M; m++) {
        int A_addr = A_smem_thr + stage_id * AB_size + (m * MMA_M * BLOCK_K * sizeof(nv_bfloat16));
        ldmatrix_x4(A_rmem[k][m], A_addr ^ (k * 32));
      }

    // B shared->regs
    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int n = 0; n < WARP_N / MMA_N; n += 2) {
        int B_addr = B_smem_thr + stage_id * AB_size + (n * MMA_N * BLOCK_K * sizeof(nv_bfloat16));
        ldmatrix_x4(B_rmem[k][n], B_addr ^ (k * 32));
      }

    // do MMA
    for (int k = 0; k < BLOCK_K / MMA_K; k++)
      for (int m = 0; m < WARP_M / MMA_M; m++)
        for (int n = 0; n < WARP_N / MMA_N; n++)
          mma_m16n8k16(A_rmem[k][m], B_rmem[k][n], acc[m][n]);
  };

  // initiate NUM_STAGES-1 stages
  for (int stage = 0; stage < NUM_STAGES - 1; stage++)
    load_AB(stage);

  // loop invariance: there is always NUM_STAGES - 1 prefetch stages in-flight
  for (int k_iter = 0; k_iter < num_k_iters - (NUM_STAGES - 1); k_iter++) {
    // cp.async prefetch
    __syncthreads();  // wait MMA
    load_AB(k_iter + NUM_STAGES - 1);  // cp.async

    // MMA
    asm volatile("cp.async.wait_group %0;" :: "n"(NUM_STAGES - 1));  // wait cp.async
    __syncthreads();
    compute(k_iter);
  }

  for (int k_iter = num_k_iters - (NUM_STAGES - 1); k_iter < num_k_iters; k_iter++) {
    asm volatile("cp.async.commit_group;");  // preserve loop invariance
    asm volatile("cp.async.wait_group %0;" :: "n"(NUM_STAGES - 1));  // wait cp.async
    __syncthreads();
    compute(k_iter);
  }

  for (int m = 0; m < WARP_M / MMA_M; m++)
    for (int n = 0; n < WARP_N / MMA_N; n++) {
      const int row = m * MMA_M + (lane_id / 4);
      const int col = n * MMA_N + (lane_id % 4) * 2;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C + ((row + 0) * N + col))[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C + ((row + 8) * N + col))[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v0(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 3;

  const int grid_size = (M / BLOCK_M) * (N / BLOCK_N);
  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

  auto kernel = matmul_v0_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kernel<<<grid_size, TB_SIZE, smem_size>>>(A, B, C, M, N, K);
}
