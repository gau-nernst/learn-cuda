#include "common.h"

#include <cuda_bf16.h>

// NOTE: stride in bytes
template <int STRIDE>
__device__ static
int swizzle(int index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  int row_idx = (index / STRIDE) % 8;
  int bits_to_xor = row_idx / std::max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
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
    int dst_addr = swizzle<WIDTH * sizeof(T)>(dst + idx * sizeof(T));
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
  static_assert(BLOCK_M % NUM_WARP_M == 0);
  static_assert(BLOCK_N % NUM_WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;
  constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // TODO: threadblock swizzling to improve L2 cache hit rate
  const int num_blocks_n = cdiv(N, BLOCK_N);
  const int bid_m = bid / num_blocks_n;
  const int bid_n = bid % num_blocks_n;
  const int offset_m = bid_m * BLOCK_M;
  const int offset_n = bid_n * BLOCK_N;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  // A is row-major, B is column-major, C is row-major
  A += offset_m * K;
  B += offset_n * K;
  C += (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);

  // convert shared memory address to 32-bit from the start
  extern __shared__ nv_bfloat16 smem[];
  const int smem_u32 = static_cast<int>(__cvta_generic_to_shared(smem));
  const int A_smem = smem_u32;
  const int B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);

  int A_rmem[NUM_MMA_K][NUM_MMA_M][4];
  int B_rmem[NUM_MMA_K][NUM_MMA_N][2];
  float C_rmem[NUM_MMA_M][NUM_MMA_N][4] = {};

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
  const int A_offk = (lane_id / 16) * 8;
  const int A_smem_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(A_smem + (A_offm * BLOCK_K + A_offk) * sizeof(nv_bfloat16));

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
  const int B_offk = ((lane_id % 16) / 8) * 8;
  const int B_smem_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(B_smem + (B_offn * BLOCK_K + B_offk) * sizeof(nv_bfloat16));

  // pre-compute the address for each stage
  int A_buffers[NUM_STAGES];
  int B_buffers[NUM_STAGES];
  for (int stage = 0; stage < NUM_STAGES; stage++) {
    A_buffers[stage] = A_smem_thread + stage * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
    B_buffers[stage] = B_smem_thread + stage * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
  }

  const int num_k_iters = cdiv(K, BLOCK_K);

  auto load_AB = [&](int k_iter) {
    if (k_iter < num_k_iters) {
      // select the correct shared memory buffer
      const int A_shared = A_smem + (k_iter % NUM_STAGES) * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
      const int B_shared = B_smem + (k_iter % NUM_STAGES) * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);

      gmem_to_smem<BLOCK_M, BLOCK_K, TB_SIZE>(A_shared, A, K, tid);
      gmem_to_smem<BLOCK_N, BLOCK_K, TB_SIZE>(B_shared, B, K, tid);

      // A/B pointer tracks position for gmem->smem load
      A += BLOCK_K;
      B += BLOCK_K;
    }
    asm volatile("cp.async.commit_group;");
  };

  // initiate NUM_STAGES-1 stages
  for (int stage = 0; stage < NUM_STAGES - 1; stage++)
    load_AB(stage);

  // loop invariance: there is always NUM_STAGES - 1 prefetch stages in-flight
  // thanks to pipelining, this loop now only has 1 __syncthreads()
  for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    // wait for previous MMA to finish using the shared buffer
    __syncthreads();

    // prefetch the next stage. add 1 more stage to the pipeline
    load_AB(k_iter + NUM_STAGES - 1);

    // wait for the 1st stage to finish. remove 1 stage from the pipeline
    // -> restore loop invariance
    asm volatile("cp.async.wait_group %0;" :: "n"(NUM_STAGES - 1));
    __syncthreads();

    // A smem->rmem
    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        int A_addr = A_buffers[k_iter % NUM_STAGES];
        A_addr += mma_id_m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        A_addr ^= mma_id_k * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_rmem[mma_id_k][mma_id_m], A_addr);
      }

    // B smem->rmem
    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++)
      for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n += 2) {
        int B_addr = B_buffers[k_iter % NUM_STAGES];
        B_addr += mma_id_n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        B_addr ^= mma_id_k * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_rmem[mma_id_k][mma_id_n], B_addr);
      }

    // do MMA. NUM_STAGES-1 prefetch stages are still on-going
    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++)
      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++)
          mma_m16n8k16(A_rmem[mma_id_k][mma_id_m],
                       B_rmem[mma_id_k][mma_id_n],
                       C_rmem[mma_id_m][mma_id_n]);
  }

  for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 4);
      const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;
      nv_bfloat16 *C_local = C + row * N + col;

      float *regs = C_rmem[mma_id_m][mma_id_n];
      reinterpret_cast<nv_bfloat162 *>(C_local)[0]         = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_local + 8 * N)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v0(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 2;

  const int grid_size = (M / BLOCK_M) * (N / BLOCK_N);
  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

  auto kernel = matmul_v0_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kernel<<<grid_size, TB_SIZE, smem_size>>>(A, B, C, M, N, K);
}
