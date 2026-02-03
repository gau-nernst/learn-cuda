#include "common.h"
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

template <int TB_SIZE, int HEIGHT, int WIDTH>
__device__
void global_to_shared_async(const nv_bfloat16 *in, int in_stride, uint32_t out, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    // NOTE: perhaps we can move swizzle out of this loop as well
    uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(out + (row * WIDTH + col) * sizeof(nv_bfloat16));
    cp_async(dst_addr, in + row * in_stride + col);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v5_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
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
  extern __shared__ nv_bfloat16 shm[];
  const uint32_t A_shared = cvta_shared(shm);                                     // BLOCK_M * BLOCK_K
  const uint32_t B_shared = A_shared + (BLOCK_M * BLOCK_K) * sizeof(nv_bfloat16); // BLOCK_N * BLOCK_K

  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE; // 4
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE; // 4
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {};

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
  const int A_offk = (lane_id / 16) * 8;
  const uint32_t A_shm_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(A_shared + (A_offm * BLOCK_K + A_offk) * sizeof(nv_bfloat16));

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
  const int B_offk = ((lane_id % 16) / 8) * 8;
  const uint32_t B_shm_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(B_shared + (B_offn * BLOCK_K + B_offk) * sizeof(nv_bfloat16));

  for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, tid);
    global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared, tid);
    cp_async_wait_all();
    __syncthreads();

    for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++) {
      // iterate MMA_K=16 -> increment bit5 (32 bytes) -> affects swizzled bits
      // assume we have alignment (bit0-6 are all zeros), increment bit5
      // is equivalent to XOR mma_id_k directly, which is commutative with swizzling
      // -> we can move swizzling outside of this loop
      // the kernel compiles to fewer instructions, but no speedup

      // load B to registers
      uint32_t B_reg[NUM_MMA_N][num_B_regs];
      for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n += 2) {
        const uint32_t B_addr = B_shm_thread + mma_id_n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_reg[mma_id_n], B_addr ^ (mma_id_k * MMA_K * sizeof(nv_bfloat16)));
      }

      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        // load A to registers
        uint32_t A_reg[num_A_regs];
        const uint32_t A_addr = A_shm_thread + mma_id_m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_reg, A_addr ^ (mma_id_k * MMA_K * sizeof(nv_bfloat16)));

        // call mma
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++)
          mma_m16n8k16(A_reg, B_reg[mma_id_n], acc[mma_id_m][mma_id_n]);
      }
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K;
  }

  for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 4);
      const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;
      nv_bfloat16 *C_local = C + row * N + col;

      float *regs = acc[mma_id_m][mma_id_n];
      reinterpret_cast<nv_bfloat162 *>(C_local)[0]         = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_local + 8 * N)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v5(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  // const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;  // same as previous kernels
  const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64; // this is only faster for this kernel
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;

  auto kernel = matmul_v5_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}
