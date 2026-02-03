#include "common.h"
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

template <int TB_SIZE, int HEIGHT, int WIDTH, int OUT_STRIDE>
__device__
void global_to_shared(const nv_bfloat16 *in, int in_stride, nv_bfloat16 *out, int tid) {
  // number of elements to do 128-bit/16-byte load
  // e.g. FP32 -> 4 elements, BF16 -> 8 elements.
  using TLoad = uint4;
  constexpr int num_elems = sizeof(TLoad) / sizeof(nv_bfloat16);

  // NOTE: write loop this way to make sure the compiler can fully unroll it.
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    TLoad tmp = reinterpret_cast<const TLoad *>(in + row * in_stride + col)[0];
    reinterpret_cast<TLoad *>(out + row * OUT_STRIDE + col)[0] = tmp;
  }
}

template <int TB_SIZE, int HEIGHT, int WIDTH, int OUT_STRIDE, bool use_swizzle>
__device__
void global_to_shared_async(const nv_bfloat16 *in, int in_stride, nv_bfloat16 *out, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);  // cp.async cp-size = 16

  // convert to shared state space outside of the loop
  // TODO: move this to kernel body
  uint32_t out_addr = cvta_shared(out);

  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    uint32_t dst_addr = out_addr + (row * OUT_STRIDE + col) * sizeof(nv_bfloat16);
    if constexpr (use_swizzle)
      dst_addr = swizzle<OUT_STRIDE * sizeof(nv_bfloat16)>(dst_addr);
    cp_async(dst_addr, in + row * in_stride + col);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int SHM_STRIDE, bool use_cp_async,
          bool use_swizzle>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v1_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
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
  static_assert(use_cp_async || !use_swizzle); // use_swizzle=true requires use_cp_async=true
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  // each warp will do (NUM_MMA_M * NUM_MMA_N) MMAs
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;

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

  extern __shared__ nv_bfloat16 shm[];
  nv_bfloat16 *A_shared = shm;                               // BLOCK_M * BLOCK_K
  nv_bfloat16 *B_shared = A_shared + (BLOCK_M * SHM_STRIDE); // BLOCK_N * BLOCK_K

  // all registers are 32-bit (4-byte)
  // - we accumulate to FP32, which is exactly 32-bit
  // - our inputs are FP16/BF16, hence each register holds 2 elements
  // - inputs and accumulate are distributed across 32 threads in a warp
  // for m16n8k8, each thread holds
  // - 4 output float
  // - 4 input A FP16/BF16
  // - 2 input B FP16/BF16
  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE;
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE;
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {};

  for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    if constexpr (use_cp_async) {
      global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K, SHM_STRIDE, use_swizzle>(A, K, A_shared, tid);
      global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K, SHM_STRIDE, use_swizzle>(B, K, B_shared, tid);
      cp_async_wait_all();
    } else {
      global_to_shared<TB_SIZE, BLOCK_M, BLOCK_K, SHM_STRIDE>(A, K, A_shared, tid);
      global_to_shared<TB_SIZE, BLOCK_N, BLOCK_K, SHM_STRIDE>(B, K, B_shared, tid);
    }
    __syncthreads();

    for (int mma_k = 0; mma_k < BLOCK_K; mma_k += MMA_K) {
      // for m16n8k8
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688
      //   A\B   [8x8-0]
      // [8x8-0]
      // [8x8-1]
      // where each [8x8] matrix can be loaded from shared memory with ldmatrix
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix

      // for m16n8k16
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
      //                [8x8-0]
      //       A\B      [8x8-1]
      // [8x8-0][8x8-2]
      // [8x8-1][8x8-3]

      // select the tile this warp is responsible for
      const nv_bfloat16 *A_shm_warp = A_shared + (warp_id_m * WARP_M) * SHM_STRIDE + mma_k;
      const nv_bfloat16 *B_shm_warp = B_shared + (warp_id_n * WARP_N) * SHM_STRIDE + mma_k;

      // to use ldmatrix: each thread holds the address of 1 row e.g.
      // - thread 0 holds address of row 0
      // - thread 1 holds address of row 1, and so on
      // when loading multiple matrices, thread0-7 specifies the 1st matrix,
      // thread 8-15 specifies the 2nd matrix, and so on

      // load B to registers
      uint32_t B_reg[NUM_MMA_N][num_B_regs];
      for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
        // NOTE: we can reduce unnecessary address calculation if we know MMA_K=8 or 16
        // convert generic address to .shared state space address expected by inline PTX
        const nv_bfloat16 *B_ptr = B_shm_warp + (mma_id_n * MMA_N + (lane_id % 8)) * SHM_STRIDE + (lane_id / 8) * 8;
        uint32_t B_addr = cvta_shared(B_ptr);
        if constexpr (use_swizzle)
          B_addr = swizzle<SHM_STRIDE * sizeof(nv_bfloat16)>(B_addr);
        ldmatrix_x2(B_reg[mma_id_n], B_addr);
      }

      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        // load A to registers
        uint32_t A_reg[num_A_regs];
        const nv_bfloat16 *A_ptr = A_shm_warp + (mma_id_m * MMA_M + (lane_id % 16)) * SHM_STRIDE + (lane_id / 16) * 8;
        uint32_t A_addr = cvta_shared(A_ptr);
        if constexpr (use_swizzle)
          A_addr = swizzle<SHM_STRIDE * sizeof(nv_bfloat16)>(A_addr);
        ldmatrix_x4(A_reg, A_addr);

        // call mma
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++)
          mma_m16n8k16(A_reg, B_reg[mma_id_n], acc[mma_id_m][mma_id_n]);
      }
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K;
  }

  // check output layout here
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
  // NOTE: we can do some warp shuffle to get coalesced write
  for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 4);
      const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;
      nv_bfloat16 *C_local = C + row * N + col;

      float *regs = acc[mma_id_m][mma_id_n];
      reinterpret_cast<nv_bfloat162 *>(C_local)[0]         = __float22bfloat162_rn({regs[0], regs[1]});  // c0 and c1
      reinterpret_cast<nv_bfloat162 *>(C_local + 8 * N)[0] = __float22bfloat162_rn({regs[2], regs[3]});  // c2 and c3
    }
}

void matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int SHM_STRIDE = BLOCK_K; // no padding
  const int use_cp_async = false;
  const int use_swizzle = false;

  auto kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void matmul_v2(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int SHM_STRIDE = BLOCK_K; // no padding
  const int use_cp_async = true;
  const int use_swizzle = false;

  auto kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void matmul_v3(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int SHM_STRIDE = BLOCK_K + 8; // pad shmem to avoid bank conflict
  const int use_cp_async = true;
  const int use_swizzle = false;

  auto kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void matmul_v4(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int SHM_STRIDE = BLOCK_K;
  const int use_cp_async = true;
  const int use_swizzle = true;

  auto kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}
