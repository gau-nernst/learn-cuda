#include "mma.cuh"
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <stdio.h>

#define PRINT_IF(cond, ...)                                                                                            \
  if (cond)                                                                                                            \
    printf(__VA_ARGS__);

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; } // https://stackoverflow.com/a/1804686
constexpr __device__ int log2_int(int x) { return x == 1 ? 0 : 1 + log2_int(x >> 1); }
constexpr int WARP_SIZE = 32;

template <typename T> __device__ ushort f32_to_b16(float x);
template <> __device__ ushort f32_to_b16<half>(float x) { return __half_as_ushort(__float2half(x)); }
template <> __device__ ushort f32_to_b16<nv_bfloat16>(float x) { return __bfloat16_as_ushort(__float2bfloat16(x)); }

// convert generic address (C++ address, 64-bit) to shared state space address (32-bit)
// all PTX instructions expect share memory address to be in shared state space (not 100%)
__device__ uint32_t cvta_shared(const void *ptr) { return static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); }

// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy
template <int size> __device__ void cp_async(uint32_t dst, const void *src) {
  // .ca means cache to L1 and L2. .cg means cache to L2 only.
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(dst), "l"(src), "n"(size));
};
__device__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); };
template <int N> __device__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); };
__device__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); };

// NOTE: stride in bytes
template <int STRIDE> __device__ uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  int row_idx = (index / STRIDE) % 8;
  int bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int TB_SIZE, int HEIGHT, int WIDTH, int OUT_STRIDE, typename T>
__device__ void global_to_shared(const T *in, int in_stride, T *out, int tid) {
  // number of elements to do 128-bit/16-byte load
  // e.g. FP32 -> 4 elements, BF16 -> 8 elements.
  using TLoad = uint4;
  constexpr int num_elems = sizeof(TLoad) / sizeof(T);

  for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += TB_SIZE * num_elems) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    TLoad tmp = reinterpret_cast<const TLoad *>(in + row * in_stride + col)[0];
    reinterpret_cast<TLoad *>(out + row * OUT_STRIDE + col)[0] = tmp;
  }
}

template <int TB_SIZE, int HEIGHT, int WIDTH, int OUT_STRIDE, bool use_swizzle, typename T>
__device__ void global_to_shared_async(const T *in, int in_stride, T *out, int tid) {
  constexpr int cp_size = 16;
  constexpr int num_elems = cp_size / sizeof(T);

  for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += TB_SIZE * num_elems) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    uint32_t dst_addr = cvta_shared(out + row * OUT_STRIDE + col);
    if constexpr (use_swizzle)
      dst_addr = swizzle<OUT_STRIDE * sizeof(T)>(dst_addr);
    cp_async<cp_size>(dst_addr, in + row * in_stride + col);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_K, int SHM_STRIDE, bool use_cp_async,
          bool use_swizzle, typename T>
__global__ void
__launch_bounds__((BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE) // maxThreadsPerBlock
matmul_v1_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  static_assert(BLOCK_M % WARP_M == 0);
  static_assert(BLOCK_N % WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  static_assert(use_cp_async || !use_swizzle); // use_swizzle=true requires use_cp_async=true
  constexpr int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;

  // each warp will do (NUM_MMA_M * NUM_MMA_N) MMAs
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;

  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // TODO: threadblock swizzling to improve L2 cache hit rate
  const int num_blocks_n = cdiv(N, BLOCK_N);
  const int bid_m = block_id / num_blocks_n;
  const int bid_n = block_id % num_blocks_n;
  const int offset_m = bid_m * BLOCK_M;
  const int offset_n = bid_n * BLOCK_N;

  constexpr int num_warps_n = BLOCK_N / WARP_N;
  const int warp_id_m = warp_id / num_warps_n;
  const int warp_id_n = warp_id % num_warps_n;

  // A is row-major, B is column-major, C is row-major
  A += offset_m * K;
  B += offset_n * K;
  C += (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);

  extern __shared__ T shm[];
  T *A_shared = shm;                               // BLOCK_M * BLOCK_K
  T *B_shared = A_shared + (BLOCK_M * SHM_STRIDE); // BLOCK_N * BLOCK_K

  // all registers are 32-bit (4-byte)
  // - we accumulate to FP32, which is exactly 32-bit
  // - our inputs are FP16/BF16, hence each register holds 2 elements
  // - inputs and accumulate are distributed across 32 threads in a warp
  // for m16n8k8, each thread holds
  // - 4 output float
  // - 4 input A FP16/BF16
  // - 2 input B FP16/BF16
  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;
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
      const T *A_shm_warp = A_shared + (warp_id_m * WARP_M) * SHM_STRIDE + mma_k;
      const T *B_shm_warp = B_shared + (warp_id_n * WARP_N) * SHM_STRIDE + mma_k;

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
        const T *B_ptr = B_shm_warp + (mma_id_n * MMA_N + (lane_id % 8)) * SHM_STRIDE + (lane_id / 8) * 8;
        uint32_t B_addr = cvta_shared(B_ptr);
        if constexpr (use_swizzle)
          B_addr = swizzle<SHM_STRIDE * sizeof(T)>(B_addr);
        ldmatrix<num_B_regs>(B_reg[mma_id_n], B_addr);
      }

      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        // load A to registers
        uint32_t A_reg[num_A_regs];
        const T *A_ptr = A_shm_warp + (mma_id_m * MMA_M + (lane_id % 16)) * SHM_STRIDE + (lane_id / 16) * 8;
        uint32_t A_addr = cvta_shared(A_ptr);
        if constexpr (use_swizzle)
          A_addr = swizzle<SHM_STRIDE * sizeof(T)>(A_addr);
        ldmatrix<num_A_regs>(A_reg, A_addr);

        // call mma
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++)
          mma<MMA_K, T>(A_reg, B_reg[mma_id_n], acc[mma_id_m][mma_id_n]);
      }
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K;
  }

  // check output layout here
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-c-f16-f32
  // m16n8k16 has the same layout
  const int a0_row = lane_id >> 2;
  const int a0_col = (lane_id % 4) * 2;
  C += a0_row * N + a0_col;

  for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
      T *C_local = C + mma_id_m * MMA_M * N + mma_id_n * MMA_N;
      float *acc_frag = acc[mma_id_m][mma_id_n];
      ushort2 tmp;

      // write a0 and a1
      tmp.x = f32_to_b16<T>(acc_frag[0]);
      tmp.y = f32_to_b16<T>(acc_frag[1]);
      reinterpret_cast<ushort2 *>(C_local)[0] = tmp;

      // write a2 and a3
      tmp.x = f32_to_b16<T>(acc_frag[2]);
      tmp.y = f32_to_b16<T>(acc_frag[3]);
      reinterpret_cast<ushort2 *>(C_local + 8 * N)[0] = tmp;
    }
}

void matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int WARP_M = 64, WARP_N = 64;
  const int MMA_K = 16;
  const int SHM_STRIDE = BLOCK_K; // no padding
  const int use_cp_async = false;
  const int use_swizzle = false;

  using T = nv_bfloat16;
  using KernelFn = void (*)(const T *A, const T *B, T *C, int M, int N, int K);
  KernelFn kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_K, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(T);

  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<grid_size, TB_SIZE, shm_size>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

void matmul_v2(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int WARP_M = 64, WARP_N = 64;
  const int MMA_K = 16;
  const int SHM_STRIDE = BLOCK_K; // no padding
  const int use_cp_async = true;
  const int use_swizzle = false;

  using T = nv_bfloat16;
  using KernelFn = void (*)(const T *A, const T *B, T *C, int M, int N, int K);
  KernelFn kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_K, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(T);

  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<grid_size, TB_SIZE, shm_size>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

void matmul_v3(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int WARP_M = 64, WARP_N = 64;
  const int MMA_K = 16;
  const int SHM_STRIDE = BLOCK_K + 8; // pad shmem to avoid bank conflict
  const int use_cp_async = true;
  const int use_swizzle = false;

  using T = nv_bfloat16;
  using KernelFn = void (*)(const T *A, const T *B, T *C, int M, int N, int K);
  KernelFn kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_K, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(T);

  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<grid_size, TB_SIZE, shm_size>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

void matmul_v4(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int WARP_M = 64, WARP_N = 64;
  const int MMA_K = 16;
  const int SHM_STRIDE = BLOCK_K;
  const int use_cp_async = true;
  const int use_swizzle = true;

  using T = nv_bfloat16;
  using KernelFn = void (*)(const T *A, const T *B, T *C, int M, int N, int K);
  KernelFn kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_K, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(T);

  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<grid_size, TB_SIZE, shm_size>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int NUM_STAGES, typename T>
__global__ void
__launch_bounds__((BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE) // maxThreadsPerBlock
matmul_v5_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  static_assert(BLOCK_M % WARP_M == 0);
  static_assert(BLOCK_N % WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  constexpr int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;

  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // TODO: threadblock swizzling to improve L2 cache hit rate
  const int num_blocks_n = cdiv(N, BLOCK_N);
  const int bid_m = block_id / num_blocks_n;
  const int bid_n = block_id % num_blocks_n;
  const int offset_m = bid_m * BLOCK_M;
  const int offset_n = bid_n * BLOCK_N;

  constexpr int num_warps_n = BLOCK_N / WARP_N;
  const int warp_id_m = warp_id / num_warps_n;
  const int warp_id_n = warp_id % num_warps_n;

  // A is row-major, B is column-major, C is row-major
  A += offset_m * K;
  B += offset_n * K;
  C += (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);

  extern __shared__ T shm[];

  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {};

  // initiate NUM_STAGES-1 async global->shared
  for (int stage = 0; stage < NUM_STAGES - 1; stage++) {
    if (stage * BLOCK_K < K) {
      T *A_shared = shm + stage * (BLOCK_M + BLOCK_N) * BLOCK_K; // BLOCK_M, BLOCK_K
      T *B_shared = A_shared + BLOCK_M * BLOCK_K;                // BLOCK_N, BLOCK_K
      global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K, BLOCK_K, true>(A, K, A_shared, tid);
      global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K, BLOCK_K, true>(B, K, B_shared, tid);

      // NOTE: A and B pointers now track position for global->shared load
      A += BLOCK_K;
      B += BLOCK_K;
    }

    // mark this stage as a commit group. we still need to commit empty groups
    // so that we have invariance of NUM_STAGES-1 prefetch in flight.
    cp_async_commit_group();
  }

  // loop invariance: NUM_STAGES-1 global->shared prefetch in flight
  for (int k_iter = 0; k_iter < K / BLOCK_K; k_iter++) {
    // select the shared memory of this stage
    const int stage = k_iter % NUM_STAGES;
    T *A_shared = shm + stage * (BLOCK_M + BLOCK_N) * BLOCK_K;
    T *B_shared = A_shared + BLOCK_M * BLOCK_K;

    // wait for the 1st commit group to finish i.e. FIFO
    // this consumes 1 prefetch
    // TODO: learn about barrier
    cp_async_wait_group<NUM_STAGES - 2>();
    __syncthreads();

    // we should also pipeline this loop? shared->registers
    for (int mma_k = 0; mma_k < BLOCK_K; mma_k += MMA_K) {
      // select the tile this warp is responsible for
      const T *A_shm_warp = A_shared + (warp_id_m * WARP_M) * BLOCK_K + mma_k;
      const T *B_shm_warp = B_shared + (warp_id_n * WARP_N) * BLOCK_K + mma_k;

      // load B to registers
      uint32_t B_reg[NUM_MMA_N][num_B_regs];
      for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
        const T *B_ptr = B_shm_warp + (mma_id_n * MMA_N + (lane_id % 8)) * BLOCK_K + (lane_id / 8) * 8;
        const uint32_t B_addr = swizzle<BLOCK_K * sizeof(T)>(cvta_shared(B_ptr));
        ldmatrix<num_B_regs>(B_reg[mma_id_n], B_addr);
      }

      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        // load A to registers
        uint32_t A_reg[num_A_regs];
        const T *A_ptr = A_shm_warp + (mma_id_m * MMA_M + (lane_id % 16)) * BLOCK_K + (lane_id / 16) * 8;
        const uint32_t A_addr = swizzle<BLOCK_K * sizeof(T)>(cvta_shared(A_ptr));
        ldmatrix<num_A_regs>(A_reg, A_addr);

        // call mma
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++)
          mma<MMA_K, T>(A_reg, B_reg[mma_id_n], acc[mma_id_m][mma_id_n]);
      }
    }

    // prefetch the next block. this produces 1 prefetch, restore loop invariance
    if ((k_iter + NUM_STAGES - 1) * BLOCK_K < K) {
      const int stage = (k_iter + NUM_STAGES - 1) % NUM_STAGES;
      A_shared = shm + stage * (BLOCK_M + BLOCK_N) * BLOCK_K;
      B_shared = A_shared + BLOCK_M * BLOCK_K;
      global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K, BLOCK_K, true>(A, K, A_shared, tid);
      global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K, BLOCK_K, true>(B, K, B_shared, tid);
      A += BLOCK_K;
      B += BLOCK_K;
    }
    cp_async_commit_group();
  }

  // check output layout here
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-c-f16-f32
  // m16n8k16 has the same layout
  const int a0_row = lane_id >> 2;
  const int a0_col = (lane_id % 4) * 2;
  C += a0_row * N + a0_col;

  for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
      T *C_local = C + mma_id_m * MMA_M * N + mma_id_n * MMA_N;
      float *acc_frag = acc[mma_id_m][mma_id_n];
      ushort2 tmp;

      // write a0 and a1
      tmp.x = f32_to_b16<T>(acc_frag[0]);
      tmp.y = f32_to_b16<T>(acc_frag[1]);
      reinterpret_cast<ushort2 *>(C_local)[0] = tmp;

      // write a2 and a3
      tmp.x = f32_to_b16<T>(acc_frag[2]);
      tmp.y = f32_to_b16<T>(acc_frag[3]);
      reinterpret_cast<ushort2 *>(C_local + 8 * N)[0] = tmp;
    }
}

void matmul_v5(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
  const int WARP_M = 64, WARP_N = 64;
  const int NUM_STAGES = 2;

  using T = nv_bfloat16;
  using KernelFn = void (*)(const T *A, const T *B, T *C, int M, int N, int K);
  KernelFn kernel = matmul_v5_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, NUM_STAGES>;

  const int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(T) * NUM_STAGES;

  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<grid_size, TB_SIZE, shm_size>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
