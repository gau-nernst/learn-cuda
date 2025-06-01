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
template <int SIZE> __device__ void cp_async(uint32_t dst, const void *src) {
  // .ca means cache to L1 and L2. .cg means cache to L2 only.
  if constexpr (SIZE == 4)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
  else if constexpr (SIZE == 8)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src));
  else if constexpr (SIZE == 16)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
};
__device__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); };
__device__ void cp_async_wait_group(int N) { asm volatile("cp.async.wait_group {%0};" ::"r"(N)); };
__device__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); };

template <int TB_SIZE, int HEIGHT, int WIDTH, typename T>
__device__ void global_to_shared(const T *in, int in_stride, T *out, int out_stride, int tid) {
  // number of elements to do 128-bit/16-byte load
  // e.g. FP32 -> 4 elements, BF16 -> 8 elements.
  using TLoad = uint4;
  constexpr int num_elems = sizeof(TLoad) / sizeof(T);

  for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += TB_SIZE * num_elems) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    TLoad tmp = reinterpret_cast<const TLoad *>(&in[row * in_stride + col])[0];
    reinterpret_cast<TLoad *>(&out[row * out_stride + col])[0] = tmp;
  }
}

template <int TB_SIZE, int HEIGHT, int WIDTH, typename T>
__device__ void global_to_shared_async(const T *in, int in_stride, T *out, int out_stride, int tid) {
  constexpr int cp_size = 16;
  constexpr int num_elems = cp_size / sizeof(T);

  for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += TB_SIZE * num_elems) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    cp_async<cp_size>(cvta_shared(out + row * out_stride + col), in + row * in_stride + col);
  }
  cp_async_wait_all();
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_K, bool use_cp_async, typename T>
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
  T *A_shared = shm;                            // BLOCK_M * BLOCK_K
  T *B_shared = A_shared + (BLOCK_M * BLOCK_K); // BLOCK_N * BLOCK_K

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
      global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, BLOCK_K, tid);
      global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(B, K, B_shared, BLOCK_K, tid);
    } else {
      global_to_shared<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, BLOCK_K, tid);
      global_to_shared<TB_SIZE, BLOCK_M, BLOCK_K>(B, K, B_shared, BLOCK_K, tid);
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
      const T *A_shm_warp = A_shared + (warp_id_m * WARP_M) * BLOCK_K + mma_k;
      const T *B_shm_warp = B_shared + (warp_id_n * WARP_N) * BLOCK_K + mma_k;

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
        const T *B_ptr = B_shm_warp + (mma_id_n * MMA_N + (lane_id % 8)) * BLOCK_K + (lane_id / 8) * 8;
        ldmatrix<num_B_regs>(B_reg[mma_id_n], cvta_shared(B_ptr));
      }

      for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        // load A to registers
        uint32_t A_reg[num_A_regs];
        const T *A_ptr = A_shm_warp + (mma_id_m * MMA_M + (lane_id % 16)) * BLOCK_K + (lane_id / 16) * 8;
        ldmatrix<num_A_regs>(A_reg, cvta_shared(A_ptr));

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

  using T = nv_bfloat16;
  using KernelFn = void (*)(const T *A, const T *B, T *C, int M, int N, int K);
  KernelFn kernel = matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_K, false>;

  const int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(T);

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

  using T = nv_bfloat16;
  using KernelFn = void (*)(const T *A, const T *B, T *C, int M, int N, int K);
  KernelFn kernel = matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_K, true>;

  const int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(T);

  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<grid_size, TB_SIZE, shm_size>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

// NOTE: to re-do
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp
template <int WIDTH, typename T> __device__ int swizzle(int x) {
  constexpr int num_elems = 16 / sizeof(T);
  constexpr int stride = WIDTH / num_elems; // stride for 16-byte word.
  // we don't touch the first MBase bits because they belong to the same 16-byte row (8x 16-bit).
  constexpr int MBase = log2_int(num_elems);
  // TODO: seems like we have to add 1 to BBits? bug in logic?
  // we permute BBits, which is the no. of non-overlapping bits between row index and 4-bank-group index.
  constexpr int BBits = std::min(log2_int(stride), 3);
  constexpr int SShift = log2_int(stride); // relative difference from 4-bank-group index to row index.

  constexpr int mask = ((1 << BBits) - 1) << MBase; // BBits 1s and MBase 0sa
  if constexpr (BBits == 0)
    return x;
  else
    return x ^ ((x >> SShift) & mask);
}

template <int BLOCK_SIZE, int HEIGHT, int WIDTH, typename T>
__device__ void load_shared_swizzle(const T *in, int in_row_stride, T *out, int tid) {
  constexpr int num_elems = 16 / sizeof(T);

  for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE * num_elems) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    uint4 tmp = reinterpret_cast<const uint4 *>(&in[row * in_row_stride + col])[0];

    int swizzled_idx = swizzle<WIDTH, T>(row * WIDTH + col);
    reinterpret_cast<uint4 *>(&out[swizzled_idx])[0] = tmp;
  }
}
