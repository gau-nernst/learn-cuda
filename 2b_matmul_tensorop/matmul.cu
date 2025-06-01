#include "mma.cuh"
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <stdio.h>

#define PRINT_IF(cond, ...)                                                                                            \
  if (cond)                                                                                                            \
    printf(__VA_ARGS__);

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; } // https://stackoverflow.com/a/1804686
constexpr int WARP_SIZE = 32;

template <int BLOCK_SIZE, int HEIGHT, int WIDTH, typename T>
__device__ void load_b128(const T *in, int in_row_stride, T *out, int out_row_stride, int tid) {
  // number of elements to do 128-bit/16-byte load
  // e.g. FP32 -> 4 elements, BF16 -> 8 elements.
  using load_type = uint4;
  constexpr int num_elems = sizeof(load_type) / sizeof(T);

  for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE * num_elems) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    load_type tmp = reinterpret_cast<const load_type *>(&in[row * in_row_stride + col])[0];
    reinterpret_cast<load_type *>(&out[row * out_row_stride + col])[0] = tmp;
  }
}

template <typename T> __device__ ushort f32_to_b16(float x);
template <> __device__ ushort f32_to_b16<half>(float x) { return __half_as_ushort(__float2half(x)); }
template <> __device__ ushort f32_to_b16<nv_bfloat16>(float x) { return __bfloat16_as_ushort(__float2bfloat16(x)); }

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_M, int MMA_N, int MMA_K, typename T>
__global__ void matmul_v1_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
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

  const int num_blocks_per_row = cdiv(N, BLOCK_N);
  const int block_id_m = block_id / num_blocks_per_row;
  const int block_id_n = block_id % num_blocks_per_row;
  const int offset_m = block_id_m * BLOCK_M;
  const int offset_n = block_id_n * BLOCK_N;

  constexpr int num_warps_per_row = BLOCK_N / WARP_N;
  const int warp_id_m = warp_id / num_warps_per_row;
  const int warp_id_n = warp_id % num_warps_per_row;
  const int warp_tile_offset_m = warp_id_m * WARP_M;
  const int warp_tile_offset_n = warp_id_n * WARP_N;

  // A is row-major, B is column-major
  A += offset_m * K;
  B += offset_n * K;

  __shared__ T A_shared[BLOCK_M * BLOCK_K];
  __shared__ T B_shared[BLOCK_N * BLOCK_K];

  // 32-bit (4-byte) registers
  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {0.0f}; // for m16n8k8, each thread holds 4 output float

  // first A and B warp-tile along BLOCK_K dim (we will iterate along BLOCK_K with step_size=MMA_K)
  const T *A_warp_tile = reinterpret_cast<const T *>(A_shared) + warp_tile_offset_m * BLOCK_K;
  const T *B_warp_tile = reinterpret_cast<const T *>(B_shared) + warp_tile_offset_n * BLOCK_K;

  for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    // TODO: use async copy
    load_b128<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, BLOCK_K, tid);
    load_b128<TB_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared, BLOCK_K, tid);
    __syncthreads();

    for (int warp_k = 0; warp_k < BLOCK_K; warp_k += MMA_K) {
      // load data from shared memory to registers using ldmatrix
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix

      // convert generic address to .shared state space address expected by inline PTX
      // thread 0 holds address of row 0
      // thread 1 holds address of row 1, and so on
      uint32_t A_tile_addr = cvta_shared(A_warp_tile + lane_id * BLOCK_K + warp_k);
      uint32_t B_tile_addr = cvta_shared(B_warp_tile + lane_id * BLOCK_K + warp_k);

      // load B to registers
      // ldmatrix can only load 8x8 matrix. for 16x8 tile, we need to use x2
      // works for both m16n8k8 and m16n8k16
      uint32_t B_reg[NUM_MMA_N][num_B_regs];
      for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
        uint32_t B_local = B_tile_addr + (mma_tile_id_n * MMA_N * BLOCK_K) * sizeof(T);
        ldmatrix<num_B_regs>(B_reg[mma_tile_id_n], B_local);
      }

      for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++) {
        uint32_t A_reg[num_A_regs];
        uint32_t A_local = A_tile_addr + (mma_tile_id_m * MMA_M * BLOCK_K) * sizeof(T);
        ldmatrix<num_A_regs>(A_reg, A_local);

        // call mma
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688
        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
          mma<MMA_M, MMA_N, MMA_K, T>(A_reg, B_reg[mma_tile_id_n], acc[mma_tile_id_m][mma_tile_id_n]);
      }
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K;
  }

  const int C_offset_m = offset_m + warp_tile_offset_m;
  const int C_offset_n = offset_n + warp_tile_offset_n;
  C += C_offset_m * N + C_offset_n;

  // check output layout here
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-c-f16-f32
  // m16n8k16 has the same layout
  const int a0_row = lane_id >> 2;
  const int a0_col = (lane_id % 4) * 2;
  C += a0_row * N + a0_col;

  for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
    for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
      T *C_local = C + mma_tile_id_m * MMA_M * N + mma_tile_id_n * MMA_N;
      float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];
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
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
  const int WARP_M = 64, WARP_N = 64;
  const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

  const int TB_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, MMA_K>
      <<<grid_size, TB_SIZE>>>(A, B, C, M, N, K);
}

constexpr __device__ int log2_int(int x) { return x == 1 ? 0 : 1 + log2_int(x >> 1); }

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

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_M, int MMA_N, int MMA_K, typename T>
__global__ void matmul_v2_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
  static_assert(BLOCK_M % WARP_M == 0);
  static_assert(BLOCK_N % WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;

  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int num_blocks_per_row = cdiv(N, BLOCK_N);
  const int block_id_m = block_id / num_blocks_per_row;
  const int block_id_n = block_id % num_blocks_per_row;
  const int offset_m = block_id_m * BLOCK_M;
  const int offset_n = block_id_n * BLOCK_N;

  constexpr int num_warps_per_row = BLOCK_N / WARP_N;
  const int warp_id_m = warp_id / num_warps_per_row;
  const int warp_id_n = warp_id % num_warps_per_row;
  const int warp_tile_offset_m = warp_id_m * WARP_M;
  const int warp_tile_offset_n = warp_id_n * WARP_N;

  // A is row-major, B is column-major
  A += offset_m * K;
  B += offset_n * K;

  __shared__ T A_shared[BLOCK_M * BLOCK_K];
  __shared__ T B_shared[BLOCK_N * BLOCK_K];

  // 32-bit (4-byte) registers
  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {0.0f};

  for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    load_shared_swizzle<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, tid);
    load_shared_swizzle<BLOCK_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared, tid);
    __syncthreads();

    for (int warp_k = 0; warp_k < BLOCK_K; warp_k += MMA_K) {
      // load B to registers
      uint32_t B_reg[NUM_MMA_N][num_B_regs];
      for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
        const int B_offset = (warp_tile_offset_n + lane_id + mma_tile_id_n * MMA_N) * BLOCK_K + warp_k;
        const T *B_local = reinterpret_cast<const T *>(B_shared) + swizzle<BLOCK_K, T>(B_offset);
        ldmatrix<num_B_regs>(B_reg[mma_tile_id_n], cvta_shared(B_local));
      }

      // call mma
      for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++) {
        uint32_t A_reg[num_A_regs];
        const int A_offset = (warp_tile_offset_m + lane_id + mma_tile_id_m * MMA_M) * BLOCK_K + warp_k;
        const T *A_local = reinterpret_cast<const T *>(A_shared) + swizzle<BLOCK_K, T>(A_offset);
        ldmatrix<num_A_regs>(A_reg, cvta_shared(A_local));

        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
          mma<MMA_M, MMA_N, MMA_K, T>(A_reg, B_reg[mma_tile_id_n], acc[mma_tile_id_m][mma_tile_id_n]);
      }
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K;
  }

  const int C_offset_m = offset_m + warp_tile_offset_m;
  const int C_offset_n = offset_n + warp_tile_offset_n;
  C += C_offset_m * N + C_offset_n;

  // check output layout here
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-c-f16-f32
  const int a0_row = lane_id >> 2;
  const int a0_col = (lane_id % 4) * 2;
  C += a0_row * N + a0_col;

  for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
    for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
      T *C_local = C + mma_tile_id_m * MMA_M * N + mma_tile_id_n * MMA_N;
      float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];
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

void matmul_v2(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
  const int WARP_M = 64, WARP_N = 64;
  const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

  const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  matmul_v2_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, MMA_K>
      <<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}
