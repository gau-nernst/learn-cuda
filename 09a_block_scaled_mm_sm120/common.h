#pragma once

#include <iostream>

#include <cuda_bf16.h>

#define CUDA_CHECK(x)                                                                                                  \
  {                                                                                                                    \
    auto error = x;                                                                                                    \
    if (error != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }

constexpr int WARP_SIZE = 32;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;

__device__ __host__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// NOTE: stride in bytes
template <int STRIDE>
__device__
int swizzle(int index) {
  // no need swizzling
  if constexpr (STRIDE <= 16)
    return index;

  int row_idx = (index / STRIDE) % 8;
  int bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline
void global_to_shared_swizzle(int dst, const T *src, int src_stride, int tid) {
  static_assert(WIDTH * sizeof(T) >= 16);
  constexpr int num_elems = 16 / sizeof(T);

  auto load = [&](int idx) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const int dst_addr = swizzle<WIDTH * sizeof(T)>(dst + (row * WIDTH + col) * sizeof(T));
    const T *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  };

  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
  for (int iter = 0; iter < num_iters; iter++)
    load((iter * TB_SIZE + tid) * num_elems);

  // handle the case when tile size is not divisible by threadblock size
  if constexpr ((HEIGHT * WIDTH) % (TB_SIZE * num_elems) != 0) {
    const int idx = (num_iters * TB_SIZE + tid) * num_elems;
    if (idx < HEIGHT * WIDTH)
      load(idx);
  }
}

template <int num>
__device__ inline
void ldmatrix(int *regs, int addr) {
  static_assert(num == 1 || num == 2 || num == 4);
  if constexpr (num == 1)
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                : "=r"(regs[0])
                : "r"(addr));
  else if constexpr (num == 2)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                : "=r"(regs[0]), "=r"(regs[1])
                : "r"(addr));
  else if constexpr (num == 4)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                : "r"(addr));
}

__device__ inline
void mma_mxfp8(
  int A[4], int B[2], float C[4],
  int scale_A, short byte_id_A, short thread_id_A,
  int scale_B, short byte_id_B, short thread_id_B
) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%0, %1, %2, %3}, "
              "{%10}, {%11, %12}, "
              "{%13}, {%14, %15};"
              : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "r"(scale_A), "h"(byte_id_A), "h"(thread_id_A),
                "r"(scale_B), "h"(byte_id_B), "h"(thread_id_B));
}

template <typename T, typename... Args>
void launch_kernel(
  T *kernel,
  int num_blocks,
  int block_size,
  int shm_size,
  Args... args) {
  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
  kernel<<<num_blocks, block_size, shm_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}
