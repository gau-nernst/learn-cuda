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
// col is in unit of 16-byte word
template <int STRIDE>
__device__
int swizzle(int row, int col) {
  if constexpr (STRIDE > 16)
    col ^= (row % 8) / std::max(128 / STRIDE, 1);
  return row * STRIDE + col * 16;
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline
void gmem_to_smem(int dst, const T *src, int src_stride, int tid) {
  static_assert(WIDTH * sizeof(T) >= 16);
  constexpr int num_elems = 16 / sizeof(T);

  auto load = [&](int idx) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const int dst_addr = dst + swizzle<WIDTH * sizeof(T)>(row, col / num_elems);
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
  int SFA, short byte_id_A, short thread_id_A,
  int SFB, short byte_id_B, short thread_id_B
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
                "r"(SFA), "h"(byte_id_A), "h"(thread_id_A),
                "r"(SFB), "h"(byte_id_B), "h"(thread_id_B));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__ inline
int elect_sync() {
  int pred = 0;
  asm volatile(
    "{\n"
    ".reg .pred P;\n"
    "elect.sync _|P, %1;\n"
    "@P mov.s32 %0, 1;\n"
    "}"
    : "+r"(pred) : "r"(0xFFFF'FFFF)
  );
  return pred;
}

template <typename T>
__device__ inline
T warp_uniform(T x) { return __shfl_sync(0xFFFF'FFFF, x, 0); }

__device__ inline
void mbarrier_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ inline
void mbarrier_arrive(int addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
}

__device__ inline
void mbarrier_arrive_expect_tx(int addr, int size) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(addr), "r"(size) : "memory");
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__ inline
void mbarrier_wait(int mbar_addr, int phase) {
  int ticks = 0x989680;  // this is optional
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n"
    "@!P1 bra.uni LAB_WAIT;\n"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__device__ inline
void tma_g2s(int dst, const void *src, int size, int mbar_addr) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
              "[%0], [%1], %2, [%3];"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr)
              : "memory");
}

__device__ inline
void tma_2d_g2s(int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
              "[%0], [%1, {%2, %3}], [%4];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr)
              : "memory");
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
