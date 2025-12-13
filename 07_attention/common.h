#pragma once

#include <iostream>
#include <cstdint>

#include <cuda_bf16.h>

#define CUDA_CHECK(x)                                                                                                  \
  {                                                                                                                    \
    auto error = x;                                                                                                    \
    if (error != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }

inline constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}


__device__ inline
bool isthread0(){
  return threadIdx.x  == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y  == 0 && blockIdx.z  == 0;
}



template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid, bool is_debug = false) {
  constexpr int num_bytes_per_thread_per_instruction = 16;
  constexpr int num_bytes_per_bf16 = 2;
  constexpr int num_elems_per_instruct = num_bytes_per_thread_per_instruction / num_bytes_per_bf16; // 8
  // constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_elems_per_block = HEIGHT * WIDTH;
  constexpr int num_iters = num_elems_per_block / (TB_SIZE * num_elems_per_instruct);
  // constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
  // For Q:
  // BLOCK_Q, DIM,   TB_SIZE
  // HEIGHT , WIDTH, TB_SIZE
  
  if (isthread0() && is_debug) {
    printf("global_to_shared: HEIGHT=%d, WIDTH=%d, TB_SIZE=%d, num_iters=%d, num_elems_per_instruct=%d, src_stride=%d, \n",
           HEIGHT, WIDTH, TB_SIZE, num_iters, num_elems_per_instruct, src_stride);
  }
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems_per_instruct;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    auto first_val = *src_addr;
    float first_val_in_float = __bfloat162float(first_val);
    if ( iter == 0 && is_debug){
        printf("tid, %d, load row, %d, col, %d, %.2f \n", tid, row, col, first_val_in_float);
    }
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
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

template <typename T, typename... Args>
void launch_kernel(
  T *kernel,
  int num_blocks,
  int block_size,
  int smem_size,
  Args... args) {
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    printf("Launch kenrel with num_blocks: %d, block_size: %d \n", num_blocks, block_size);
  kernel<<<num_blocks, block_size, smem_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}
