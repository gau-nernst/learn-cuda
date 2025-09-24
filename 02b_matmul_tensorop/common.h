#include <cstdint>
#include <iostream>
#include <cuda_bf16.h>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ ":" << __LINE__ << std::endl;        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; } // https://stackoverflow.com/a/1804686
constexpr int WARP_SIZE = 32;

// convert generic address (C++ address, 64-bit) to shared state space address (32-bit)
// all PTX instructions expect share memory address to be in shared state space (not 100%)
__device__ inline
uint32_t cvta_shared(const void *ptr) { return static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); }

__device__ inline
void ldmatrix_x2(uint32_t reg[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
              : "=r"(reg[0]), "=r"(reg[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t reg[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
              : "r"(addr));
}

__device__ inline
void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, "    // D
               "{%4, %5, %6, %7}, "    // A
               "{%8, %9}, "            // B
               "{%10, %11, %12, %13};" // C
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy
__device__ inline
void cp_async(uint32_t dst, const void *src) {
  // .ca means cache to L1 and L2. .cg means cache to L2 only.
  // .cg only accepts cp-size=16
  // .ca results in significantly slower kernel, probably because it uses up L1 resources
  // + additional copy, which is unnecessary, since we already manually cache it in shared memory.
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
};

__device__ inline
void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); };

template <int N>
__device__ inline
void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); };

__device__ inline
void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); };

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / std::max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

// NOTE: stride in element T
// row and col are also in the units of element size
template <int STRIDE, typename T>
__device__
int swizzle2(int row, int col) {
  // 16 is the minimum (ldmatrix loads rows of 16 byte each)
  // when stride is 16, 8 consecutive rows already distribute evenly across 32 banks.
  constexpr int STRIDE_BYTES = STRIDE * sizeof(T);
  if constexpr (STRIDE_BYTES == 16)
    return col;

  // ldmatrix loads a row of 16 bytes. 8 for BF16
  constexpr int group_width = 16 / sizeof(T);

  // how many rows we must go down to repeat the same bank.
  // 32 * 4 is 32 banks of 4-byte each.
  constexpr int rows_same_bank = std::max((32 * 4) / STRIDE_BYTES, 1);

  const int col_group = col / group_width;
  const int xor_pattern = (row / rows_same_bank) % (8 / rows_same_bank);  // 8 here is 8 rows in ldmatrix

  const int new_col_group = col_group ^ xor_pattern;
  return new_col_group * group_width;
}

template <typename T, typename... Args>
void launch_kernel(T *kernel, int num_blocks, int block_size, int shm_size, Args... args) {
  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<num_blocks, block_size, shm_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}
