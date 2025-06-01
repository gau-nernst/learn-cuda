#include <cstdint>
#include <cuda_bf16.h>

__device__ uint32_t cvta_shared(void const *ptr) { return static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); }

template <int num> __device__ void ldmatrix(uint32_t *reg, uint32_t addr);
template <> __device__ void ldmatrix<1>(uint32_t reg[1], uint32_t addr) {
  // "=r" is output, "r" is input
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(reg[0]) : "r"(addr));
}
template <> __device__ void ldmatrix<2>(uint32_t *reg, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
}
template <> __device__ void ldmatrix<4>(uint32_t *reg, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
               : "r"(addr));
}

template <int MMA_K, typename T> __device__ void mma(uint32_t *A, uint32_t *B, float *acc);
template <> __device__ void mma<8, half>(uint32_t *A, uint32_t *B, float *acc) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, " // D
               "{%4, %5}, "         // A
               "{%6}, "             // B
               "{%7, %8, %9, %10};" // C
               : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
               : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}
template <> __device__ void mma<8, nv_bfloat16>(uint32_t *A, uint32_t *B, float *acc) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, " // D
               "{%4, %5}, "         // A
               "{%6}, "             // B
               "{%7, %8, %9, %10};" // C
               : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
               : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}
template <> __device__ void mma<16, half>(uint32_t *A, uint32_t *B, float *acc) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "    // D
               "{%4, %5, %6, %7}, "    // A
               "{%8, %9}, "            // B
               "{%10, %11, %12, %13};" // C
               : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(acc[0]), "f"(acc[1]),
                 "f"(acc[2]), "f"(acc[3]));
}
template <> __device__ void mma<16, nv_bfloat16>(uint32_t *A, uint32_t *B, float *acc) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, "    // D
               "{%4, %5, %6, %7}, "    // A
               "{%8, %9}, "            // B
               "{%10, %11, %12, %13};" // C
               : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(acc[0]), "f"(acc[1]),
                 "f"(acc[2]), "f"(acc[3]));
}
