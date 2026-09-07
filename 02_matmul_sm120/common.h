#include <iostream>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ ":" << __LINE__ << std::endl;        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; } // https://stackoverflow.com/a/1804686
constexpr int WARP_SIZE = 32;

// NOTE: stride in bytes
// col is in unit of 16-byte word
template <int STRIDE>
__device__
int swizzle(int row, int col) {
  if constexpr (STRIDE > 16)
    col ^= (row % 8) / std::max(128 / STRIDE, 1);
  return row * STRIDE + col * 16;
}

__device__ inline
void ldmatrix_x4(int reg[4], int addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
              : "r"(addr));
}

template <typename T> struct GetType;
template<> struct GetType<nv_bfloat16> {
  using acc = float;
  static constexpr CUtensorMapDataType tmap_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
};
template<> struct GetType<int8_t> {
  using acc = int;
  static constexpr CUtensorMapDataType tmap_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
};

template <typename InType>
void init_tensor_map(
  CUtensorMap *tmap_ptr,
  const InType *gmem_ptr,
  uint64_t gmem_height, uint64_t gmem_width,
  uint32_t smem_height, uint32_t smem_width
) {
  constexpr uint32_t rank = 2;
  uint64_t size[rank]        = {gmem_width, gmem_height};
  uint64_t stride[rank - 1]  = {gmem_width * sizeof(InType)};  // in bytes
  uint32_t box_size[rank]    = {smem_width, smem_height};
  uint32_t elem_stride[rank] = {1, 1};

  const uint32_t smem_stride_B = smem_width * sizeof(InType);
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  if (smem_stride_B == 32)
    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  else if (smem_stride_B == 64)
    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  else if (smem_stride_B == 128)
    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

  auto res = cuTensorMapEncodeTiled(
    tmap_ptr, GetType<InType>::tmap_dtype, rank,
    (void *)gmem_ptr, size, stride,
    box_size, elem_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    swizzle,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  if (res != CUDA_SUCCESS) {
    const char *error_msg_ptr;
    if (cuGetErrorString(res, &error_msg_ptr) != CUDA_SUCCESS)
      error_msg_ptr = "unable to get error string";
    std::cerr << "cuTensorMapEncodeTiled error: " << error_msg_ptr << std::endl;
  }
};

template<typename InType, typename AccType>
__device__ inline
void mma(const int A[4], const int B[2], AccType C[4]) {
  if constexpr (std::is_same_v<InType, nv_bfloat16>)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, " // C
                "{%4, %5, %6, %7}, " // A
                "{%8, %9}, "         // B
                "{%0, %1, %2, %3};"  // C
                : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]));
  if constexpr (std::is_same_v<InType, int8_t>)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, " // C
                "{%4, %5, %6, %7}, " // A
                "{%8, %9}, "         // B
                "{%0, %1, %2, %3};"  // C
                : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]));
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
void tma_2d_g2s(int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
              "[%0], [%1, {%2, %3}], [%4];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr)
              : "memory");
}

__device__ inline
void tma_3d_g2s(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
              "[%0], [%1, {%2, %3, %4}], [%5];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
              : "memory");
}

template <typename T, typename... Args>
void launch_kernel(T *kernel, int num_blocks, int block_size, int smem_size, Args... args) {
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  kernel<<<num_blocks, block_size, smem_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}
