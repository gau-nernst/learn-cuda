#include <cstdint>
#include <cudaTypedefs.h>
#include <torch/library.h>

constexpr int WARP_SIZE = 32;

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template <typename T>
__device__ inline
T warp_uniform(T x) { return __shfl_sync(0xFFFF'FFFF, x, 0); }

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__ inline
uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

__device__ inline
void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__ inline
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__device__ inline
void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ inline
void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
}

__device__ inline
void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr) : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
  // when CTA_GROUP=1, we can use .shared::cta instead.
  // but .shared::cluster doesn't seem to be slower, so always use it unconditionally here.
  // .cta_group::2 allows mbar_addr and dst to be in different CTA's smem.
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
              "[%0], [%1, {%2, %3, %4}], [%5];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "n"(CTA_GROUP)
              : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline
void tcgen05_alloc(int smem_addr, int size) {
  asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
              :: "r"(smem_addr), "r"(size), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ inline
void tcgen05_dealloc(int taddr, int size) {
  asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
              :: "r"(taddr), "r"(size), "n"(CTA_GROUP));
}

// https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cute/arch/mma_sm100_umma.hpp#L86
template <int CTA_GROUP = 1>
__device__ inline
void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, int enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"  // predicate register enable-input-d
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
    "}"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d), "n"(CTA_GROUP)
  );
}

template <int CTA_GROUP = 1>
__device__ inline
void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.b64 [%0], %1;"
              :: "r"(mbar_addr), "n"(CTA_GROUP) : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline
void tcgen05_commit_mcast(int mbar_addr, int16_t cta_mask) {
  asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
              :: "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
}

__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; };

inline
void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *msg;
  if (cuGetErrorString(err, &msg) != CUDA_SUCCESS)
    msg = "unable to get error string";
  TORCH_CHECK(false, msg);
}

inline
void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) return;
  TORCH_CHECK(false, cudaGetErrorString(err));
}

inline
void init_tmap_2d_simple(
  CUtensorMap *tmap,
  const nv_bfloat16 *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width,
  CUtensorMapSwizzle swizzle
) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank]       = {global_width, global_height};
  uint64_t globalStrides[rank-1] = {global_width * sizeof(nv_bfloat16)};  // in bytes
  uint32_t boxDim[rank]          = {shared_width, shared_height};
  uint32_t elementStrides[rank]  = {1, 1};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    swizzle,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  check_cu(err);
}
