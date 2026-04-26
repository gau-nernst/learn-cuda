#pragma once

constexpr int WARP_SIZE = 32;

__device__ inline
void bf16x2_to_fp32x2(float *out, int x) {
  asm volatile("shl.b32 %0, %2, 16;\n"          // low 16-bit
               "and.b32 %1, %2, 0xFFFF0000;\n"  // high 16-bit
               : "=f"(out[0]), "=f"(out[1])
               : "r"(x));
}

__device__ inline
int fp32x2_to_bf16x2(float a, float b) {
  int tmp;
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(tmp) : "f"(a), "f"(b));
  return tmp;
}

template <int VEC>
__device__ inline
void ldg_b32(void *data_, const void *ptr) {
  int *data = reinterpret_cast<int *>(data_);
  if constexpr (VEC == 1)
    asm volatile("ld.global.b32 %0, [%1];"
                : "=r"(data[0])
                : "l"(ptr));
  if constexpr (VEC == 2)
    asm volatile("ld.global.v2.b32 {%0, %1}, [%2];"
                : "=r"(data[0]), "=r"(data[1])
                : "l"(ptr));
  if constexpr (VEC == 4)
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3])
                : "l"(ptr));
  if constexpr (VEC == 8)
    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
                  "=r"(data[4]), "=r"(data[5]), "=r"(data[6]), "=r"(data[7])
                : "l"(ptr));
}

template <int VEC>
__device__ inline
void ldg_b32_fast(void *data_, const void *ptr) {
  int *data = reinterpret_cast<int *>(data_);
  if constexpr (VEC == 1)
    asm volatile("ld.global.relaxed.cta.L1::no_allocate.b32 %0, [%1];"
                : "=r"(data[0])
                : "l"(ptr));
  if constexpr (VEC == 2)
    asm volatile("ld.global.relaxed.cta.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                : "=r"(data[0]), "=r"(data[1])
                : "l"(ptr));
  if constexpr (VEC == 4)
    asm volatile("ld.global.relaxed.cta.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3])
                : "l"(ptr));
  if constexpr (VEC == 8)
    asm volatile("ld.global.relaxed.cta.L1::no_allocate.L2::evict_first.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
                  "=r"(data[4]), "=r"(data[5]), "=r"(data[6]), "=r"(data[7])
                : "l"(ptr));
}

template <int VEC>
__device__ inline
void stg_b32(void *ptr, const void *data_) {
  const int *data = reinterpret_cast<const int *>(data_);
  if constexpr (VEC == 1)
    asm volatile("st.global.b32 [%0], %1;"
                :: "l"(ptr),
                "r"(data[0]));
  if constexpr (VEC == 2)
    asm volatile("st.global.v2.b32 [%0], {%1, %2};"
                :: "l"(ptr),
                "r"(data[0]), "r"(data[1]));
  if constexpr (VEC == 4)
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                :: "l"(ptr),
                "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]));
  if constexpr (VEC == 8)
    asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"(ptr),
                "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]),
                "r"(data[4]), "r"(data[5]), "r"(data[6]), "r"(data[7]));
}

template <int VEC>
__device__ inline
void stg_b32_fast(void *ptr, const void *data_) {
  const int *data = reinterpret_cast<const int *>(data_);
  if constexpr (VEC == 1)
    asm volatile("st.global.relaxed.cta.L1::no_allocate.b32 [%0], %1;"
                :: "l"(ptr),
                "r"(data[0]));
  if constexpr (VEC == 2)
    asm volatile("st.global.relaxed.cta.L1::no_allocate.v2.b32 [%0], {%1, %2};"
                :: "l"(ptr),
                "r"(data[0]), "r"(data[1]));
  if constexpr (VEC == 4)
    asm volatile("st.global.relaxed.cta.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};"
                :: "l"(ptr),
                "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]));
  if constexpr (VEC == 8)
    asm volatile("st.global.relaxed.cta.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"(ptr),
                "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]),
                "r"(data[4]), "r"(data[5]), "r"(data[6]), "r"(data[7]));
}

template <int VEC>
__device__ inline
void lds_b32(void *data_, int addr) {
  int *data = reinterpret_cast<int *>(data_);
  if constexpr (VEC == 1)
    asm volatile("ld.shared.b32 %0, [%1];"
                : "=r"(data[0])
                : "r"(addr));
  if constexpr (VEC == 2)
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                : "=r"(data[0]), "=r"(data[1])
                : "r"(addr));
  if constexpr (VEC == 4)
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3])
                : "r"(addr));
}

template <int VEC>
__device__ inline
void sts_b32(int addr, const void *data_) {
  const int *data = reinterpret_cast<const int *>(data_);
  if constexpr (VEC == 1)
    asm volatile("st.shared.b32 [%0], %1;"
                :: "r"(addr),
                "r"(data[0]));
  if constexpr (VEC == 2)
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                :: "r"(addr),
                "r"(data[0]), "r"(data[1]));
  if constexpr (VEC == 4)
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                :: "r"(addr),
                "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]));
}

__device__ inline
int atomic_add_release_gpu(int *ptr, int val) {
  int tmp;
  asm volatile("atom.release.gpu.global.add.s32 %0, [%1], %2;"
              : "=r"(tmp)
              : "l"(ptr), "r"(val));
  return tmp;
}

__device__ inline
int load_acquire_gpu(int *ptr) {
  int tmp;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(tmp) : "l"(ptr));
  return tmp;
}

__device__ inline
void fma_f32x2(float *d, const float *a, const float *b, const float *c) {
  asm volatile(
    "{\n"
    ".reg .b64 d, a, b, c;\n"
    "mov.b64 a, {%2, %3};\n"
    "mov.b64 b, {%4, %5};\n"
    "mov.b64 c, {%6, %7};\n"
    "fma.rn.f32x2 d, a, b, c;\n"
    "mov.b64 {%0, %1}, d;\n"
    "}"
    : "=f"(d[0]), "=f"(d[1])
    : "f"(a[0]), "f"(a[1]), "f"(b[0]), "f"(b[1]), "f"(c[0]), "f"(c[1])
  );
}
