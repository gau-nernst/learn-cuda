// for mfma intrinsics later
using s16x4 = short __attribute__((__vector_size__(4 * sizeof(short))));
using fp32x4 = float __attribute__((__vector_size__(4 * sizeof(float))));

constexpr int WARP_SIZE = 64;

// this is for MI300X
// https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
// another option is m32n32k8
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;

__device__ __host__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
