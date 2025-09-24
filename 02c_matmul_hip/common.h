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

// NOTE: stride in counts of BF16
// row and col are also in the units of BF16 elements
template <int STRIDE>
__device__
int swizzle(int row, int col) {
  // we are loading a 16x16 BF16 tile, where each thread loads 4 consecutive elements.
  // not sure how MI300X will break this down into 4 "waves" of LDS bank accesses.
  // i was assuming it will load as 4 16x4 tiles, but 8x8 (like ldmatrix) also works.
  constexpr int group_height = 8;
  constexpr int group_width = 8;
  constexpr int num_elems_full_banks = 64;

  if constexpr (STRIDE == group_width)
    return col;

  // how many rows we must go down to repeat the same bank.
  constexpr int rows_same_bank = std::max(num_elems_full_banks / STRIDE, 1);

  const int xor_pattern = (row / rows_same_bank) % (group_height / rows_same_bank);
  return col ^ (xor_pattern * group_width);
}
