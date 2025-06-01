# Matrix multiplication - Tensor Cores

Resources:
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html and https://github.com/alexarmbr/matmul-playground
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/inline-ptx-assembly/
- https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch (see `copy_smxx.hpp` and `mma_smxx.hpp`)

For M = N = K = 4096, BF16 A row-major x B column-major, 5090 @ 400W, compile with CUDA 12.9, `-O3 --use_fast_math`
- Theoretical limit: 209.5 TFLOPS

Kernel name                            |  TFLOPS | % of SOL
---------------------------------------|---------|----------
CuBLAS 12.8 (via PyTorch)              |  177.34 |   84.65%
v1 (block+warp tiling, `mma.m16n8k16`) |  144.15 |   69.28%

Lessons learned:
- Inline PTX: instruction, outputs, inputs, constraints
- `ldmatrix`: a warp loads 8x8 tiles of 16-bit from shared memory to registers. Each thread holds 2 elements (there are 32 threads in a warp). Which thread holds which element is conveniently correct for `mma` instructions later. We can load 1x, 2x, or 4x of 8x8 16-bit tiles.
  - To be more generic, it loads 8 rows of 8 consecutive 16-bit elements (each 8-element row can be anywhere in shared memory, given 16-byte alignment constraint). To be even more generic, the instruction collectively loads 8x 16-byte words.
  - To use `ldmatrix`, we need to convert pointer in generic address to shared address required by PTX using `ctva` instruction (convert address).
- `mma`: typical shape for FP16/BF16 is `m16n8k8` and `m16n8k16`. Each thread in a warp must hold specific elements in the tile, which can be done using `ldmatrix`. To load 16x8 tile, we use `ldmatrix.x2`. Similarly, for 16x16 tile, we use `ldmatrix.x4`.
- Accumulate result is held in register memory (across threads in a warp). We can write the results from register directly to global memory. Again, each thread hold specific elements of the output.
- When we use normal layout for shared memory, there will be bank conflicts when using `ldmatrix`. There are 32 banks, each holds 4 bytes. Each row of `ldmatrix` tile resides in 4 banks (16 bytes). If our shared memory has width = 8x 16-bit elements (16 bytes), there would be no bank conflicts. However, usually we use a larger width to take advantage of vectorized global memory read.
- One classic way to fix this is to pad shared memory, which wastes resources (the padded shared memory is not used). The amount of padding must be 8 element to ensure 16-byte alignment required by `ldmatrix`.
- The better way is to use **swizzled layout** for shared memory. The idea is that we spread 8 rows of each 8x8 `ldmatrix` tile across 32 banks. To use the same amount of shared memory, we need to **permute** the order/position of `ldmatrix` rows within the block tile.
  - For layout conversion, there is always a mapping of indices between those in the "special" layout (logical view) and those in linear layout (physical view). For swizzled layout, typically the **swizzle functor** only consists of bit manipulation operations.
  - Each 8x-16bit row must stay together. In other words, we can use 16-byte as our unit of swizzling -> taking the word size as 16-byte. Each 16-byte word resides in 4 consecutive memory banks. We can call this "4-bank group". There are 8 groups in total (32 banks). We need to distribute 8 rows of `ldmatrix` tile (or simply 8x 16-byte words) among the 8 "4-bank groups".
  - Looking at the index of the 16-byte elements, bits 0-2 determine the "4-bank group" (physical view). We also look that the row index. Which bits determine row index depends on the row stride (or width) of shared memory. **Bank conflict happens when row index changes but bank-group index does not change**.
  - We XOR bank-group index with row-index -> bank-group index is shuffled by row-index. We should do this only for non-overlapping bits.
    - Row stride is 1. Bits 0-2 determine row index. Bits fully overlap. No bank conflict
    - Row stride is 2. Bits 1-3 determine row index. Bits 1-2 overlap (when we change bits 1-2, bank-group index also changes, while if we change bit 3, bank-group index does not change). XOR bit 0 with bit 3 -> when bit 3 changes, bank-group index also changes.
    - Row stride is 4. Bits 2-4 determine row index. Bit 2 overlaps. XOR bits 0-1 with bits 3-4.
    - Row stride is 8. Bits 3-5 determine row index. No overlap. XOR bits 0-2 with bits 3-5.
    - Row stride is 16. Bits 4-6 determine row index. No overlap. XOR bits 0-2 with bits 4-6.
  - Number of bits to XOR = min(log2(row stride), 3) (aka BBits in CUTE). XOR bit [0,num_bits-1] with [log2(row stride), log2(row stride) + num_bits - 1]. log2(row stride) is SShift in CUTE.
