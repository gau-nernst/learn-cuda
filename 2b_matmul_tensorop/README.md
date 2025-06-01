# Matrix multiplication - Tensor Cores

Resources:
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html and https://github.com/alexarmbr/matmul-playground
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/inline-ptx-assembly/
- https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch (see `copy_smxx.hpp` and `mma_smxx.hpp`)

For M = N = K = 4096, BF16 A row-major x B column-major, 5090 @ 400W, compile with CUDA 12.9, `-O3 --use_fast_math`
- Theoretical limit: 209.5 TFLOPS

Kernel name                             | TFLOPS | % of SOL
----------------------------------------|--------|----------
CuBLAS 12.8 (via PyTorch)               | 177.34 |   84.65%
v1 (block+warp tiling, vectorized load) | 144.15 |   69.28%
v2 (`cp.async`)                         | 163.35 |   77.97%
v3 (pad shared memory)                  | 176.31 |   84.16%

Lessons learned:
- Inline PTX: instruction, outputs, inputs, constraints
- `ldmatrix`: a warp loads 8x8 tiles of 16-bit words from shared memory to registers. Each thread holds 2 elements (there are 32 threads in a warp). Which thread holds which element is conveniently correct for `mma` instructions later. We can load 1x, 2x, or 4x of 8x8 16-bit tiles.
  - More generically, it loads 8 rows of 8 consecutive 16-bit words: each row can be anywhere in shared memory (given alignment constraint).
  - To use `ldmatrix`, we need to convert pointer in generic address to shared address required by PTX using `ctva` instruction (convert address). Most, if not all, PTX instructions expect shared memory address to be 32-bit (instead of 64-bit).
- `mma`: typical shape for FP16/BF16 is `m16n8k8` and `m16n8k16`. Each thread in a warp must hold specific elements in the tile, which can be done using `ldmatrix`. To load 16x8 tile, we use `ldmatrix.x2`. Similarly, for 16x16 tile, we use `ldmatrix.x4`. Refer to PTX docs on layout of 8x8 tiles.
- Accumulate result is held in register memory (across threads in a warp). We can write the results from register directly to global memory. Again, each thread hold specific elements of the output.
- When we use normal layout for shared memory, there will be bank conflicts when using `ldmatrix`.
  - Shared memory is backed by 32 banks. Consecutive 32-bit words (4 bytes) reside in consecutive banks.
  - `ldmatrix` loads one 8x8 tile at a time (according to @firadeoclus from GPU-MODE Discord). It means that for `ldmatrix.x2`, it will load the 1st 8x8 tile then the 2nd 8x8. It also means our analysis is simpler, since we only need to consider a 8x8 tile.
  - Consider our shared memory layout be `(BLOCK_M,BLOCK_K) = (128,32)`. `BLOCK_K` is the row stride (offset to the next row) = 16 banks.
  - Each row of `ldmatrix` tile spans over 4 banks (8x16-bit). row0 spans bank0-3. row1 spans bank16-20. row2 spans bank0-3 -> 16-way bank conflict.
  - This is worse with larger `BLOCK_K` -> with `BLOCK_K>=64`, we get 32-way bank conflict.
- One classic way to fix this is to pad shared memory.
  - Pad 8 elements (4 banks) to `BLOCK_K` dim -> stride is `BLOCK_K + 8 = 40`, or 20 banks. We can only pad 8 elements due to `ldmatrix` alignment constraint.
  - row0 spans bank0-3. row1 spans bank20-23. row2 spans bank8-11. row3 spans bank28-31.
  - row4 spans bank16-19. row5 spans bank4-7. row6 spans bank24-27. row7 spans bank12-15.
  - -> no bank conflicts! However, this wastes shared memory -> reduce available L1 cache.
  - For `BLOCK_K>=64`, we also get no bank conflict with 8 elements (16-byte) padding.
  - Note that we also want `BLOCK_K>=64` to utilize 128-byte cache line.
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
