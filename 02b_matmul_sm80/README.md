# Matrix multiplication - Tensor Cores

Resources:
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html and https://github.com/alexarmbr/matmul-playground
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/inline-ptx-assembly/
- https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch (see `copy_smxx.hpp` and `mma_smxx.hpp`)
- https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/ (Appendix)
- https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cutlass/gemm/threadblock/mma_multistage.h
- https://www.spatters.ca/mma-matmul

BF16 A row-major x B column-major, compile with CUDA 13.0.

**5090 @ 400W**: Fixed M=N=K=4096. Max 209.5 TFLOPS.

Kernel name                                            | TFLOPS | %SOL
-------------------------------------------------------|--------|-------
CuBLAS 13.0 (via PyTorch 2.10)                         | 159.50 | 76.14%
Inductor Triton (PyTorch 2.10)                         | 171.56 | 81.89%
v1 (block+warp tiling, vectorized load)                | 128.07 | 61.13%
v2 (`cp.async`)                                        | 139.59 | 66.63%
v3 (pad shared memory)                                 | 152.18 | 72.64%
v4 (swizzle shared memory)                             | 165.18 | 78.85%
v5 (`ldmatrix.x4` for B, optimize address computation) | 170.76 | 81.51%
v6 (2-stage pipelining)                                | 169.33 | 80.82%
v7 (better swizzling logic, unroll prefetch stages)    | 174.54 | 83.31%

**5090 @ 400W**: Varying problem shapes. Max 209.5 TFLOPS. Report `TFLOPS (%SOL)`

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) | 140.10 (66.87%) | 159.50 (76.14%) | 200.32 (95.62%)
Inductor Triton (PyTorch 2.10) | 137.13 (65.46%) | 171.56 (81.89%) | 200.47 (95.69%)
v7                             | 133.15 (63.56%) | 174.54 (83.31%) | 162.81 (77.71%)

**Modal A100 80GB PCIe**: Varying problem shapes. Max 312 TFLOPS. Report `TFLOPS (%SOL)`

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) | 151.15 (48.44%) | 227.49 (72.91%) | 228.16 (73.13%)
Inductor Triton (PyTorch 2.10) | 159.78 (51.21%) | 205.86 (65.98%) | 211.93 (67.93%)
v7                             | 174.76 (56.01%) | 166.73 (53.44%) | 141.76 (45.44%)

Lessons learned:
- Inline PTX: instruction, outputs, inputs, constraints
- `ldmatrix`: a warp loads 8x8 tiles of 16-bit words from shared memory to registers. Each thread holds 2 elements (there are 32 threads in a warp). Which thread holds which element is conveniently correct for `mma` instructions later. We can load 1x, 2x, or 4x of 8x8 16-bit tiles.
  - More generically, it loads 8 rows of 8 consecutive 16-bit words: each row can be anywhere in shared memory (given alignment constraint).
  - To use `ldmatrix`, we need to convert pointer in generic address to shared address required by PTX using `ctva` instruction (convert address). Most, if not all, PTX instructions expect shared memory address to be 32-bit (instead of 64-bit).
- `mma`: typical shape for FP16/BF16 is `m16n8k8` and `m16n8k16`. Each thread in a warp must hold specific elements in the tile, which can be done using `ldmatrix`. To load 16x8 tile, we use `ldmatrix.x2`. Similarly, for 16x16 tile, we use `ldmatrix.x4`. Refer to PTX docs on layout of 8x8 tiles.
- Accumulate result is held in register memory (across threads in a warp). We can write the results from register directly to global memory. Again, each thread hold specific elements of the output.
- `cp.async`
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
- The better way is to use **swizzled layout** for shared memory. The idea is that we spread 8 rows of the 8x8 tile across 32 banks. To do so, we **permute** the position of each row within the block tile.
  - Look at bit pattern of shared memory address. Bit0-1 are within a bank. Bit2-6 determines bank index (32 = 2^5).
  - Due to 16-byte alignment constraint of `ldmatrix`, bit0-3 of row address are always zeros.
  - -> we only need to permute bit4-6 of row address (3 bits).
  - For `BLOCK_K=64` (stride = 128 bytes = 2^7), all row addresses have the same bit0-6 -> 32-way bank conflict.
  - We XOR bit4-6 with row index to permute row positions in shared memory. XOR ensures that there is one-to-one mapping between input and output.
  - For `BLOCK_K=32` (stride = 64 bytes = 2^6), all row addresses have the same bit0-5 -> 16-way bank conflict.
  - We XOR bit4-5 with bit1-2 of row index. Note that we don't use bit0 of row index since that bit changes bit6 of row address.
  - Note that row index is also encoded in pre-permuted address: it starts at bit-log2(stride), and spans 3 bits.
- Updated swizzling instruction. Key idea: permute (i.e. XOR) column indices with row indices
  - Think in terms of 16-byte units. This only changes column indices, row indices are not affected.
  - Instead of looking at `ldmatrix` as loading an 8x8 tile of 16-bit, think of it as loading 8 16-byte words. So the main problem is to distribute these 8 16-byte words to 32 different banks.
  - Before any swizzling, row stride (`BLOCK_K * sizeof(nv_bfloat16)`) determines which banks the next row falls into. For example, with stride = 128 bytes = 32 banks, 8 16-byte words, each being 128 bytes apart, map to the same bank. For stride = 64 bytes = 16 banks, word0, 2, 4, 6 map to the same bank, word1, 3, 5, 7 map to the same bank.
  - Hence, how we do swizzling is dependent on the row stride. For stride >= 128 bytes, all 8 words of `ldmatrix` maps to the same bank, so we can simply use their **row indices** to swizzle / XOR the column indices. To get the "relative" row index, simply take modulo 8 (or 3 LSBs) -> `new_col = col ^ (row % 8)`.
  - For stride = 64 bytes, every 2 words of `ldmatrix` maps to the same bank. Hence, we use the relative row indices within each 4-word group for swizzling -> `new_col = col ^ ((row % 8) / 2)`.
- Multi-stage pipeline.
- Threadblock swizzling only matters for large problem shapes.
