# Matrix multiplication - Tensor Cores

Resources:
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html and https://github.com/alexarmbr/matmul-playground
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/inline-ptx-assembly/
- https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch (see `copy_smxx.hpp` and `mma_smxx.hpp`)
- https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/ (Appendix)
- https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cutlass/gemm/threadblock/mma_multistage.h
- https://www.spatters.ca/mma-matmul
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog - TMA
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma

For M = N = K = 4096, BF16 A row-major x B column-major, 5090 @ 400W, compile with CUDA 12.9
- Theoretical limit: 209.5 TFLOPS

TODO: TMA

Kernel name                                            | TFLOPS | % of SOL
-------------------------------------------------------|--------|----------
CuBLAS 12.8 (via PyTorch)                              | 176.89 |   84.44%
Inductor Triton (v3.3.1)                               | 203.81 |   97.29%
v1 (block+warp tiling, vectorized load)                | 147.44 |   70.38%
v2 (`cp.async`)                                        | 157.67 |   75.26%
v3 (pad shared memory)                                 | 172.47 |   82.33%
v4 (swizzle shared memory)                             | 192.68 |   91.97%
v5 (`ldmatrix.x4` for B, optimize address computation) | 200.03 |   95.48%
v6 (2-stage pipelining)                                | 202.05 |   96.44%

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
- Multi-stage pipeline.
- TMA / `cp.async.bulk.tensor`
  - From host side, create `CUtensorMap` using CUDA Driver API (link with `-lcuda`). This type encodes the following information: global memory layout (dims and strides, 1st dim must be contiguous), shared memory layout (dims only, but we can choose a few supported swizzled layouts).
  - `mbarrier` object is used to synchronize TMA transfer.
  - `fence.proxy.async.shared::cta` is required to make changes by CUDA thread visible to **async proxy** (TMA hardware?).
  - **Only 1 thread** is required to issue instructions to TMA.
  - Use `cp_async_bulk_tensor_2d_global_to_shared()` / `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` to initiate TMA load. We can specify global memory offset to select the tile from global memory to load to shared memory.
  - Unit of data is 16-byte (1 row of `ldmatrix` / 4 memory banks, same as `cp.async.cg`). Swizzle layout will permute these data units.
  - `CU_TENSOR_MAP_SWIZZLE_128B`: 128-byte block `uint128_t[8][8]`. Column index is XOR-ed with row index. Note that this row/column does not correspond to shared memory row/column. Inner-most dim of shared memory must be <= 128 bytes. Looking at memory address, it XORs bit4-6 with bit7-9.
  - `CU_TENSOR_MAP_SWIZZLE_64B`: 64-byte block `uint128_t[4][8]`. Inner-most dim of shared memory must be <= 64 bytes. XOR bit4-5 with bit 7-8.
