
Resources:
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog - TMA
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma

For M = N = K = 4096, BF16 A row-major x B column-major, compile with CUDA 13.0.
- Theoretical limit: 209.5 TFLOPS for 5090, 503.8 for PRO 6000.

Kernel name                    | 5090 @ 400W                | PRO 6000 @ 600W
-------------------------------|----------------------------|---------------------------
CuBLAS 13.0 (via PyTorch 2.10) | 160.54 TFLOPS (76.63% SOL) | 348.62 TFLOPS (69.20% SOL)
Inductor Triton (PyTorch 2.10) | 173.04 TFLOPS (82.60% SOL) | 327.36 TFLOPS (64.98% SOL)
v0 (`cp.async`)                | 175.30 TFLOPS (83.68% SOL) | 326.56 TFLOPS (64.82% SOL)
v1 (TMA)                       | 180.40 TFLOPS (86.11% SOL) | 339.79 TFLOPS (67.45% SOL)

TODO:
- Warp specialization

Learnings
- TMA / `cp.async.bulk.tensor`
  - From host side, create `CUtensorMap` using CUDA Driver API (link with `-lcuda`). This type encodes the following information: global memory layout (dims and strides, 1st dim must be contiguous), shared memory layout (dims only, but we can choose a few supported swizzled layouts).
  - `mbarrier` object is used to synchronize TMA transfer.
  - `fence.proxy.async.shared::cta` is required to make changes by CUDA thread visible to **async proxy** (TMA hardware?).
  - **Only 1 thread** is required to issue instructions to TMA.
  - Use `cp_async_bulk_tensor_2d_global_to_shared()` / `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` to initiate TMA load. We can specify global memory offset to select the tile from global memory to load to shared memory.
  - Unit of data is 16-byte (1 row of `ldmatrix` / 4 memory banks, same as `cp.async.cg`). Swizzle layout will permute these data units.
  - `CU_TENSOR_MAP_SWIZZLE_128B`: 128-byte block `uint128_t[8][8]`. Column index is XOR-ed with row index. Note that this row/column does not correspond to shared memory row/column. Inner-most dim of shared memory must be <= 128 bytes. Looking at memory address, it XORs bit4-6 with bit7-9.
  - `CU_TENSOR_MAP_SWIZZLE_64B`: 64-byte block `uint128_t[4][8]`. Inner-most dim of shared memory must be <= 64 bytes. XOR bit4-5 with bit 7-8.
