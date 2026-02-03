
Resources:
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog - TMA
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma

BF16 A row-major x B column-major, compile with CUDA 13.0.
- Theoretical limit: 209.5 TFLOPS for 5090, 503.8 for PRO 6000.

**5090**: Max 209.5 TFLOPS. Report `TFLOPS (%SOL)`

Kernel name                    | 1024            | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) |  81.44 (38.87%) | 140.14 (66.89%) | 160.54 (76.63%) | 202.88 (96.84%)
Inductor Triton (PyTorch 2.10) | 104.86 (50.05%) | 139.41 (66.54%) | 173.04 (82.60%) | 201.49 (96.18%)
v0 (`cp.async`)                | 103.09 (49.21%) | 133.15 (63.56%) | 175.30 (83.68%) | 167.22 (79.82%)
v1 (TMA)                       | 104.86 (50.05%) | 137.52 (65.64%) | 180.40 (86.11%) | 172.66 (82.42%)

**PRO 6000**: Max 503.8 TFLOPS. Report `TFLOPS (%SOL)`

Kernel name                    | 1024            | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) |  76.61 (15.21%) | 199.36 (39.57%) | 353.23 (70.11%) | 431.72 (85.69%)
Inductor Triton (PyTorch 2.10) |  87.50 (17.37%) | 232.31 (46.11%) | 331.99 (65.90%) | 405.64 (80.52%)
v0 (`cp.async`)                |  68.41 (13.58%) | 215.09 (42.69%) | 329.52 (65.41%) | 400.20 (79.44%)
v1 (TMA)                       |  69.91 (13.88%) | 220.75 (43.82%) | 360.80 (71.62%) | 420.80 (83.53%)

TODO:
- Warp specialization

Learnings
- TMA / `cp.async.bulk.tensor`
  - From host side, create `CUtensorMap` using CUDA Driver API (link with `-lcuda`). This type encodes the following information: global memory layout (dims and strides, 1st dim must be contiguous), shared memory layout (dims only, but we can choose a few supported swizzled layouts).
  - `mbarrier` object is used to synchronize TMA transfer.
  - `fence.proxy.async.shared::cta` is required to make changes by CUDA thread visible to **async proxy** (TMA hardware?).
  - **Only 1 thread** is required to issue instructions to TMA.
  - Use `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` to initiate TMA load. We can specify global memory offset to select the tile from global memory to load to shared memory.
  - Unit of data is 16-byte (1 row of `ldmatrix` / 4 memory banks, same as `cp.async.cg`). Swizzle layout will permute these data units.
  - `CU_TENSOR_MAP_SWIZZLE_128B`: 128-byte block `uint128_t[8][8]`. Column index is XOR-ed with row index. Note that this row/column does not correspond to shared memory row/column. Inner-most dim of shared memory must be <= 128 bytes. Looking at memory address, it XORs bit4-6 with bit7-9.
  - `CU_TENSOR_MAP_SWIZZLE_64B`: 64-byte block `uint128_t[4][8]`. Inner-most dim of shared memory must be <= 64 bytes. XOR bit4-5 with bit 7-8.
