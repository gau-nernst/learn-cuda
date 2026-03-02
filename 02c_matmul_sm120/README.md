
Resources:
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog - TMA
- https://github.com/NVIDIA/cutlass/blob/v4.3.5/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py

BF16 A row-major x B column-major, compile with CUDA 13.0.

**5090 @ 400W**: Max 209.5 BF16 TFLOPS, 838 INT8 TFLOPS. Driver 580.126.20. Report `TFLOPS (%SOL)`

BF16

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) | 167.77 (80.08%) | 175.46 (83.75%) | 209.92 (100.20%)
Inductor Triton (PyTorch 2.10) | 174.71 (83.39%) | 198.55 (94.77%) | 209.43 (99.97%)
v0 (`cp.async`)                | 161.32 (77.00%) | 200.32 (95.62%) | 195.23 (93.19%)
v1 (TMA)                       | 167.35 (79.88%) | 206.81 (98.71%) | 202.21 (96.52%)

INT8

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) | 470.53 (56.15%) | 553.76 (66.08%) | 503.31 (60.06%)
Inductor Triton (PyTorch 2.10) | 310.51 (37.05%) | 368.70 (44.00%) | 340.42 (40.62%)
v0 (`cp.async`)                | 399.46 (47.67%) | 486.68 (58.08%) | 483.14 (57.65%)
v1 (TMA)                       | 415.86 (49.62%) | 507.68 (60.58%) | 497.87 (59.41%)

Note:
- Exceeding 100% BF16 SOL is not unexpected on 5090, due to nerfed BF16 MMA.
- Looks like driver version can heavily affect benchmark results...
- TODO: we get less TFLOPS from 4096->8192 -> something is wrong with our implementation.

**PRO 6000**: Max 503.8 TFLOPS. Report `TFLOPS (%SOL)`

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) | 199.36 (39.57%) | 370.80 (73.60%) | 431.72 (85.69%)
Inductor Triton (PyTorch 2.10) | 232.31 (46.11%) | 349.55 (69.38%) | 405.64 (80.52%)
v0 (`cp.async`)                | 220.12 (43.69%) | 334.50 (66.40%) | 405.05 (80.40%)
v1 (TMA)                       | 226.62 (44.98%) | 362.75 (72.00%) | 425.84 (84.53%)

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
