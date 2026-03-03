
Resources:
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog - TMA
- https://github.com/NVIDIA/cutlass/blob/v4.3.5/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py

BF16 A row-major x B column-major, compile with CUDA 13.0.

**5090 @ 400W**: Max 209.5 BF16 TFLOPS, 838 INT8 TFLOPS. Report `TFLOPS (%SOL)`

BF16

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) | 161.42 (77.05%) | 175.03 (83.55%) | 191.29 (91.31%)
Inductor Triton (PyTorch 2.10) | 140.03 (66.84%) | 166.94 (79.69%) | 187.40 (89.45%)
v0 (`cp.async`)                | 148.93 (71.09%) | 172.57 (82.37%) | 166.25 (79.36%)
v1 (TMA)                       | 153.60 (73.32%) | 179.07 (85.48%) | 172.37 (82.28%)

INT8

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.10) | 395.42 (47.19%) | 427.07 (50.96%) | 433.40 (51.72%)
Inductor Triton (PyTorch 2.10) | 259.67 (30.99%) | 281.29 (33.57%) | 290.61 (34.68%)
v0 (`cp.async`)                | 348.98 (41.64%) | 388.23 (46.33%) | 396.32 (47.29%)
v1 (TMA)                       | 365.03 (43.56%) | 406.09 (48.46%) | 415.84 (49.62%)

Note:
- Exceeding/reaching 100% BF16 SOL is not unexpected for 5090, due to nerfed BF16 MMA.
- TODO: we get less TFLOPS from 4096->8192 -> something is wrong with our implementation.

**PRO 6000** (old results, not updated): Max 503.8 TFLOPS. Report `TFLOPS (%SOL)`

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
