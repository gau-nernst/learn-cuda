
Resources:
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog - TMA
- https://github.com/NVIDIA/cutlass/blob/v4.3.5/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py

BF16 A row-major x B column-major, compile with CUDA 13.0.

**5090 @ 400W**: Max 209.5 BF16 TFLOPS, 838 INT8 TFLOPS. Report `TFLOPS (%SOL)`

BF16

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.11) | 164.42 (78.48%) | 172.61 (82.39%) | 166.78 (79.61%)
Inductor Triton (PyTorch 2.11) | 141.45 (67.52%) | 174.40 (83.25%) | 184.88 (88.25%)
v0 (`cp.async`)                | 158.24 (75.53%) | 164.01 (78.29%) | 192.29 (91.78%)
v1 (TMA)                       | 164.27 (78.41%) | 171.24 (81.74%) | 200.29 (95.60%)
v2 (warp specialization)       | 164.41 (78.48%) | 173.07 (82.61%) | 201.03 (95.95%)

INT8

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.11) | 396.99 (47.37%) | 423.61 (50.55%) | 424.18 (50.62%)
Inductor Triton (PyTorch 2.11) | 423.78 (50.57%) | 447.90 (53.45%) | 474.87 (56.67%)
v0 (`cp.async`)                | 401.87 (47.96%) | 425.00 (50.72%) | 448.42 (53.51%)
v1 (TMA)                       | 440.57 (52.57%) | 463.30 (55.29%) | 485.84 (57.98%)
v2 (warp specialization)       | 439.72 (52.47%) | 464.89 (55.48%) | 485.83 (57.97%)

Note:
- Exceeding/reaching 100% BF16 SOL is not unexpected for 5090, due to nerfed BF16 MMA.
- Some combinations can be better if we autotune.
- For CuBLAS, the result is very different at 600W.

TODO:
- setmaxnreg
- persistent kernel w/ ping-pong

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
