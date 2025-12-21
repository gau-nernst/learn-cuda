# Blackwell matmul with tcgen05

Resources:
- https://www.modular.com/matrix-multiplication-on-blackwell
- https://github.com/deepseek-ai/DeepGEMM
- https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog

Modal B200, BF16 matmul, A/B K-major, M=N=K=4096

Kernel name                          | TFLOPS
-------------------------------------|--------
CuBLAS (PyTorch 2.9.1 + CUDA 13)     | 1506.74
v0 (sm80)                            |  378.08
v1a (basic tcgen05 + 2D 16B TMA)     |  254.62
v1b (3D 16B TMA)                     |  252.81
v2a (2D 128B TMA)                    |  681.20
v2b (3D 128B TMA)                    |  695.43
v3 (pipelining)                      |  939.61
v4 (warp specialization)             | 1208.83
v5 (2-SM MMA)                        | 1302.29
v6 (persistent w/ static scheduling) | 1475.93

```bash
uv pip install modal
modal setup

# run benchmarking to reproduce the table above
modal run main.py --action benchmark

# run intra-kernel profiling
# go to the code to select which kernel to be profiled
modal run main.py --action profile
```

Notes:

**`tcgen05`**
- after thread sync: fence, tcgen05 ops after this can't run before this (acquire)
- MMA shape:
  - K is always 32-byte -> K=16 for FP16/BF16, K=32 for FP8/INT8, K=64 for FP4
  - M: 1 CTA - 64 or 128; 2 CTAs - 128 or 256
  - N: multiple of 8, up to 256
- Instruction descriptor: 32 bit value
- Shared memory descriptor: 64 bit value describing A and B in shared memory
- `tcgen05.ld`: tmem->rmem, used in epilogue e.g. cast FP32 to BF16. A warp can only access 32 lanes -> we need 4 warps to read all lanes of tensor memory.

**Tensor memory**
- Structure: lanes = rows (max 128), columns (max 512), each 32-bit
- Tensor memory allocation: `tcgen05.alloc/dealloc`, address of tensor memory is written to smem, specify number of columns -> size is always 128xncols (32-bit). Must deallocate before kernel exit
- memory access via `tcgen05.ld` (tmem->rmem), `.st` (rmem->tmem) and `.cp` (smem->tmem)

**mbarrier**
- Reside in smem, 64-bit, can be used in multicast sync. Create multiple barriers to sync multiple stages and multiple producer-consumer pairs.
- Initialize with `mbarrier.init` at start of kernel (used with `elect.sync`), specifying **expected arrival count**. We can invalidate it with `mabbrier.inval`.
- Keeps track of **arrival count** (how many threads have "arrived") and tx-count (how many bytes are transferred).
- Arrival count
  - Expected arrival count is set at init.
  - arrive-on: decrement arrival count e.g. `mbarrier.arrive`
- tx-count
  - expect-tx: increment tx-count e.g. `mbarrier.arrive.expect_tx` -> increment tx-count, THEN decrement arrival count.
  - complete-tx: decrement tx-count e.g. `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` -> decrement tx-count.
- Current phase completes when (1) arrival count = 0, AND (2) tx-count = 0
  - when this happens, automatically move on to the next phase. reinint arrival count -> current phase is always incomplete
- There are memory semantics for all operations on mbarrier as well
- test_wait / try_wait: check if the specified phase has completed. there is phaseParity, which is like a flip

Others
- Errors related to mbarriers/tmem/tcgen05 may result in launch failure instead of runtime error
