# Row-scaled matmul

For M = N = 4096, K = 8192, INT8/FP8 A row-major x B column-major, 5090 @ 400W, compile with CUDA 12.9

### INT8

- Theoretical limit: 838 INT8 TFLOPS

Kernel name               | TFLOPS | % of SOL
--------------------------|--------|----------
PyTorch (Inductor Triton) | 555.05 | 66.23%
v1                        | 559.04 | 66.71%

### FP8

- Theoretical limit: 419 FP8 TFLOPS

Kernel name               | TFLOPS | % of SOL
--------------------------|--------|----------
PyTorch (Inductor Triton) | 394.76 | 94.21%
v1                        | 502.95 | 120.04%

Observations
- It's strange that **NOT** using pipelining results in the fastest runtime for my kernel.
- For FP8, we can exceed SOL on 5090 because we use the faster instruction - MMA for MXFP8. The normal FP8 (and BF16) instruction is "nerfed" by NVIDIA for consumer cards.
- The exact same kernel (with the same kernel params) is faster with INT8 than with FP8. This may indicate that INT8 MMA uses **less power** than FP8 MMA, hence the kernel is less power-limited.
