# Attention

Resources:
- https://tridao.me/publications/flash2/flash2.pdf

For bs=4, num_heads=8, len_query=4096, len_kv = 8192. 5090 @ 400W, compile with CUDA 12.9
- Theoretical limit: 209.5 TFLOPS

Kernel                         | TFLOPS | % of SOL
-------------------------------|--------|---------
`F.sdpa()` (Flash Attention)   | 186.73 | 89.13%
`F.sdpa()` (CuDNN)             | 203.61 | 97.19%
`flash-attn`                   | 190.58 | 90.97%
v1                             | 142.87 | 68.20%
v2 (shared memory swizzling)   | 181.11 | 86.45%
v3 (2-stage pipelining)        | 189.84 | 90.62%
v4 (`ldmatrix.x4` for K and V) | 194.33 | 92.76%
v5 (better pipelining)         | 197.74 | 94.39%

## Update 2026/10/03

We present a new table here because the previous optimization progression from v4 to v5 doesn't survive the new system setup. Hence, the old table is preserved as a historical artifact.
- Setup: PyTorch 2.14.1+cu130, system CUDA 13.3, driver 615.71.09, flash-attn-4 4.0.0b33
- Shape: bs=4, num_heads=8, len_query=4096, len_kv = 8192

TODO: add flash-attn and flash-attn-4 baseline

5090 @ 400W

| Kernel                |   Latency (ms) |   TFLOPS |   % SOL |
|:----------------------|---------------:|---------:|--------:|
| F.sdpa() - FA         |         2.9398 |   187.00 |   89.26 |
| F.sdpa() - CuDNN      |         2.7842 |   197.45 |   94.25 |
| v4 (cp.async 4-stage) |         2.7614 |   199.09 |   95.03 |
| v5 (cp.async 3-stage) |         2.8017 |   196.22 |   93.66 |
| v6 (TMA 3-stage)      |         2.6778 |   205.30 |   98.00 |

5090 @ 600W

| Kernel                |   Latency (ms) |   TFLOPS |   % SOL |
|:----------------------|---------------:|---------:|--------:|
| F.sdpa() - FA         |         2.8846 |   190.58 |   90.97 |
| F.sdpa() - CuDNN      |         2.6952 |   203.98 |   97.36 |
| flash-attn (CuteDSL)  |         2.7741 |   198.18 |   94.60 |
| v4 (cp.async 4-stage) |         2.5470 |   215.84 |  103.03 |
| v5 (cp.async 3-stage) |         2.6159 |   210.16 |  100.32 |
| v6 (TMA 3-stage)      |         2.5062 |   219.36 |  104.71 |
