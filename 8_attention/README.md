# Attention

Resources:
- https://tridao.me/publications/flash2/flash2.pdf

For bs=1, num_heads=8, len_query=4096, len_kv = 8192. 5090 @ 400W, compile with CUDA 12.9
- Theoretical limit: 209.5 TFLOPS

Kernel                         | TFLOPS | % of SOL
-------------------------------|--------|---------
`F.sdpa()` (Flash Attention)   | 187.13 | 89.32%
`F.sdpa()` (CuDNN)             | 204.31 | 97.52%
v1                             | 143.32 | 68.41%
v2 (shared memory swizzling)   | 181.48 | 86.62%
v3 (2-stage pipelining)        | 191.15 | 91.24%
v4 (`ldmatrix.x4` for K and V) | 195.08 | 93.12%
