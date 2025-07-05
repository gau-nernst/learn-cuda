# Attention

Resources:
- https://tridao.me/publications/flash2/flash2.pdf

For bs=1, num_heads=8, len_query=4096, len_kv = 8192. 5090 @ 400W, compile with CUDA 12.9
- Theoretical limit: 209.5 TFLOPS

Kernel                       | TFLOPS | % of SOL
-----------------------------|--------|---------
`F.sdpa()` (Flash Attention) | 185.18 | 88.39%
`F.sdpa()` (CuDNN)           | 202.07 | 96.45%
v1                           | 140.82 | 67.22%
v2 (shared memory swizzling) | 179.38 | 85.62%
v3 (2-stage pipelining)      | 186.82 | 89.18%
