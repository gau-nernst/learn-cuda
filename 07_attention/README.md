# Attention

Resources:
- https://tridao.me/publications/flash2/flash2.pdf

For bs=1, num_heads=8, len_query=4096, len_kv = 8192. 5090 @ 400W, compile with CUDA 12.9
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
