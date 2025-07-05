# Attention

Resources:
- https://tridao.me/publications/flash2/flash2.pdf

For bs=1, num_heads=8, len_query=4096, len_kv = 8192. 5090 @ 400W, compile with CUDA 12.9
- Theoretical limit: 209.5 TFLOPS

Kernel                         | TFLOPS | % of SOL
-------------------------------|--------|---------
`F.sdpa()` (Flash Attention)   | 185.19 | 88.40%
`F.sdpa()` (CuDNN)             | 202.18 | 96.51%
v1                             | 141.13 | 67.36%
v2 (shared memory swizzling)   | 179.62 | 85.74%
v3 (2-stage pipelining)        | 186.93 | 89.23%
v4 (`ldmatrix.x4` for K and V) | 192.01 | 91.65%
v5 (better pipelining)         | 196.44 | 93.77%
