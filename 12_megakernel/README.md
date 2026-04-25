# Megakernels for LLM

Plan
- Target Qwen3 dense arch, small variants: 0.6B, 4B
  - Smallest Qwen3 MoE is still too large for a 5090 - 30B-A3B. May target a small MoE e.g. gpt-oss-20b?
- Start from small to big: MLP (gate+up+down projections), attention (qkv projections, RoPE, attention, out projection); from Triton to CUDA C++.
- Test at bs=1,8,64,512

## End2end decode benchmark

To try interactive chat (only Qwen3-0.6B is supported at the moment)

```bash
python chat.py
```

The measurement is done with `chat.py`, which is not rigorous by any means. Reporting decode speed at various number of input tokens. Using Qwen3-0.6B. HF eager and Ours use PyTorch 2.11 (CUDA 13.0), vLLM xx uses PyTorch 2.10.

Implementation | 15 in toks | 1144 in tokens
---------------|------------|----------------
HF eager       | 135 tok/s  | 132 tok/s
vLLM 0.19      | 629 tok/s  | 610 tok/s
Ours           | 859 tok/s  | 664 tok/s

## MLP Benchmark results

Note:
- Include RMS norm and residual connection
- For memory bandwidth, we also consider temporary buffers:
  - RMS norm: input (M * K), norm (K), output (M * K)
  - w13 projection: input (M * K), w13 (2 * N * K), output (M * N)
  - w2 projection: input (M * N), w2 (N * K), output (M * K)

**M=1, N=3072, K=1024**

PyTorch 2.11 (CUDA 13.0)

Kernel                 | H200 (Modal)
-----------------------|-----------------------
Eager                  | 62.61us /  301.84 GB/s
torch.compile          | 62.23us /  303.68 GB/s
Triton v1 fused MLP    | 31.19us /  605.82 GB/s
Triton v1 2-kernel MLP | 55.70us /  339.26 GB/s
GEMV Triton v1         | 22.61us /  835.71 GB/s
GEMV CUDA v1           | 11.65us / 1621.78 GB/s

5090 (400W). PyTorch 2.11 (CUDA 13.0) - to be updated

Kernel                 | Time (us) | TFLOPS | Memory BW (GB/s)
-----------------------|-----------|--------|-----------------
Eager                  | 22.28     | 0.85   |  848.01
torch.compile          | 22.51     | 0.84   |  839.47
Triton v1 fused MLP    | 17.73     | 1.06   | 1065.63
Triton v1 2-kernel MLP | 21.70     | 0.87   |  870.94
GEMV Triton v1         | 15.87     | 1.19   | 1190.96

**M=256, N=3072, K=1024**

PyTorch 2.11 (CUDA 13.0)

Kernel                 | H200 (Modal)
-----------------------|---------------------------------------
Eager                  | 56.53us /  85.47 TFLOPS / 426.63 GB/s
torch.compile          | 79.42us /  60.84 TFLOPS / 303.68 GB/s
Triton v1 fused MLP    | 32.67us / 147.90 TFLOPS / 738.29 GB/s
Triton v1 2-kernel MLP | 54.55us /  88.58 TFLOPS / 442.18 GB/s

5090 (400W). PyTorch 2.11 (CUDA 13.0)

Kernel                 | Time (us) | TFLOPS | Memory BW (GB/s)
-----------------------|-----------|--------|-----------------
Eager                  | 48.57     |  99.48 | 496.55
torch.compile          | 47.05     | 102.69 | 512.61
Triton v1 fused MLP    | 48.89     |  98.83 | 493.36
Triton v1 2-kernel MLP | 46.24     |  90.68 | 521.63

## Decode Attention Benchmark results

Note:
- Include RMS norm and residual connection
- torch.compile will specialize on `kv_size`, which is not quite valid.

**kv_size=128, dim=1024, num_heads=16, num_kv_heads=8**

PyTorch 2.11 (CUDA 13.0)

Kernel        | H200 (Modal)
--------------|------------------------
Eager         | 361.64us /  36.27 GB/s
torch.compile | 170.58us /  76.91 GB/s
Triton v1     |  25.83us / 507.81 GB/s
Triton v2     |  29.06us / 451.44 GB/s

5090 (400W). PyTorch 2.11 (CUDA 13.0)

Kernel        | Time (us) | Memory BW (GB/s)
--------------|-----------|-----------------
Eager         | 157.29    |  83.40
torch.compile |  69.37    | 189.10
Triton v1     |  18.60    | 705.18

**kv_size=4096, dim=1024, num_heads=16, num_kv_heads=8**

PyTorch 2.11 (CUDA 13.0)

Kernel        | H200 (Modal)
--------------|------------------------
Eager         | 366.52us /  80.13 GB/s
torch.compile | 175.89us / 166.99 GB/s
Triton v1     |  81.41us / 360.80 GB/s
Triton v2     |  30.30us / 969.44 GB/s

5090 (400W). PyTorch 2.11 (CUDA 13.0)

Kernel        | Time (us) | Memory BW (GB/s)
--------------|-----------|-----------------
Eager         | 171.12    | 171.64
torch.compile |  68.32    | 429.90
Triton v1     |  59.43    | 494.24
