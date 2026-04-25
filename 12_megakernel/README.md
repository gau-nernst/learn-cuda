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

The measurement is done with `chat.py`, which is not rigorous by any means. Reporting decode speed at various number of input tokens. Using Qwen3-0.6B. HF eager and Ours use PyTorch 2.11 (CUDA 13.0).

Implementation           | 15 in toks | 1144 in tokens
-------------------------|------------|----------------
HF eager                 | 127 tok/s  | 132 tok/s
vLLM 0.19 (PyTorch 2.10) | 616 tok/s  | 599 tok/s
Ours                     | 815 tok/s  | 795 tok/s

## MLP Benchmark results

Note:
- Include RMS norm and residual connection
- For memory bandwidth, we also consider temporary buffers:
  - RMS norm: input (M * K), norm (K), output (M * K)
  - w13 projection: input (M * K), w13 (2 * N * K), output (M * N)
  - w2 projection: input (M * N), w2 (N * K), output (M * K)

**M=1, N=3072, K=1024**

PyTorch 2.11 (CUDA 13.0)

Kernel                 | 5090 (400W)            | H200 (Modal)
-----------------------|------------------------|-----------------------
Eager                  | 22.26us /  849.03 GB/s | 62.61us /  301.84 GB/s
torch.compile          | 22.46us /  841.43 GB/s | 62.23us /  303.68 GB/s
Triton v1 fused MLP    | 17.71us / 1067.16 GB/s | 31.19us /  605.82 GB/s
Triton v1 2-kernel MLP | 21.55us /  876.79 GB/s | 55.70us /  339.26 GB/s
GEMV Triton v1         | 15.85us / 1192.12 GB/s | 22.61us /  835.71 GB/s
GEMV CUDA v1           | 15.40us / 1226.72 GB/s | 11.65us / 1621.78 GB/s

**M=256, N=3072, K=1024**

PyTorch 2.11 (CUDA 13.0)

Kernel                 | 5090 (400W)                           | H200 (Modal)
-----------------------|---------------------------------------|--------------------------------------
Eager                  | 47.35us / 102.04 TFLOPS / 509.34 GB/s | 56.53us /  85.47 TFLOPS / 426.63 GB/s
torch.compile          | 45.78us / 105.53 TFLOPS / 526.80 GB/s | 79.42us /  60.84 TFLOPS / 303.68 GB/s
Triton v1 fused MLP    | 48.61us /  99.40 TFLOPS / 496.16 GB/s | 32.67us / 147.90 TFLOPS / 738.29 GB/s
Triton v1 2-kernel MLP | 45.91us / 105.24 TFLOPS / 525.35 GB/s | 54.55us /  88.58 TFLOPS / 442.18 GB/s

## Decode Attention Benchmark results

Note:
- Include RMS norm and residual connection
- torch.compile will specialize on `kv_size`, which is not quite valid.

**kv_size=128, dim=1024, num_heads=16, num_kv_heads=8**

PyTorch 2.11 (CUDA 13.0)

Kernel        | 5090 (400W)            | H200 (Modal)
--------------|------------------------|-----------------------
Eager         | 157.10us /  83.50 GB/s | 361.64us /  36.27 GB/s
torch.compile |  69.08us / 189.90 GB/s | 170.58us /  76.91 GB/s
Triton v1     |  19.15us / 685.06 GB/s |  25.83us / 507.81 GB/s
Triton v2     |  19.27us / 680.84 GB/s |  29.06us / 451.44 GB/s

**kv_size=4096, dim=1024, num_heads=16, num_kv_heads=8**

PyTorch 2.11 (CUDA 13.0)

Kernel        | 5090 (400W)             | H200 (Modal)
--------------|-------------------------|-----------------------
Eager         | 171.03us /  171.74 GB/s | 366.52us /  80.13 GB/s
torch.compile |  67.72us /  433.70 GB/s | 175.89us / 166.99 GB/s
Triton v1     |  68.45us /  429.08 GB/s |  81.41us / 360.80 GB/s
Triton v2     |  28.67us / 1024.62 GB/s |  30.30us / 969.44 GB/s
