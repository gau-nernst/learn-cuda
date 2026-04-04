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

The measurement is done with `chat.py`, which is not rigorous by any means. Reporting decode speed at various number of input tokens. Using Qwen3-0.6B.

Implementation | 15 in toks | 1218 in tokens
---------------|------------|----------------
HF eager       | 130 tok/s  | 130 tok/s
vLLM           | 620 tok/s  | 615 tok/s
Ours           | 820 tok/s  | 613 tok/s

## MLP Benchmark results

Note:
- Include RMS norm and residual connection
- For memory bandwidth, we also consider temporary buffers:
  - RMS norm: input (M * K), norm (K), output (M * K)
  - w13 projection: input (M * K), w13 (2 * N * K), output (M * N)
  - w2 projection: input (M * N), w2 (N * K), output (M * K)

**M=1, N=3072, K=1024**

Modal H200

Kernel                 | Time (us) | TFLOPS | Memory BW (GB/s)
-----------------------|-----------|--------|-----------------
Eager                  | 56.19     | 0.43   | 336.32
torch.compile          | 50.93     | 0.46   | 371.03
Triton v1 fused MLP    | 30.34     | 0.73   | 622.79
Triton v1 2-kernel MLP | 54.06     | 0.52   | 349.57
Triton v2              | 21.31     | 0.89   | 886.75

5090 (400W). PyTorch 2.11 (CUDA 13.0)

Kernel                 | Time (us) | TFLOPS | Memory BW (GB/s)
-----------------------|-----------|--------|-----------------
Eager                  | 22.28     | 0.85   |  848.01
torch.compile          | 22.51     | 0.84   |  839.47
Triton v1 fused MLP    | 17.73     | 1.06   | 1065.63
Triton v1 2-kernel MLP | 21.70     | 0.87   |  870.94
Triton v2              | 15.87     | 1.19   | 1190.96

**M=256, N=3072, K=1024**

Modal H200

Kernel                 | Time (us) | TFLOPS | Memory BW (GB/s)
-----------------------|-----------|--------|-----------------
Eager                  | 58.79     |  82.19 | 410.28
torch.compile          | 72.73     |  66.43 | 331.61
Triton v1 fused MLP    | 32.64     | 148.05 | 739.04
Triton v1 2-kernel MLP | 54.57     |  88.54 | 441.97

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

Modal H200

Kernel        | Time (us) | Memory BW (GB/s)
--------------|-----------|-----------------
Eager         | 401.03    |  32.71
torch.compile | 162.57    |  80.70
Triton v1     |  25.51    | 514.31

5090 (400W)

Kernel        | Time (us) | Memory BW (GB/s)
--------------|-----------|-----------------
Eager         | 163.26    |  80.35
torch.compile |  69.61    | 188.45
Triton v1     |  22.70    | 577.97

**kv_size=4096, dim=1024, num_heads=16, num_kv_heads=8**

Modal H200

Kernel        | Time (us) | Memory BW (GB/s)
--------------|-----------|-----------------
Eager         | 402.88    |  72.90
torch.compile | 178.22    | 164.81
Triton v1     |  93.61    | 313.76

5090 (400W)

Kernel        | Time (us) | Memory BW (GB/s)
--------------|-----------|-----------------
Eager         | 205.04    | 143.25
torch.compile |  75.87    | 387.15
Triton v1     |  93.34    | 314.68
