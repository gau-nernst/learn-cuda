# Megakernels for LLM

Plan
- Target Qwen3 dense arch, small variants: 0.6B, 4B
  - Smallest Qwen3 MoE is still too large for a 5090 - 30B-A3B. May target a small MoE e.g. gpt-oss-20b?
- Start from small to big: MLP (gate+up+down projections), attention (qkv projections, RoPE, attention, out projection); from Triton to CUDA C++.
- Test at bs=1,8,64,512

## MLP Benchmark results

Note: for memory bandwidth, we consider input (M * K), temporary buffer (read and write, 2 * M * N), and w1, w2, w3 weights (3 * N * K), and output (M * K).

**M=1, N=1024, K=3072**

Modal H200

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 40.3      | 0.47   | 468.74
torch.compile       | 39.07     | 0.48   | 483.47
Triton fused MLP    | 27.23     | 0.69   | 693.65
Triton 2-kernel MLP | 36.12     | 0.52   | 523.06

**M=256, N=1024, K=3072**

Modal H200

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 42.43     | 113.87 | 543.66
torch.compile       | 66.19     |  73.00 | 348.52
Triton fused MLP    | 28.34     | 170.50 | 814.02
Triton 2-kernel MLP | 34.38     | 140.56 | 671.05
