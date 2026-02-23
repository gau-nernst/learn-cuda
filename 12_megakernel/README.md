# Megakernels for LLM

Plan
- Target Qwen3 dense arch, small variants: 0.6B, 4B
  - Smallest Qwen3 MoE is still too large for a 5090 - 30B-A3B. May target a small MoE e.g. gpt-oss-20b?
- Start from small to big: MLP (gate+up+down projections), attention (qkv projections, RoPE, attention, out projection); from Triton to CUDA C++.
- Test at bs=1,8,64,512

## MLP Benchmark results

Note: for memory bandwidth, we consider input (M * K), temporary buffer (read and write, 2 * M * N), and w1, w2, w3 weights (3 * N * K), and output (M * K).

**M=1, N=3072, K=1024**

Modal H200

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 43.92     | 0.43   | 430.15
torch.compile       | 41.08     | 0.46   | 459.83
Triton fused MLP    | 25.76     | 0.73   | 733.25
Triton 2-kernel MLP | 36.21     | 0.52   | 521.69

5090 (400W)

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 24.05     | 0.78   |  785.58
torch.compile       | 18.10     | 1.04   | 1043.41
Triton fused MLP    | 20.56     | 0.69   |  918.73
Triton 2-kernel MLP | 22.58     | 0.77   |  836.79

**M=256, N=3072, K=1024**

Modal H200

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 41.31     | 116.97 | 558.45
torch.compile       | 58.48     |  82.63 | 394.50
Triton fused MLP    | 29.56     | 163.49 | 780.53
Triton 2-kernel MLP | 38.56     | 125.31 | 598.27

5090 (400W)

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 70.18     |  68.85 | 328.70
torch.compile       | 46.66     | 103.54 | 494.35
Triton fused MLP    | 55.33     |  87.33 | 416.93
Triton 2-kernel MLP | 55.33     |  87.32 | 416.90
