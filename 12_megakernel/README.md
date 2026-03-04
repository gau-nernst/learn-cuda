# Megakernels for LLM

Plan
- Target Qwen3 dense arch, small variants: 0.6B, 4B
  - Smallest Qwen3 MoE is still too large for a 5090 - 30B-A3B. May target a small MoE e.g. gpt-oss-20b?
- Start from small to big: MLP (gate+up+down projections), attention (qkv projections, RoPE, attention, out projection); from Triton to CUDA C++.
- Test at bs=1,8,64,512

## MLP Benchmark results

Note:
- Include RMS norm and residual connection
- For memory bandwidth, we also consider temporary buffers:
  - RMS norm: input (M * K), norm (K), output (M * K)
  - w13 projection: input (M * K), w13 (2 * N * K), output (M * N)
  - w2 projection: input (M * N), w2 (N * K), output (M * K)

**M=1, N=3072, K=1024**

Modal H200

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 56.19     | 0.43   | 336.32
torch.compile       | 50.93     | 0.46   | 371.03
Triton fused MLP    | 30.34     | 0.73   | 622.79
Triton 2-kernel MLP | 54.06     | 0.52   | 349.57

5090 (400W)

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 28.82     | 0.65   | 655.69
torch.compile       | 20.86     | 0.90   | 905.68
Triton fused MLP    | 22.56     | 0.91   | 837.64
Triton 2-kernel MLP | 24.76     | 0.84   | 763.09

**M=256, N=3072, K=1024**

Modal H200

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 58.79     |  82.19 | 410.28
torch.compile       | 72.73     |  66.43 | 331.61
Triton fused MLP    | 32.64     | 148.05 | 739.04
Triton 2-kernel MLP | 54.57     |  88.54 | 441.97

5090 (400W)

Kernel              | Time (us) | TFLOPS | Memory BW (GB/s)
--------------------|-----------|--------|-----------------
Eager               | 57.39     |  84.20 | 420.28
torch.compile       | 48.48     |  99.67 | 497.51
Triton fused MLP    | 55.71     |  86.72 | 432.91
Triton 2-kernel MLP | 57.53     |  83.99 | 419.26
