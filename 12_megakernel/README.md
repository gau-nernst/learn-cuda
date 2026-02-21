# Megakernels for LLM

Plan
- Target Qwen3 dense arch, small variants: 0.6B, 4B
  - Smallest Qwen3 MoE is still too large for a 5090 - 30B-A3B. May target a small MoE e.g. gpt-oss-20b?
- Start from small to big: MLP (gate+up+down projections), attention (qkv projections, RoPE, attention, out projection); from Triton to CUDA C++.
- Test at bs=1,8,64,512
