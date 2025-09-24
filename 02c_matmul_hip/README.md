# Matmul on MI300X

Resources:
- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- https://github.com/tile-ai/tilelang/blob/v0.1.6.post1/src/tl_templates/hip/gemm.h
- https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html

Kernel name               | TFLOPS 
--------------------------|--------
PyTorch (2.8.0+rocm6.4)   | 548.09
v1a                       | 122.21
v1b - Pad smem            | 244.72
v2 - smem swizzling       | 369.57

TODO:
- Swap A and B in `mfma` to use `bf16x4` store for C
- Pipelining via registers - separate gmem->smem to gmem->rmem and rmem->smem
- kpack - What is it? Mix loop ordering to try different instruction ordering
- Make sure loops are unrolled

To use ROCm Compute profiler

```bash
uv pip install -r /opt/rocm-6.4.1/libexec/rocprofiler-compute/requirements.txt
rocprof-compute profile -n mm_v1a -- python main.py --profile v1a
rocprof-compute analyze -p workloads/mm_v1a/MI300
```

Worklog
- v1a: Basic matmul structure. gmem->smem via registers. No pipelining. Understand mfma layout.
- Tried `rocprof-compute`, but I didn't know where to look for bottlenecks, like Nsight Compute's warp stall analysis does.
- v1b: Reduce smem bank conflicts with padded smem or swizzling.
- Most errors don't segfault / error immediately e.g. reading out of bounds for smem. This makes it hard to identify bugs.
