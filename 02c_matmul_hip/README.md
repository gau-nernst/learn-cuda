# Matmul on MI300X

Resources:
- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- https://github.com/tile-ai/tilelang/blob/v0.1.6.post1/src/tl_templates/hip/gemm.h

Kernel name               | TFLOPS 
--------------------------|--------
PyTorch (2.8.0+rocm6.4)   | 558.25
v1                        | 123.00

TODO:
- Swap A and B in `mfma` to use `bf16x4` store for C
- Pipelining via registers - separate gmem->smem to gmem->rmem and rmem->smem
  - Is staging via smem necessary?
- kpack - What is it? Mix loop ordering to try different instruction ordering
- Make sure loops are unrolled
- Profiling tools: https://github.com/ROCm/rocm-systems
- Smem swizzling

To use ROCm Compute profiler

```bash
uv pip install -r /opt/rocm-6.4.1/libexec/rocprofiler-compute/requirements.txt
rocprof-compute profile -n mm -- python main.py --profile v1
```
