# Matmul on MI355X

Resources:
- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html
- https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
- https://github.com/ROCm/gfx950-gluon-tutorials/blob/main/kernels/gemm/README.md

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14
uv pip install flydsl
```

Benchmark using Triton's `do_bench`

Kernel name                  | 4096    | 8192    | 16384
-----------------------------|---------|---------|--------
PyTorch (2.14.0+rocm7.14)    | 1450.22 | 1589.08 | 1267.29
v0 - Basic FlyDSL, FMA       |   53.97 |   29.31 |   22.37
v1 - Buffer DMA, MFMA layout |  835.61 | 1016.57 |  949.97

Learnings
- There are scalar (SGPRs) and vector (VGPRs) registers. Scalar means wave-uniform (same value across all lanes), vector means lane-private data. This is analogous to NVIDIA's uniform and normal registers.
- For global memory accesses, there are `BUFFER_LOAD_*` and `GLOBAL_LOAD_*`. The former has built-in bounds check, hence it is preferred if we need bounds check i.e. save registers and avoid complicated control flow.
- To use buffer load instructions, we have to create a **buffer resource descriptor**, which is a 128-bit value held in 4 SGPRs.
- MFMA layout: see `matmul_v1.py` for illustration.
