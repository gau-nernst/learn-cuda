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
