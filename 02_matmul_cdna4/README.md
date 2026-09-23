# Matmul on MI355X

Resources:
- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html
- https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
- https://github.com/ROCm/gfx950-gluon-tutorials/blob/main/kernels/gemm/README.md: general GEMM design
  - https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/intra_wave/a16w16: hill-climb
  - https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/inter_wave/a16w16: modified to inter-wave
  - https://github.com/ROCm/gfx950-gluon-tutorials/blob/main/docs/lds_throughput.md: mental modal of LDS performance
- https://rocm.docs.amd.com/projects/FlyDSL/en/latest/kernel_tuning_guide.html

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14
uv pip install flydsl

# profile
# view with https://github.com/ROCm/rocprof-compute-viewer/releases
rocprofv3 --att --kernel-include-regex matmul_v1 -d profile/matmul_v1 -- python main.py --profile 1

# measure bank conflicts
rocprof-compute profile -n matmul_banks -k matmul_v1 -b 12.2.9 --no-roof -- python main.py --profile 1
rocprof-compute analyze -p workloads/matmul_banks/MI355 -b 12.2.9
```

Benchmark using Triton's `do_bench`

Kernel name                  | 4096    | 8192    | 16384
-----------------------------|---------|---------|--------
PyTorch (2.14.0+rocm7.14)    | 1450.22 | 1589.08 | 1267.29
v0 - Basic FlyDSL, FMA       |   53.97 |   29.31 |   22.37
v1 - Buffer DMA, MFMA layout |  835.61 | 1016.57 |  949.97
v2 - LDS swizzle             | 1103.81 | 1276.01 | 1190.64

Learnings
- There are scalar (SGPRs) and vector (VGPRs) registers. Scalar means wave-uniform (same value across all lanes), vector means lane-private data. This is analogous to NVIDIA's uniform and normal registers.
- For global memory accesses, there are `BUFFER_LOAD_*` and `GLOBAL_LOAD_*`. The former has built-in bounds check, hence it is preferred if we need bounds check i.e. save registers and avoid complicated control flow.
- To use buffer load instructions, we have to create a **buffer resource descriptor**, which is a 128-bit value held in 4 SGPRs.
- MFMA layout: see `matmul_v1.py` for illustration.
- There are 4 SIMDs per CU, hence we need at least 4 waves to saturate the execution units.
