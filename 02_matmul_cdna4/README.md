# Matmul on MI355X

Resources:
- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html
- 8-wave ping-pong vs 4-wave interleave:
  - https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
- https://github.com/ROCm/gfx950-gluon-tutorials/blob/main/kernels/gemm/README.md: general GEMM design
  - https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/intra_wave/a16w16: hill-climb
  - https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/inter_wave/a16w16: modified to inter-wave
  - https://github.com/ROCm/gfx950-gluon-tutorials/blob/main/docs/lds_throughput.md: mental modal of LDS performance
- https://rocm.docs.amd.com/projects/FlyDSL/en/latest/kernel_tuning_guide.html
- https://llvm.org/docs/AMDGPUUsage.html

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
PyTorch (2.14.0+rocm7.14)    | 1420.69 | 1568.50 | 1516.97
v0 - Basic FlyDSL, FMA       |   53.34 |   29.74 |   22.47
v1 - Buffer DMA, MFMA layout |  858.96 | 1003.89 |  935.53
v2 - LDS swizzle             | 1092.32 | 1271.48 | 1183.53
v3 - Double buffer G2S       | 1047.24 | 1071.21 | 1002.35
v2b - Scheduling intrinsics  | 1144.30 | 1296.94 | 1190.24

Learnings
- There are scalar (SGPRs) and vector (VGPRs) registers. Scalar means wave-uniform (same value across all lanes), vector means lane-private data. This is analogous to NVIDIA's uniform and normal registers.
- For global memory accesses, there are `BUFFER_LOAD_*` and `GLOBAL_LOAD_*`. The former has built-in bounds check, hence it is preferred if we need bounds check i.e. save registers and avoid complicated control flow.
- To use buffer load instructions, we have to create a **buffer resource descriptor**, which is a 128-bit value held in 4 SGPRs.
- MFMA layout: see `matmul_v1.py` for illustration.
- There are 4 SIMDs per CU, hence we need at least 4 waves to saturate the execution units.
- Use async version of DMA buffer load to implement double buffering. The async API (`asyncmark()` and `wait_asyncmark()`) is compiler helpers: the compiler tracks number of DMA issues and inserts `s_waitcnt vmcnt(X)` accordingly.
- There are 512 registers per lane per SIMD. Using 4 waves matching 4 SIMD:
  - Occupancy=1: 512 registers/thread
  - Occupancy=2: 256 registers/thread
  - Occpuancy=3: 168 registers/thread
  - Occupancy=4: 128 registers/thread
  - Careful when crossing the threshold, which reduces occupancy abruptly.
- VGPRs and AGPRs share the same pool of registers. In addition, each thread can only have up to 256 VGPRs. Hence, we only need to use AGPRs when we target occupancy=1 i.e. 1 wave/SIMD, 256 VGPRs and AGPRs. Otherwise, there is no point using AGPRs (AGPRs can only be used as MFMA accumulator).
- **Scheduling intrinsics**: There are intrinsics to influence instruction scheduling. See https://llvm.org/docs/AMDGPUUsage.html `llvm.amdgcn.sched` for more details.
  - `sched_barrier(0)`: no instructions can cross the barrier. Change `0` to another mask values to select what instruction types can still cross the barrier.
  - `sched_dsrd(M) / sched_mfma(N)`: schedule groups, enforcing ordering between groups. For example, `sched_dsrd(M) + sched_mfma(N)` means schedule M `ds_read` THEN schedule N `mfma`. It will schedule ANY instructions of that type appearing before the intrinsic, not necessarily the immediately preceding instructions.
