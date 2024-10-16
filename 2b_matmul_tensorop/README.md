# Matrix multiplication - Tensor Cores

Resources:
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html and https://github.com/alexarmbr/matmul-playground
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/inline-ptx-assembly/
- https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch (see `copy_smxx.hpp` and `mma_smxx.hpp`)

For M = N = K = 4096, BF16 A row-major x B column-major, 4070Ti SUPER, compile with `-O3 --use_fast_math`

Kernel name                                                                        | Duration (ms) | % of CuBLAS
-----------------------------------------------------------------------------------|---------------|-------------
CuBLAS (via PyTorch) `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn` |          1.84 |     100.00%
v1 (block+warp tiling, `mma.m16n8k8`)                                              |          2.55 |      72.16%
v2 (`mma.m16n8k16`)                                                                |          2.46 |      74.80%
