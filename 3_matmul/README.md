# Matrix Multiplication

Resources:
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
- https://siboehm.com/articles/22/CUDA-MMM
- https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/

For M = N = K = 4096, 4070Ti SUPER, compile with `-O3 --use_fast_math`

Kernel name                                                        | Latency (ms) | % of CuBLAS | Bandwidth (GB/s)
-------------------------------------------------------------------|--------------|-------------|------------------
CuBLAS (via PyTorch) `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` |         4.77 |     100.00% |           104.25
v1 (naive 1 row dot 1 column)                                      |        56.21 |       8.49% |           195.98
v2 (shared memory cache with 2D block)                             |        44.38 |      10.75% |           387.27
v3 (thread coarsening)                                             |        39.03 |      12.22% |            38.49
