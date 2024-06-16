# Matrix Multiplication

Resources:
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
- https://siboehm.com/articles/22/CUDA-MMM
- https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/

For M = N = K = 4096

Kernel name                      | Latency (ms) | % of CuBLAS
---------------------------------|--------------|-------------
CuBLAS (via PyTorch)             |              |        100%
v1 (naive 1 row dot 1 column)    |
v2 (block read to shared memory) |
