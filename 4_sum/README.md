# Sum

Resources:
- https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction
- https://github.com/cuda-mode/lectures/tree/main/lecture9

For M = 64, N = 32000, 4070Ti SUPER, compile with `-O3 --use_fast_math`

Kernel name  | Latency (ms) | % of PyTorch | Bandwidth (GB/s)
-------------|--------------|--------------|------------------
PyTorch      |        16.26 |      100.00% |           507.03
