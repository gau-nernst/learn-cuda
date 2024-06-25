# Sum

Resources:
- https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction
- https://github.com/cuda-mode/lectures/tree/main/lecture9

For M = 64, N = 32000, 4070Ti SUPER, compile with `-O3 --use_fast_math`

Kernel name                                         | Latency (us) | % of PyTorch | Bandwidth (GB/s)
----------------------------------------------------|--------------|--------------|------------------
Max theoretical bandwidth                           |           -- |           -- |           672.00
PyTorch                                             |        16.26 |      100.00% |           507.03
v1 (1 thread per row)                               |       799.49 |        2.03% |            10.25
v2 (parallel reduction tree)                        |        26.05 |       64.42% |           314.63
v3 (thread coarsening)                              |        15.46 |      105.17% |           530.33
v4a (warp-level reduction - use `volatile` keyword) |        15.17 |      107.19% |           540.41
v4b (use `__syncwrap()`)                            |        15.30 |      106.26% |           535.92
v4b (use warp shuffle intrinsic `__shfl_down_sync`) |        15.10 |      106.68% |           542.72
