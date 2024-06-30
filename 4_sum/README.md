# Sum

Resources:
- https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction
- https://github.com/cuda-mode/lectures/tree/main/lecture9
- https://developer.nvidia.com/blog/cooperative-groups/

For M = 64, N = 32000, 4070Ti SUPER, compile with `-O3 --use_fast_math`

Kernel name                                         | Latency (us) | % of PyTorch | Bandwidth (GB/s)
----------------------------------------------------|--------------|--------------|------------------
Max theoretical bandwidth                           |           -- |           -- |           672.00
PyTorch                                             |        16.26 |      100.00% |           507.03
v1 (1 thread per row)                               |       799.49 |        2.03% |            10.25
v2 (parallel reduction tree)                        |        26.05 |       64.42% |           314.63
v3 (thread coarsening)                              |        15.36 |      105.86% |           533.64
v4a (warp-level reduction - use `volatile` keyword) |        15.07 |      107.90% |           543.86
v4b (use `__syncwrap()`)                            |        15.14 |      107.40% |           541.59
v4c (use warp shuffle intrinsic `__shfl_down_sync`) |        15.07 |      107.90% |           543.88
v5 (cooperative groups)                             |        15.14 |      107.40% |           541.59
v6 (vectorized load)                                |        15.01 |      108.33% |           546.14

Lessons learned:
- Parallel reduction tree. Avoid bank conflicts by adding a tile of data to a tile of data (sequential addressing).
- Warp intrinsics for warp-to-warp communication (avoid round trip to shared memory).
- Cooperative groups: seem like they are meant to manage sub-warp computations better. For reduction use case, it is not faster.
