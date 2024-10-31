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
v1 (block+warp tiling, `mma.m16n8k8`)                                              |          1.98 |      92.93%
v2 (`mma.m16n8k16`)                                                                |          2.12 |      86.79%
v3 (`m16n8k6` with padded A shared memory)                                         |          1.92 |      95.83%

Lessons learned:
- Inline PTX: instruction, outputs, inputs, constraints
- `ldmatrix`: load 8x8 tiles of 16-bit from shared memory to registers. To be more exact, load 8 rows of 8 consecutive elements. Each thread holds 2 elements (there are 32 threads in a warp). Which thread holds which element is conveniently correct for `mma` instructions later. We can also load 2x and 4x of 8x8 tiles.
- To use `ldmatrix`, we need to convert pointer in generic address to shared address required by PTX using `ctva` instruction (convert address).
- `mma`: typical shape for FP16/BF16 is `m16n8k8` and `m16n8k16`. Each thread in a warp must hold specific elements in the tile, which can be done using `ldmatrix`. To load 16x8 tile, we use `ldmatrix.x2`. Similarly, for 16x16 tile, we use `ldmatrix.x4`.
- Accumulate result is held in register memory (across threads in a warp). We can write the results from register directly to global memory. Again, each thread hold specific elements of the output.
- When we use normal layout for shared memory, there will be bank conflicts when using `ldmatrix`: 8 rows reside in the same 4 banks (There are 32 banks. Since our block tile will always be a multiple of 32 to use vectorized global memory read, there will always be bank conflict for shared memory read).
- One classic way to fix this is to pad shared memory, which wastes resources (the padded shared memory is not used). The amount of padding must be 8 to ensure 16-bit alignment.
- The better way is to use **swizzled layout** for shared memory (TODO).
