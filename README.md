# Learn CUDA with PyTorch

Resources:

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://github.com/cuda-mode/lectures
- https://github.com/NVIDIA/cuda-samples
- https://leimao.github.io/tags/CUDA/

Name | Description
-----|-------------
1\. [Vector addition](1_vector_addition/) | Simple example to get everything working.
2\. [Box blur](2_box_blur/) | 2D CUDA blocks/threads. TODO: optimize with separable filters, moving average.
3\. [Matrix multiplication](3_matmul/) | Tiling.
4\. Softmax | TODO

## Learnings

- CUDA architecture: a GPU consists of multiple Streaming Multiprocessors (SMs). Each SM contains several warps. Each warp contains 32 CUDA threads.
- All threads in a warp share a program counter. In other words, all threads in a warp must execute the same instruction (Single Instruction, Multiple Threads - SIMT).
- Block (of threads) is a software-defined concept. It consists of a user-defined number of threads (max 1024 for most consumer GPUs). It will be executed by a single SM (to be exact, by multiple warps in an SM).
- There are 3 main types of memory (in order of increasing speed and decreasing capacity):
    - Global memory (DRAM or HBM): this consists of off-chip memory modules. Your normal data (tensors, ...) reside here. This is typically in order of GBs (8-16GB for consumer GPUs). In user code, you allocate globaly memory with `cudaMalloc()` and write data to it with `cudaMemcpy()`.
    - Shared memory (SRAM): this is on-chip memory. Each SM has 64kb of shared memory, which is shared among all threads in a block. This can be used as working space for storing intermediate results and collaboration among threads within a block. To use it in kernel code, we either write `__shared__ float shmem[64];` for static-size array, or `extern __shared__ float shmem[];` for dynamically allocated shared memory, which is set by kernel call execution configuration `kernel<<<n_blocks, n_threads, shmem_size>>>`().
    - Register/local memory: per-thread memory. This is where the local variables live.
