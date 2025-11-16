# Learn CUDA with PyTorch

Resources:

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://github.com/cuda-mode/lectures
- https://github.com/NVIDIA/cuda-samples
- https://leimao.github.io/tags/CUDA/

Name | Description
-----|-------------
01\. [Vector addition](01_vector_addition/) | Simple example to get everything working.
02a\. [Matrix multiplication SIMT](02a_matmul_simt/) | Block tiling, thread tiling, warp tiling.
02b\. [Matrix multiplication SM80](02b_matmul_sm80/) | Inline PTX, `cvta`, `ldmatrix`, `mma`.
02c\. [Matrix multiplication SM120](02c_matmul_sm120/)
02d\. [Matrix multiplication CDNA3](02d_matmul_cdna3/)
03\. [Sum](03_sum/) | Reduction in general.  Prepare for softmax (max and sum).
04\. [Softmax](04_softmax/) | Naive (safe) softmax, online softmax. `atomicCAS()`. Single-block and multi-block per row.
05\. [FP6](05_fp6/) | FP6 primitives (FP32/FP16/BF16<->FP6).
06\. [Box blur](06_box_blur/) | 2D CUDA blocks/threads. TODO: optimize with separable filters, moving average.
07\. [Attention](07_attention/) | Flash attention
08\. [Row-scaled matmul](08_row_scaled_mm/) | Simple epilogue
09\. [Block-scaled matmul](09_block_scaled_mm/) | MXFP8

```bash
# profile a CUDA kernel
ncu --set full python main.py

# debug illegal memory access
compute-sanitizer python main.py
```

and open the generated `profile.ncu-rep` file in Nsight Compute. See more here: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html

To profile PyTorch program, run

```python
with torch.profiler.profile() as prof:
    # PyTorch program here
    ...

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

and open the saved `trace.json` file at https://ui.perfetto.dev/. See more here: https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile

To support syntax highlighting and code suggestion in IDE, add the following include paths (can be seen in the command that PyTorch uses to compile inline C++ code).

```
"/home/thien/miniconda3/envs/dev/lib/python3.10/site-packages/torch/include",
"/home/thien/miniconda3/envs/dev/lib/python3.10/site-packages/torch/include/torch/csrc/api/include",
"/home/thien/miniconda3/envs/dev/lib/python3.10/site-packages/torch/include/THC",
"/usr/local/cuda/include",
"/home/thien/miniconda3/envs/dev/include/python3.10"
```

Change the paths appropriately for your system. For Windows, the paths are slightly different, but then again, you can see them in the PyTorch compile command. For VSCode, add the paths to `.vscode/c_cpp_properties.json` (VSCode will prompt you to create one if it does not exist).

## Learnings

- CUDA architecture: a GPU consists of multiple **Streaming Multiprocessors** (SMs). Each SM executes multiple **warps**. Each warp contains **32 CUDA threads**.
- All threads in a warp share a program counter. In other words, all threads in a warp must execute the same instruction (Single Instruction, Multiple Threads - SIMT). If threads in a warp have to execute different instructions (e.g. due to branching), there will be **warp divergence**: each instruction will be executed sequentially. Note that threads in different warps executing different instructions is fine (e.g. threads 0-31 do A, threads 32-63 do B -> no warp divergence).
- Block (of threads) is a software-defined concept. It consists of a user-defined number of threads (max 1024 for most consumer GPUs). It will be executed by a single SM - the threadblock is partitioned into warps, and each warp is scheduled to execute.
- There are 3 main types of memory (in order of increasing speed and decreasing capacity):
  - **Global memory** (DRAM or HBM): this consists of off-chip memory modules. Your normal data (tensors, ...) reside here. This is typically in order of GBs (8-16GB for consumer GPUs). In user code, you allocate globaly memory with `cudaMalloc()` and write data to it with `cudaMemcpy()`.
  - **L2 cache**
  - **Shared memory + L1 cache** (SRAM): this is on-chip memory. Traditionally NVIDIA GPUs only have 48kb shared memory per SM, but later generations can have up to 100kb. This is private for each SM, hence it is shared within a threadblock.
    - Unused shared memory is allocated to L1 cache.
    - This can be used as working space for storing intermediate results and collaboration among threads within a block.
    - To use it in kernel code, we either write `__shared__ float shmem[64];` for static-size array, or `extern __shared__ float shmem[];` for dynamically allocated shared memory, which is set by kernel call execution configuration `kernel<<<n_blocks, n_threads, shmem_size>>>`().
    - To use more than 48kb of shared memory, we need to use `cudaSetFuncAttribute()`.
  - **Register file**: per-thread memory. Usually local variables are backed by registers, as determined by the compiler. All registers are 32-bit.
- Performant memory access: (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
    - **Global memory**: to ensure coalesced memory access (i.e. threads in a warp accessing different memory locations with a single instruction), the rule of thumb is to make consecutive threads accessing consecutive memory locations (+ memory alignment requirement, which is most of the time satisfied). (L1 and L2. Cache line...) A global memory transaction has a maximum of 128 bytes (e.g. 32 fp32 or 32 int32). Rule of thumb - each thread loads 16 bytes (128 bits) (equivalent to `float4`). For direct global->shared memory copy without using registers, we can use `cp.async`.
    - **Shared memory**: to avoid memory bank conflict...
- Typically, we load a tile of data (might be larger than block size) from global memory to shared memory, do computations on shared memory, then write results from shared memory back to global memory. To achieve coalesced memory access (and making the kernel easier to reason about), usually we decouple global memory access from computation: during global memory access, we assign consecutive threads to consecutive data, but during computation, a thread may be responsible for a different data element.
- When writing Triton kernel, always use `tl.dot(allow_tf32=False)`. With fp32 inputs, the outputs are wrong. There are no clear reasons for this. With fp16/bf16, this flag doesn't matter.
