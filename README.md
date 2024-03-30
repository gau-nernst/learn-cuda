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
4\. [Sum](4_sum/) | Reduction in general.  Prepare for softmax (max and sum).
5\. [Softmax](5_softmax) | TODO
... optimizers, quantization, flash attention | TODO

To profile a CUDA kernel, run

```bash
ncu --set full python main.py
```

and open the generated `profile.ncu-rep` file in Nsight Compute. See more here: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html

To profile PyTorch program, run

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    # PyTorch program here
    ...

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

and open the generated `trace.json` file in Chrome's `about:tracing`. See more here: https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile

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

- CUDA architecture: a GPU consists of multiple Streaming Multiprocessors (SMs). Each SM contains several warps. Each warp contains 32 CUDA threads.
- All threads in a warp share a program counter. In other words, all threads in a warp must execute the same instruction (Single Instruction, Multiple Threads - SIMT).
- Block (of threads) is a software-defined concept. It consists of a user-defined number of threads (max 1024 for most consumer GPUs). It will be executed by a single SM (to be exact, by multiple warps in an SM).
- There are 3 main types of memory (in order of increasing speed and decreasing capacity):
    - Global memory (DRAM or HBM): this consists of off-chip memory modules. Your normal data (tensors, ...) reside here. This is typically in order of GBs (8-16GB for consumer GPUs). In user code, you allocate globaly memory with `cudaMalloc()` and write data to it with `cudaMemcpy()`.
    - Shared memory (SRAM): this is on-chip memory. Each SM has 64kb of shared memory, which is shared among all threads in a block. This can be used as working space for storing intermediate results and collaboration among threads within a block. To use it in kernel code, we either write `__shared__ float shmem[64];` for static-size array, or `extern __shared__ float shmem[];` for dynamically allocated shared memory, which is set by kernel call execution configuration `kernel<<<n_blocks, n_threads, shmem_size>>>`().
    - Register/local memory: per-thread memory. This is where the local variables live.
- When writing Triton kernel, always use `tl.dot(allow_tf32=False)`. With fp32 inputs, the outputs are wrong. There are no clear reasons for this. With fp16/bf16, this flag doesn't matter.
