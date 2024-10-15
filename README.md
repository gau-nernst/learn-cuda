# Learn CUDA with PyTorch

Resources:

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://github.com/cuda-mode/lectures
- https://github.com/NVIDIA/cuda-samples
- https://leimao.github.io/tags/CUDA/

Name | Description
-----|-------------
1\. [Vector addition](1_vector_addition/) | Simple example to get everything working.
2\. [Matrix multiplication SIMT](2a_matmul_simt/) | Block tiling, thread tiling, warp tiling.
2\. [Matrix multiplication TensorOp](2b_matmul_tensorop/) | Inline PTX, `cvta`, `ldmatrix`, `mma`.
3\. [Sum](3_sum/) | Reduction in general.  Prepare for softmax (max and sum).
4\. [Softmax](4_softmax) | TODO
5\. [FP6](5_fp6) | FP6 primitives (FP32/FP16/BF16<->FP6).
6\. [Box blur](6_box_blur/) | 2D CUDA blocks/threads. TODO: optimize with separable filters, moving average.
7\. Matrix multiplication Tensor Cores | Tensor cores
... optimizers, quantization, flash attention, gemv (split-K, stream-K), scan | TODO

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

- CUDA architecture: a GPU consists of multiple **Streaming Multiprocessors** (SMs). Each SM contains several **warps**. Each warp contains **32 CUDA threads**.
- All threads in a warp share a program counter. In other words, all threads in a warp must execute the same instruction (Single Instruction, Multiple Threads - SIMT). If threads in a warp have to execute different instructions (e.g. due to branching), there will be **warp divergence**: each instruction will be executed sequentially. Note that threads in different warps executing different instructions is fine (e.g. threads 0-31 do A, threads 32-63 do B -> no warp divergence).
- Block (of threads) is a software-defined concept. It consists of a user-defined number of threads (max 1024 for most consumer GPUs). It will be executed by a single SM (to be exact, by multiple warps in an SM).
- There are 3 main types of memory (in order of increasing speed and decreasing capacity):
    - **Global memory** (DRAM or HBM): this consists of off-chip memory modules. Your normal data (tensors, ...) reside here. This is typically in order of GBs (8-16GB for consumer GPUs). In user code, you allocate globaly memory with `cudaMalloc()` and write data to it with `cudaMemcpy()`.
    - **Shared memory** (SRAM): this is on-chip memory. Each SM has 64kb of shared memory, which is shared among all threads in a block. This can be used as working space for storing intermediate results and collaboration among threads within a block. To use it in kernel code, we either write `__shared__ float shmem[64];` for static-size array, or `extern __shared__ float shmem[];` for dynamically allocated shared memory, which is set by kernel call execution configuration `kernel<<<n_blocks, n_threads, shmem_size>>>`().
    - **Register/local memory**: per-thread memory. This is where the local variables live.
- Accessing memory: (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
    - **Global memory**: to ensure coalesced memory access (i.e. threads in a warp accessing different memory locations with a single instruction), the rule of thumb is to make consecutive threads accessing consecutive memory locations. There is also memory alignment requirement but most of the time it should be fine. A global memory transaction has a maximum of 128 bytes (e.g. 32 fp32 or 32 int32). If a dtype is less than 4 bytes (32 bits) (e.g. uint8, float16), packing them into 4-byte (32-bit) struct (e.g. `char4`, `half2`) should improve memory throughput as we can use 128-byte transactions. Packing data into 8-byte (64-bit) struct (or even larger) would not improve memory throughput, but we will issue fewer instructions.
    - **Shared memory**: to avoid memory bank conflict...
- Typically, we load a tile of data (might be larger than block size) from global memory to shared memory, do computations on shared memory, then write results from shared memory back to global memory. To achieve coalesced memory access (and making the kernel easier to reason about), usually we decouple global memory access from computation: during global memory access, we assign consecutive threads to consecutive data, but during computation, a thread may be responsible for a different data element.
- When writing Triton kernel, always use `tl.dot(allow_tf32=False)`. With fp32 inputs, the outputs are wrong. There are no clear reasons for this. With fp16/bf16, this flag doesn't matter.
