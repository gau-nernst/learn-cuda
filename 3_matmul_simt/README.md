# Matrix Multiplication

Resources:
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
- https://siboehm.com/articles/22/CUDA-MMM
- https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/ and https://github.com/leimao/CUDA-GEMM-Optimization/
- https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md
- https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

For M = N = K = 4096, 4070Ti SUPER, compile with `-O3 --use_fast_math`

Kernel name                                                        | Latency (ms) | % of CuBLAS | Bandwidth (GB/s)
-------------------------------------------------------------------|--------------|-------------|------------------
CuBLAS (via PyTorch) `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` |         4.77 |     100.00% |           104.25
v1 (naive 1 row dot 1 column)                                      |        56.21 |       8.49% |           195.98
v2 (shared memory cache with 2D block tiling)                      |        48.14 |       9.91% |           179.61
v3 (thread coarsening)                                             |        39.03 |      12.22% |            38.49
v4 (register cache with 2D thread tiling)                          |         8.92 |      53.48% |            76.43
v5 (warp tiling)                                                   |         8.83 |      54.02% |           140.26
v6a (remove bounds check. vectorized global memory access)         |         6.60 |      72.27% |           320.89
v6b (transpose A in shared memory)                                 |         5.66 |      84.28% |           202.24

## CuBLAS `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`

ncu output of an ideal kernel

```
NOTE: block_size=256, memory throughput ~40% (compute bound).
  void cutlass::Kernel<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params) (256, 2, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- -------------
    Metric Name               Metric Unit  Metric Value
    ----------------------- ------------- -------------
    DRAM Frequency          cycle/nsecond         10.24
    SM Frequency            cycle/nsecond          2.34
    Elapsed Cycles                  cycle    11,161,801
    Memory Throughput                   %         42.78
    DRAM Throughput                     %         15.44
    Duration                      msecond          4.77
    L1/TEX Cache Throughput             %         44.29
    L2 Cache Throughput                 %         22.56
    SM Active Cycles                cycle 10,781,032.98
    Compute (SM) Throughput             %         80.97
    ----------------------- ------------- -------------

    INF   The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To   
          further improve performance, work will likely need to be shifted from the most utilized to another unit.      
          Start by analyzing workloads in the Compute Workload Analysis section.                                        

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 64:1. The kernel achieved 73%   
          of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide    
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline      
          analysis.                                                                                                     

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte         2.62
    Dropped Samples                sample            0
    Maximum Sampling Interval     msecond         1.02
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         3.35
    Executed Ipc Elapsed  inst/cycle         3.24
    Issue Slots Busy               %        83.83
    Issued Ipc Active     inst/cycle         3.35
    SM Busy                        %        83.83
    -------------------- ----------- ------------

    OPT   FMA is the highest-utilized pipeline (75.9%) based on active cycles, taking into account the rates of its     
          different instructions. It executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD)    
          operations. The pipeline is well-utilized, but might become a bottleneck if more work is added. Based on the  
          number of executed instructions, the highest utilized pipeline (75.7%) is FMA. It executes 32-bit floating    
          point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations. Comparing the two, the overall pipeline    
          utilization appears to be caused by frequent, low-latency instructions. See the Kernel Profiling Guide        
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder) or hover over the          
          pipeline name to understand the workloads handled by each pipeline. The Instruction Statistics section shows  
          the mix of executed instructions in this kernel.                                                              

    Section: Memory Workload Analysis
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second       101.21
    Mem Busy                               %        42.78
    Max Bandwidth                          %        35.22
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %        90.47
    Mem Pipes Busy                         %        35.22
    --------------------------- ------------ ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        83.79
    Issued Warp Per Scheduler                        0.84
    No Eligible                            %        16.21
    Active Warps Per Scheduler          warp         2.00
    Eligible Warps Per Scheduler        warp         1.50
    ---------------------------- ----------- ------------

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle         2.39
    Warp Cycles Per Executed Instruction           cycle         2.39
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    31.90
    ---------------------------------------- ----------- ------------

    Section: Instruction Statistics
    ---------------------------------------- ----------- -------------
    Metric Name                              Metric Unit  Metric Value
    ---------------------------------------- ----------- -------------
    Avg. Executed Instructions Per Scheduler        inst     9,037,312
    Executed Instructions                           inst 2,385,850,368
    Avg. Issued Instructions Per Scheduler          inst     9,037,358
    Issued Instructions                             inst 2,385,862,512
    ---------------------------------------- ----------- -------------

NOTE: Used 49KB of shared memory, 216 registers.
    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                    512
    Registers Per Thread             register/thread             216
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block           49.15
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              66
    Threads                                   thread         131,072
    Uses Green Context                                             0
    Waves Per SM                                                7.76
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block            1
    Block Limit Shared Mem                block            1
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp            8
    Theoretical Occupancy                     %        16.67
    Achieved Occupancy                        %        16.66
    Achieved Active Warps Per SM           warp         8.00
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 83.33%                                                                                    
          The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required      
          registers. This kernel's theoretical occupancy (16.7%) is limited by the required amount of shared memory.    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle     7,544,548
    Total DRAM Elapsed Cycles        cycle   390,899,712
    Average L1 Active Cycles         cycle 10,781,032.98
    Total L1 Elapsed Cycles          cycle   736,652,446
    Average L2 Active Cycles         cycle  9,381,469.83
    Total L2 Elapsed Cycles          cycle   228,416,256
    Average SM Active Cycles         cycle 10,781,032.98
    Total SM Elapsed Cycles          cycle   736,652,446
    Average SMSP Active Cycles       cycle 10,785,407.96
    Total SMSP Elapsed Cycles        cycle 2,946,609,784
    -------------------------- ----------- -------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.00
    Branch Instructions              inst    2,134,016
    Branch Efficiency                   %          100
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------

    OPT   Est. Speedup: 26.01%                                                                                          
          This kernel has uncoalesced shared accesses resulting in a total of 83886080 excessive wavefronts (27% of the 
          total 311476253 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source            
          locations. The CUDA Best Practices Guide                                                                      
           (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c
          -ab) has an example on optimizing shared memory accesses.                                                     
```
