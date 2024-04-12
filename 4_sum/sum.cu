#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define cdiv(a, b) ((a) + (b)-1) / (b)

// Kahan sum to reduce errors
// 1 thread is responsible for 1 row.
__global__ void sum_v1_kernel(const float *input, float *output, int m, int n) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m)
    return;

  input += row * n;
  float sum = 0.0f;
  float error = 0.0f;

  for (int col = 0; col < n; col++) {
    float item = input[col] - error;
    float new_sum = sum + item;
    error = new_sum - sum - item;
    sum = new_sum;
  }

  output[row] = sum;
}

void sum_v1(const float *input, float *output, int m, int n, int block_size) {
  int n_blocks = cdiv(m, block_size);
  sum_v1_kernel<<<n_blocks, block_size>>>(input, output, m, n);
}

// parallel sum with shared memory
__global__ void sum_v2_kernel(const float *input, float *output, int m, int n) {
  const int tid = threadIdx.x;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  extern __shared__ float shmem[];

  // load data to shared memory
  shmem[tid] = col < n ? input[row * n + col] : 0.0f;
  __syncthreads();

  // parallel sum
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(output + row, shmem[0]);
}

void sum_v2(const float *input, float *output, int m, int n, int block_size) {
  dim3 n_blocks(cdiv(n, block_size), m);
  int shmem_size = sizeof(float) * block_size;
  sum_v2_kernel<<<n_blocks, block_size, shmem_size>>>(input, output, m, n);
}

// thread coarsening and warp-level reduction
// NOTE: block_size must be >= 64 for this kernel
__global__ void sum_v3_kernel(const float *input, float *output, int m, int n, int coarse_factor) {
  const int tid = threadIdx.x;
  const int row = blockIdx.y;
  extern __shared__ float shmem[];
  input += row * n;

  // reduction within a thread
  // store results to shared memory
  float sum = 0.0f;
  for (int tile = 0; tile < coarse_factor; tile++) {
    int input_col = (blockIdx.x * coarse_factor + tile) * blockDim.x + threadIdx.x;
    if (input_col < n)
      sum += input[input_col];
  }
  shmem[tid] = sum;
  __syncthreads();

  // reduction within a block
  // no warp divergence since all threads in a 32-thread warp will either do the addition or not.
  for (int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }
  sum = shmem[tid];

  // reduction within a warp
  if (tid < 32) {
    // approach 0: this won't work, even though all reads and writes are done at the same time (no race condition).
    // this is because the compiler will optimize memory read of shmem[tid + stride] -> cache the data instead of
    // reading the updated shared memory.
    // for (int stride = 16; stride > 0; stride /= 2)
    //   shmem[tid] += shmem[tid + stride];

    // approach 1: cast shared memory as volatile -> compiler will issue a true memory read
    // volatile float *_shmem = shmem;
    // for (int stride = 16; stride > 0; stride /= 2)
    //   _shmem[tid] += _shmem[tid + stride];

    // approach 2: use __syncwarp() -> wait for shared memory update, and issue a true memory read
    // for (int stride = 16; stride > 0; stride /= 2) {
    //   shmem[tid] += shmem[tid + stride];
    //   __syncwarp();
    // }

    // approach 3: use warp-level primitives -> register-to-register communication
    for (int offset = 16; offset > 0; offset /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  // reduction across blocks
  // alternatives:
  // - write to global memory, and call 1 more reduction kernel
  // - thread fence reduction
  // - cooperative groups
  if (tid == 0)
    atomicAdd(output + row, sum);
}

void sum_v3(const float *input, float *output, int m, int n, int block_size, int coarse_factor) {
  dim3 grid_size(cdiv(n, block_size * coarse_factor), m);
  int shmem_size = sizeof(float) * block_size;
  sum_v3_kernel<<<grid_size, block_size, shmem_size>>>(input, output, m, n, coarse_factor);
}

// cooperative group
__global__ void sum_v4_kernel(const float *input, float *output, int m, int n, int coarse_factor) {
  cg::thread_block block = cg::this_thread_block();

  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  extern __shared__ float shmem[];
  input += row * n;

  // thread-level reduction
  float sum = 0.0f;
  for (int tile = 0; tile < coarse_factor; tile++) {
    int input_col = (blockIdx.x * coarse_factor + tile) * blockDim.x + tid;
    if (input_col < n)
      sum += input[input_col];
  }

  for (int stride = block.size() / 2; stride >= 32; stride /= 2) {
    shmem[tid] = sum;
    block.sync();
    if (tid < stride)
      sum += shmem[tid + stride];
    block.sync();
  }

  cg::thread_block_tile warp = cg::tiled_partition<32>(block);
  sum = cg::reduce(warp, sum, cg::plus<float>());

  // grid-level reduction
  if (block.thread_rank() == 0)
    atomicAdd(output + row, sum);
}

void sum_v4(const float *input, float *output, int m, int n, int block_size, int coarse_factor) {
  dim3 grid_size(cdiv(n, block_size * coarse_factor), m);
  int shmem_size = sizeof(float) * block_size;
  sum_v4_kernel<<<grid_size, block_size, shmem_size>>>(input, output, m, n, coarse_factor);
}
