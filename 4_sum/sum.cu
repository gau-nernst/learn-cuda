#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__host__ __device__ int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr int WARP_SIZE = 32;

// Kahan sum to reduce errors
// 1 thread is responsible for 1 row.
__global__ void sum_v1_kernel(const float *input, float *output, int M, int N) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  input += row * N;
  float sum = 0.0f;
  float error = 0.0f;

  for (int col = 0; col < N; col++) {
    float item = input[col] - error;
    float new_sum = sum + item;
    error = new_sum - sum - item;
    sum = new_sum;
  }

  output[row] = sum;
}

void sum_v1(const float *input, float *output, int M, int N, int BLOCK_SIZE) {
  sum_v1_kernel<<<cdiv(M, BLOCK_SIZE), BLOCK_SIZE>>>(input, output, M, N);
}

// parallel sum with shared memory
// each thread block calculates sum for BLOCK_SIZE elements of input
__global__ void sum_v2_kernel(const float *input, float *output, int M, int N) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int col = blockIdx.x * BLOCK_SIZE + tid;
  const int row = blockIdx.y;

  // should have size = BLOCK_SIZE
  extern __shared__ float shmem[];

  input += row * N;

  // load data to shared memory
  shmem[tid] = col < N ? input[col] : 0.0f;
  __syncthreads();

  // parallel sum
  // after each iteration, only half of the remaining threads are active
  // warp divergence only happens when less than a full warp is active
  // no bank conflicts
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  // global synchronization
  // alternative: write to global memory scratch space, and call 1 more reduction kernel
  if (tid == 0)
    atomicAdd(output + row, shmem[0]);
}

void sum_v2(const float *input, float *output, int M, int N, int BLOCK_SIZE) {
  dim3 grid_size(cdiv(N, BLOCK_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  sum_v2_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N);
}

// thread coarsening
// each thread block calculates sum for TILE_SIZE elements of input
// TILE_SIZE must be a multiple of BLOCK_SIZE
__global__ void sum_v3_kernel(const float *input, float *output, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int row = blockIdx.y;
  const int tile_id = blockIdx.x;

  // should have size = BLOCK_SIZE
  extern __shared__ float shmem[];

  input += row * N + tile_id * TILE_SIZE;

  // thread-level reduction
  // load a tile of BLOCK_SIZE one at a time -> coalesced memory access
  float sum = 0.0f;
  for (int col = tid; col < TILE_SIZE; col += BLOCK_SIZE)
    if (col < N - tile_id * TILE_SIZE)
      sum += input[col];

  // store per-thread result in shared memory
  shmem[tid] = sum;
  __syncthreads();

  // block-level reduction
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  // grid-level reduction
  if (tid == 0)
    atomicAdd(output + row, shmem[0]);
}

void sum_v3(const float *input, float *output, int M, int N, int BLOCK_SIZE, int coarse_factor) {
  const int TILE_SIZE = BLOCK_SIZE * coarse_factor;
  dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  sum_v3_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N, TILE_SIZE);
}

// warp-level reduction
// NOTE: block_size must be >= 64 for this kernel
template <int WARP_REDUCTION_IMPL>
__global__ void sum_v4_kernel(const float *input, float *output, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int row = blockIdx.y;
  const int tile_id = blockIdx.x;

  // should have size = BLOCK_SIZE
  extern __shared__ float shmem[];

  input += row * N + tile_id * TILE_SIZE;

  // thread-level reduction
  float sum = 0.0f;
  for (int col = tid; col < TILE_SIZE; col += BLOCK_SIZE) {
    if (col < N - tile_id * TILE_SIZE)
      sum += input[col];
  }
  shmem[tid] = sum;
  __syncthreads();

  // block-level reduction
  // no warp divergence since all threads in a 32-thread warp will either do the addition or not.
  for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }
  sum = shmem[tid];

  // warp-level reduction
  // no synchronization
  if (tid < WARP_SIZE) {
    // approach 0: this won't work, even though all reads and writes are done at the same time (no race condition).
    // this is because the compiler will optimize memory read of shmem[tid + stride] -> cache the data instead of
    // reading the updated shared memory.
    // for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2)
    //   shmem[tid] += shmem[tid + stride];

    // approach 1: cast shared memory as volatile -> compiler will issue a true memory read
    if (WARP_REDUCTION_IMPL == 1) {
      volatile float *_shmem = shmem;
      for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2)
        _shmem[tid] += _shmem[tid + stride];
      sum = _shmem[tid];
    }

    // approach 2: use __syncwarp() -> wait for shared memory update, and issue a true memory read
    if (WARP_REDUCTION_IMPL == 2) {
      for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        shmem[tid] += shmem[tid + stride];
        __syncwarp();
      }
      sum = shmem[tid];
    }

    // approach 3: use warp-level primitives -> register-to-register communication
    if (WARP_REDUCTION_IMPL == 3) {
      for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }

  // grid-level reduction
  if (tid == 0)
    atomicAdd(output + row, sum);
}

void sum_v4a(const float *input, float *output, int M, int N, int BLOCK_SIZE, int coarse_factor) {
  const int TILE_SIZE = BLOCK_SIZE * coarse_factor;
  dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  sum_v4_kernel<1><<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N, TILE_SIZE);
}

void sum_v4b(const float *input, float *output, int M, int N, int BLOCK_SIZE, int coarse_factor) {
  const int TILE_SIZE = BLOCK_SIZE * coarse_factor;
  dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  sum_v4_kernel<2><<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N, TILE_SIZE);
}

void sum_v4c(const float *input, float *output, int M, int N, int BLOCK_SIZE, int coarse_factor) {
  const int TILE_SIZE = BLOCK_SIZE * coarse_factor;
  dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  sum_v4_kernel<3><<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N, TILE_SIZE);
}

// cooperative group
__global__ void sum_v5_kernel(const float *input, float *output, int m, int n, int coarse_factor) {
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

void sum_v5(const float *input, float *output, int m, int n, int block_size, int coarse_factor) {
  dim3 grid_size(cdiv(n, block_size * coarse_factor), m);
  int shmem_size = sizeof(float) * block_size;
  sum_v5_kernel<<<grid_size, block_size, shmem_size>>>(input, output, m, n, coarse_factor);
}

// vectorized load. n must be divisible by 4
__global__ void sum_v6_kernel(const float *input, float *output, int m, int n, int coarse_factor) {
  cg::thread_block block = cg::this_thread_block();

  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  extern __shared__ float shmem[];
  input += row * n;

  // thread-level reduction w/ vectorized load
  float sum = 0.0f;
  for (int tile = 0; tile < coarse_factor; tile++) {
    int input_col = (blockIdx.x * coarse_factor + tile) * blockDim.x + tid;
    if (input_col < n / 4) {
      float4 in = reinterpret_cast<const float4 *>(input)[input_col];
      sum += in.x + in.y + in.z + in.w;
    }
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

void sum_v6(const float *input, float *output, int m, int n, int block_size, int coarse_factor) {
  dim3 grid_size(cdiv(n, block_size * coarse_factor * 4), m);
  int shmem_size = sizeof(float) * block_size;
  sum_v6_kernel<<<grid_size, block_size, shmem_size>>>(input, output, m, n, coarse_factor);
}
