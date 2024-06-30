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

// thread-level reduction
// load a tile of BLOCK_SIZE one at a time -> coalesced memory access
__device__ float thread_sum(const float *input, int TILE_SIZE, int BLOCK_SIZE, int tid, int max_idx) {
  float sum = 0.0f;
  for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE)
    if (idx < max_idx)
      sum += input[idx];
  return sum;
}

// thread coarsening
// each thread block calculates sum for TILE_SIZE elements of input
// TILE_SIZE must be a multiple of BLOCK_SIZE
__global__ void sum_v3_kernel(const float *input, float *output, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  // should have size = BLOCK_SIZE
  extern __shared__ float shmem[];
  input += row * N + tile_id * TILE_SIZE;

  // store per-thread result in shared memory
  shmem[tid] = thread_sum(input, TILE_SIZE, BLOCK_SIZE, tid, N - tile_id * TILE_SIZE);
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
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  // should have size = BLOCK_SIZE
  extern __shared__ float shmem[];
  input += row * N + tile_id * TILE_SIZE;

  shmem[tid] = thread_sum(input, TILE_SIZE, BLOCK_SIZE, tid, N - tile_id * TILE_SIZE);
  __syncthreads();

  // block-level reduction
  // no warp divergence since all threads in a 32-thread warp will either do the addition or not.
  for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  // warp-level reduction
  float sum;
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
      sum = shmem[tid];
      for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, stride);
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
__global__ void sum_v5_kernel(const float *input, float *output, int M, int N, int TILE_SIZE) {
  cg::thread_block block = cg::this_thread_block();

  const int tid = block.thread_index().x;
  const int BLOCK_SIZE = block.size();
  const int tile_id = block.group_index().x;
  const int row = block.group_index().y;

  // should have size = BLOCK_SIZE
  extern __shared__ float shmem[];
  input += row * N + tile_id * TILE_SIZE;

  shmem[tid] = thread_sum(input, TILE_SIZE, BLOCK_SIZE, tid, N - tile_id * TILE_SIZE);
  block.sync();

  // block-level reduction
  for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    block.sync();
  }

  // warp-level reduction
  float sum;
  if (tid < WARP_SIZE) {
    sum = shmem[tid];
    sum = cg::reduce(cg::tiled_partition<WARP_SIZE>(block), sum, cg::plus<float>());
  }

  // grid-level reduction
  if (block.thread_rank() == 0)
    atomicAdd(output + row, sum);
}

void sum_v5(const float *input, float *output, int M, int N, int BLOCK_SIZE, int coarse_factor) {
  const int TILE_SIZE = BLOCK_SIZE * coarse_factor;
  dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  sum_v5_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N, TILE_SIZE);
}

// vectorized load. N must be divisible by 4
// TILE_SIZE must be at least 4x larger than BLOCK_SIZE
__global__ void sum_v6_kernel(const float *input, float *output, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  extern __shared__ float shmem[];
  input += row * N + tile_id * TILE_SIZE;

  // thread-level reduction w/ vectorized load
  float sum = 0.0f;
  for (int idx = tid * 4; idx < TILE_SIZE; idx += BLOCK_SIZE * 4)
    if (idx < N - tile_id * TILE_SIZE) {
      float4 tmp = reinterpret_cast<const float4 *>(&input[idx])[0];
      sum += tmp.x + tmp.y + tmp.z + tmp.w;
    }
  shmem[tid] = sum;
  __syncthreads();

  // block-level reduction
  for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  // warp-level reduction
  if (tid < WARP_SIZE) {
    sum = shmem[tid];
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, stride);
  }

  // grid-level reduction
  if (tid == 0)
    atomicAdd(output + row, sum);
}

void sum_v6(const float *input, float *output, int M, int N, int BLOCK_SIZE, int coarse_factor) {
  const int TILE_SIZE = BLOCK_SIZE * coarse_factor;
  dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  sum_v6_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N, TILE_SIZE);
}
