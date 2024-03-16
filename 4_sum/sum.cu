#define cdiv(a, b) ((a) + (b)-1) / (b)

// Kahan sum to reduce errors
// 1 thread is responsible for 1 row.
__global__ void sum_v1_kernel(const float *input, float *output, int m, int n) {
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx >= m)
    return;

  float sum = 0.0f;
  float error = 0.0f;

  for (int i = 0; i < n; i++) {
    float item = input[row_idx * n + i] - error;
    float new_sum = sum + item;
    error = new_sum - sum - item;
    sum = new_sum;
  }

  output[row_idx] = sum;
}

void sum_v1_launch(const float *input, float *output, int m, int n, int block_size) {
  int n_blocks = cdiv(m, block_size);
  sum_v1_kernel<<<n_blocks, block_size>>>(input, output, m, n);
}

// parallel sum with shared memory
__global__ void sum_v2_kernel(const float *input, float *output, int m, int n) {
  const int tid = threadIdx.x;
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y;
  extern __shared__ float shmem[];

  // load data to shared memory
  shmem[tid] = col_idx < n ? input[row_idx * n + col_idx] : 0.0f;
  __syncthreads();

  // parallel sum
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(output + row_idx, shmem[0]);
}

void sum_v2_launch(const float *input, float *output, int m, int n, int block_size) {
  dim3 n_blocks(cdiv(n, block_size), m);
  int shmem_size = sizeof(float) * block_size;
  sum_v2_kernel<<<n_blocks, block_size, shmem_size>>>(input, output, m, n);
}

// thread coarsening and warp-level reduction
// NOTE: block_size must be >= 64 for this kernel
__global__ void sum_v3_kernel(const float *input, float *output, int m, int n, int coarse_factor) {
  const int tid = threadIdx.x;
  const int col_idx = blockIdx.x * blockDim.x * coarse_factor + threadIdx.x;
  const int row_idx = blockIdx.y;
  extern __shared__ float shmem[];

  // reduction within a thread
  // store results to shared memory
  float sum = 0.0f;
  for (int tile = 0; tile < coarse_factor; tile++) {
    int new_col_idx = col_idx + blockDim.x * tile;
    if (new_col_idx < n)
      sum += input[row_idx * n + new_col_idx];
  }
  shmem[tid] = sum;
  __syncthreads();

  // reduction within a block
  // no warp divergence since all threads in a 32-thread warp will either do the addition or not.
  for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  // reduction within a warp
  if (tid < 32) {
    // approach 0: this won't work, even though all reads and writes are done at the same time (no race condition).
    // this is because the compiler will optimize memory read of shmem[tid + stride] -> cache the data instead of
    // reading the updated shared memory.
    // for (int stride = 32; stride > 0; stride /= 2)
    //   shmem[tid] += shmem[tid + stride];

    // approach 1: cast shared memory as volatile -> compiler will issue a true memory read
    // volatile float *_shmem = shmem;
    // for (int stride = 32; stride > 0; stride /= 2)
    //   _shmem[tid] += _shmem[tid + stride];

    // approach 2: use __syncwarp() -> wait for shared memory update, and issue a true memory read
    // for (int stride = 32; stride > 0; stride /= 2) {
    //   shmem[tid] += shmem[tid + stride];
    //   __syncwarp();
    // }

    // approach 3: use warp-level primitives
    float val = shmem[tid] + shmem[tid + 32];
    for (int offset = 16; offset > 0; offset /= 2)
      val += __shfl_down_sync(0xffffffff, val, offset);
    shmem[tid] = val;
  }

  // reduction across blocks
  if (tid == 0)
    atomicAdd(output + row_idx, shmem[0]);
}

void sum_v3_launch(const float *input, float *output, int m, int n, int block_size, int coarse_factor) {
  dim3 n_blocks(cdiv(n, block_size * coarse_factor), m);
  int shmem_size = sizeof(float) * block_size;
  sum_v3_kernel<<<n_blocks, block_size, shmem_size>>>(input, output, m, n, coarse_factor);
}
