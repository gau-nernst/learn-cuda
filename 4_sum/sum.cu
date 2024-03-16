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
    shmem[tid] += tid < stride ? shmem[tid + stride] : 0.0f;
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

// thread coarsening -> increase compute load for each thread
__global__ void sum_v3_kernel(const float *input, float *output, int m, int n, int coarse_factor) {
  const int tid = threadIdx.x;
  const int col_idx = blockIdx.x * blockDim.x * coarse_factor + threadIdx.x;
  const int row_idx = blockIdx.y;
  extern __shared__ float shmem[];

  // reduction within a thread
  float sum = 0.0f;
  for (int tile = 0; tile < coarse_factor; tile++) {
    int new_col_idx = col_idx + blockDim.x * tile;
    sum += new_col_idx < n ? input[row_idx * n + new_col_idx] : 0.0f;
  }
  shmem[tid] = sum;
  __syncthreads();

  // reduction within a block
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    shmem[tid] += tid < stride ? shmem[tid + stride] : 0.0f;
    __syncthreads();
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
