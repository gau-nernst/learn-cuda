// Kahan sum to reduce errors
__global__ void sum_kernel_v1(const float *input, float *output, int m, int n) {
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

template <int BLOCK_SIZE>
__global__ void sum_kernel_v2(const float *input, float *output, int m, int n) {
  const int tid = threadIdx.x;
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y;
  __shared__ float shmem[BLOCK_SIZE];

  // load data to shared memory
  shmem[tid] = col_idx < n ? input[row_idx * n + col_idx] : 0.0f;
  __syncthreads();

  // parallel sum
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(output + row_idx, shmem[0]);
}

template __global__ void sum_kernel_v2<256>(const float *input, float *output, int m, int n);
template __global__ void sum_kernel_v2<512>(const float *input, float *output, int m, int n);
template __global__ void sum_kernel_v2<1024>(const float *input, float *output, int m, int n);

template <int BLOCK_SIZE, int COARSE_FACTOR>
__global__ void sum_kernel_v3(const float *input, float *output, int m, int n) {
  const int tid = threadIdx.x;
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y;
  __shared__ float shmem[BLOCK_SIZE];

  float sum = 0.0f;
  for (int tile = 0; tile < COARSE_FACTOR; tile++)
    sum += col_idx + BLOCK_SIZE < n ? input[row_idx * n + col_idx + BLOCK_SIZE] : 0.0f;

  // load data to shared memory
  shmem[tid] = sum;
  __syncthreads();

  // parallel sum
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shmem[tid] += shmem[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(output + row_idx, shmem[0]);
}

template __global__ void sum_kernel_v3<256>(const float *input, float *output, int m, int n);
template __global__ void sum_kernel_v3<512>(const float *input, float *output, int m, int n);
template __global__ void sum_kernel_v3<1024>(const float *input, float *output, int m, int n);
