__host__ __device__ int cdiv(int a, int b) { return (a + b - 1) / b; }
__device__ float add(float a, float b) { return a + b; }

template <float f(float, float)>
__device__ void block_reduce(int BLOCK_SIZE, int tid, float *shmem) {
  for (int stride = BLOCK_SIZE / 2; stride > 32; stride /= 2) {
    if (tid < stride)
      shmem[tid] = f(shmem[tid], shmem[tid + stride]);
    __syncthreads();
  }
}

template <float f(float, float)>
__device__ void warp_reduce(int tid, float *shmem) {
  float val = f(shmem[tid], shmem[tid + 32]);
  for (int offset = 16; offset > 0; offset /= 2)
    val = f(val, __shfl_down_sync(0xffffffff, val, offset));
  shmem[tid] = val;
}

// 1. find max
// 2. subtract max and take exponential
// 3. find sum
// 4. divide by sum
// assume we can fit 1 row (n elements) in shared memory (48kb -> 12k floats)
__global__ void mini_softmax_kernel(const float *input, float *output, int M, int N) {
  const int tid = threadIdx.x;
  const int row = blockIdx.y;
  const int BLOCK_SIZE = blockDim.x;

  extern __shared__ float shmem_reduce[];          // size block_size
  float *shmem_elems = shmem_reduce + BLOCK_SIZE;  // size n

  // load data to shared memory and perform thread-level max
  // each thread will load tid, tid + block_size, ... to ensure coalesce memory access
  float max_val = -INFINITY;
  for (int col = tid; col < N; col += BLOCK_SIZE) {
    float val = input[row * N + col];
    max_val = max(max_val, val);
    shmem_elems[col] = val;
  }

  shmem_reduce[tid] = max_val;
  __syncthreads();

  block_reduce<max>(BLOCK_SIZE, tid, shmem_reduce);

  if (tid < 32)
    warp_reduce<max>(tid, shmem_reduce);
  __syncthreads();

  // subtract max and apply exponential. also perform thread-level sum
  max_val = shmem_reduce[0];
  float sum = 0.0f;
  for (int col = tid; col < N; col += BLOCK_SIZE) {
    float val = expf(shmem_elems[col] - max_val);
    sum += val;
    shmem_elems[col] = val;
  }

  shmem_reduce[tid] = sum;
  __syncthreads();

  block_reduce<add>(BLOCK_SIZE, tid, shmem_reduce);

  if (tid < 32)
    warp_reduce<add>(tid, shmem_reduce);
  __syncthreads();

  float normalizer = 1.0f / shmem_reduce[0];
  for (int col = 0; col < N; col += BLOCK_SIZE) {
    output[row * N + col] = shmem_elems[col] * normalizer;
  }
}

void mini_softmax(const float *input, float *output, int M, int N, int BLOCK_SIZE) {
  dim3 grid_size(1, M);
  int shmem_size = sizeof(float) * (BLOCK_SIZE + N);
  mini_softmax_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N);
}
