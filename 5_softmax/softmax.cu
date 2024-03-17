#define cdiv(a, b) ((a) + (b)-1) / (b)

__device__ inline float add(float a, float b) { return a + b; }

template <float f(float, float)> __device__ void block_reduce(int block_size, int tid, float *shmem) {
  for (int stride = block_size / 2; stride > 32; stride /= 2) {
    if (tid < stride)
      shmem[tid] = f(shmem[tid], shmem[tid + stride]);
    __syncthreads();
  }
}

template <float f(float, float)> __device__ void warp_reduce(int tid, float *shmem) {
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
__global__ void softmax_v1_kernel(const float *input, float *output, int m, int n) {
  const int tid = threadIdx.x;
  const int row_idx = blockIdx.y;
  extern __shared__ float shmem_reduce[];
  float *shmem_elems = shmem_reduce + blockDim.x;

  // load data to shared memory and perform thread-level min
  // each thread will load tid, tid + block_size, ... to ensure coalesce memory access
  float max_val = -INFINITY;
  for (int tile = 0; tile < cdiv(n, blockDim.x); tile++) {
    int col_idx = tile * blockDim.x + tid;
    if (col_idx < n) {
      float val = input[row_idx * n + col_idx];
      max_val = max(max_val, val);
      shmem_elems[col_idx] = val;
    }
  }

  shmem_reduce[tid] = max_val;
  __syncthreads();
  block_reduce<max>(blockDim.x, tid, shmem_reduce);
  if (tid < 32)
    warp_reduce<max>(tid, shmem_reduce);
  __syncthreads();

  // subtract max and apply exponential. also perform thread-level sum
  max_val = shmem_reduce[0];
  float sum = 0.0f;
  for (int tile = 0; tile < cdiv(n, blockDim.x); tile++) {
    int col_idx = tile * blockDim.x + tid;
    if (col_idx < n) {
      float val = __expf(shmem_elems[col_idx] - max_val);
      sum += val;
      shmem_elems[col_idx] = val;
    }
  }

  shmem_reduce[tid] = sum;
  block_reduce<add>(blockDim.x, tid, shmem_reduce);
  if (tid < 32)
    warp_reduce<add>(tid, shmem_reduce);
  __syncthreads();

  float normalizer = 1.0f / shmem_reduce[0];
  for (int tile = 0; tile < cdiv(n, blockDim.x); tile++) {
    int col_idx = tile * blockDim.x + tid;
    if (col_idx < n)
      output[row_idx * n + col_idx] = shmem_elems[col_idx] * normalizer;
  }
}

void softmax_v1_launch(const float *input, float *output, int m, int n, int block_size) {
  dim3 grid_size(1, m);
  int shmem_size = sizeof(float) * (block_size + n);
  softmax_v1_kernel<<<grid_size, block_size, shmem_size>>>(input, output, m, n);
}
