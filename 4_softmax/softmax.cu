__host__ __device__ int cdiv(int a, int b) { return (a + b - 1) / b; }
__device__ float add(float a, float b) { return a + b; }

constexpr int WARP_SIZE = 32;

template <float f(float, float)>
__device__ float thread_reduce(float val, const float *input, int TILE_SIZE, int BLOCK_SIZE, int tid, int max_idx) {
  for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE)
    if (idx < max_idx)
      val = f(val, input[idx]);
  return val;
}

template <float f(float, float)>
__device__ float block_reduce(float val, int BLOCK_SIZE, int tid, float *reduce_space) {
  reduce_space[tid] = val;
  for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      val = f(val, reduce_space[tid + stride]);
      reduce_space[tid] = val;
    }
  }
  return val;
}

template <float f(float, float)>
__device__ float warp_reduce(float val, int tid) {
  if (tid < WARP_SIZE)
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2)
      val = f(val, __shfl_down_sync(0xffffffff, val, stride));
  return val;
}

__device__ float block_broadcast(float val, int tid, float *shmem) {
  if (tid == 0)
    shmem[0] = val;
  __syncthreads();
  return shmem[0];
}

// https://stackoverflow.com/a/72461459
// when val > 0, use atomicMax signed int. in sint representation:
//   - -ve float < +ve float.
//   - less +ve float < more +ve float.
// when val < 0, use atomicMin unsigned int. in uint representation:
//   - +ve float < -ve float.
//   - less -ve float < more -ve float.
// we use !signbit(value) instead of (value > 0) because there is -0 in float.
__device__ float atomicMax(float *address, float val) {
  return !signbit(val) ? 
    __int_as_float(atomicMax(reinterpret_cast<int *>(address), __float_as_int(val))) :
    __uint_as_float(atomicMin(reinterpret_cast<unsigned int*>(address), __float_as_uint(val)));
}

__global__ void softmax_v1_kernel_pass1(const float *input, float *max_space, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  input += row * N + tile_id * TILE_SIZE;
  max_space += row;

  extern __shared__ float reduce_space[];

  float max_val = -INFINITY;
  max_val = thread_reduce<max>(max_val, input, TILE_SIZE, BLOCK_SIZE, tid, N);
  max_val = block_reduce<max>(max_val, BLOCK_SIZE, tid, reduce_space);
  max_val = warp_reduce<max>(max_val, tid);

  // TODO: atomicMax is not implemented for float
  if (tid == 0)
    atomicMax(max_space, max_val);
}

__global__ void softmax_v1_kernel_pass2(const float *input, float *output, float *max_space, float *normalizer_space, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  input += row * N + tile_id * TILE_SIZE;
  output += row * N + tile_id * TILE_SIZE;
  max_space += row;
  normalizer_space += row;

  extern __shared__ float reduce_space[];

  float sum = 0.0f;
  float subtract = max_space[0];
  for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE)
    if (idx < N) {
      float val = exp(input[idx] - subtract);
      output[idx] = val;
      sum += val;
    }
  sum = block_reduce<add>(sum, BLOCK_SIZE, tid, reduce_space);
  sum = warp_reduce<add>(sum, tid);
  if (tid == 0)
    atomicAdd(normalizer_space, sum);
}

__global__ void softmax_v1_kernel_pass3(float *output, float *normalizer_space, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  output += row * N + tile_id * TILE_SIZE;
  normalizer_space += row;

  float scale = 1.0f / normalizer_space[0];
  for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE)
    if (idx < N)
      output[idx] *= scale;
}

void softmax_v1(const float *input, float *output, float *workspace, int M, int N) {
  // need extra 2M space to store max per row and sum per row
  const int BLOCK_SIZE = 256;
  const int TILE_SIZE = BLOCK_SIZE * 4;
  const dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int reduce_space_size = sizeof(float) * BLOCK_SIZE;

  float *max_space = workspace;
  float *normalizer_space = workspace + M;

  // pass 1: max per row
  softmax_v1_kernel_pass1<<<grid_size, BLOCK_SIZE, reduce_space_size>>>(input, max_space, M, N, TILE_SIZE);

  // pass 2: exp(x - max) and sum
  softmax_v1_kernel_pass2<<<grid_size, BLOCK_SIZE, reduce_space_size>>>(input, output, max_space, normalizer_space, M, N, TILE_SIZE);

  // pass 3: normalize
  softmax_v1_kernel_pass3<<<grid_size, BLOCK_SIZE>>>(output, normalizer_space, M, N, TILE_SIZE);
}

template <bool STORE_INTERMEDIATE>
__global__ void softmax_v2_kernel(const float *input, float *output, int M, int N) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int row = blockIdx.y;

  input += row * N;
  output += row * N;

  extern __shared__ float shmem_reduce[];

  // pass 1: find max
  float max_val = -INFINITY;
  max_val = thread_reduce<max>(max_val, input, N, BLOCK_SIZE, tid, N);
  max_val = block_reduce<max>(max_val, BLOCK_SIZE, tid, shmem_reduce);
  max_val = warp_reduce<max>(max_val, tid);
  max_val = block_broadcast(max_val, tid, shmem_reduce);

  // pass 2: subtract max and apply exponential + find sum
  float sum = 0.0f;
  for (int col = tid; col < N; col += BLOCK_SIZE) {
    float val = exp(input[col] - max_val);
    sum += val;
    if (STORE_INTERMEDIATE)
      output[col] = val;
  }
  sum = block_reduce<add>(sum, BLOCK_SIZE, tid, shmem_reduce);
  sum = warp_reduce<add>(sum, tid);
  sum = block_broadcast(sum, tid, shmem_reduce);

  // pass 3: normalize
  // NOTE: if N is small, we can cache exp(input[col] - max_val) in shared memory
  float normalizer = 1.0f / sum;
  for (int col = tid; col < N; col += BLOCK_SIZE)
    output[col] = (STORE_INTERMEDIATE ? output[col] : exp(input[col] - max_val)) * normalizer;
}

void softmax_v2a(const float *input, float *output, int M, int N) {
  const int BLOCK_SIZE = 1024;
  const dim3 grid_size(1, M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  softmax_v2_kernel<false><<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N);
}

void softmax_v2b(const float *input, float *output, int M, int N) {
  const int BLOCK_SIZE = 1024;
  const dim3 grid_size(1, M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  softmax_v2_kernel<true><<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N);
}

__global__ void softmax_v3_kernel_pass1(const float *input, float *workspace, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  input += row * N + tile_id * TILE_SIZE;
  float *max_space = workspace + row;
  float *normalizer_space = workspace + M + row;

  extern __shared__ float max_shared[];
  float *normalizer_shared = max_shared + BLOCK_SIZE;

  // algorithm 3 in https://arxiv.org/pdf/1805.02867
  float max_val = -INFINITY;
  float normalizer = 0.0f;
  for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE) {
    float val = input[idx];
    float old_max_val = max_val;
    max_val = max(max_val, val);
    normalizer = normalizer * exp(old_max_val - max_val) + exp(val - max_val);
  }

  // equation 4, section 3.1 in https://arxiv.org/pdf/1805.02867
  max_shared[tid] = max_val;
  normalizer_shared[tid] = normalizer;
  for (int stride = BLOCK_SIZE / 2; stride < WARP_SIZE; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      float other_max = max_shared[tid + stride];
      float other_normalizer = normalizer_shared[tid + stride];

      float new_max = max(max_val, other_max);
      normalizer = normalizer * exp(max_val - new_max) + other_normalizer * exp(other_max - new_max);
      max_val = new_max;
      max_shared[tid] = new_max;
      normalizer_shared[tid] = normalizer;
    }
  }

  if (tid < WARP_SIZE)
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      float other_max = __shfl_down_sync(0xffffffff, max_val, stride);
      float other_normalizer = __shfl_down_sync(0xffffffff, normalizer, stride);

      float new_max = max(max_val, other_max);
      normalizer = normalizer * exp(max_val - new_max) + other_normalizer * exp(other_max - new_max);
      max_val = new_max;
    }

  if (tid == 0) {
    float other_max = atomicMax(max_space, max_val);
    float new_max = max(max_val, other_max);

    // using atomicCAS is slow
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    float other_normalizer = normalizer_space[0];
    float assumed;
    do {
      assumed = other_normalizer;
      float new_normalizer = normalizer * exp(max_val - new_max) + other_normalizer * exp(other_max - new_max);
      other_normalizer = atomicCAS(reinterpret_cast<int *>(normalizer_space), __float_as_int(other_normalizer), __float_as_int(new_normalizer));
    } while (assumed != other_normalizer);
  }
}

__global__ void softmax_v3_kernel_pass2(const float *input, float *output, const float *workspace, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  input += row * N + tile_id * TILE_SIZE;
  output += row * N + tile_id * TILE_SIZE;
  float row_max  = workspace[row];
  float scale = 1.0f / workspace[M + row];

  for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE)
    if (idx < N)
      output[idx] = exp(input[idx] - row_max) * scale;
}

// online softmax
void softmax_v3(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int TILE_SIZE = BLOCK_SIZE * 4;
  const dim3 grid_size(cdiv(N, BLOCK_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE * 2;

  // pass 1: find max and normalizer at the same time
  softmax_v3_kernel_pass1<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, workspace, M, N, TILE_SIZE);

  // pass 2: calculate output
  softmax_v3_kernel_pass2<<<grid_size, BLOCK_SIZE>>>(input, output, workspace, M, N, TILE_SIZE);
}
