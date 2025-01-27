__host__ __device__ int cdiv(int a, int b) { return (a + b - 1) / b; }
__device__ float add(float a, float b) { return a + b; }

constexpr int WARP_SIZE = 32;

template <float f(float, float)>
__device__ float thread_reduce(float val, const float *input, int TILE_SIZE, int BLOCK_SIZE, int tid, int max_idx) {
  for (int idx = tid * 4; idx < TILE_SIZE; idx += BLOCK_SIZE * 4)
    if (idx < max_idx) {
      float4 tmp = reinterpret_cast<const float4 *>(input + idx)[0];
      val = f(val, f(f(tmp.x, tmp.y), f(tmp.z, tmp.w)));
    }
  return val;
}

template <float f(float, float)>
__device__ float block_reduce(float val, int BLOCK_SIZE, int tid, float *workspace) {
  workspace[tid] = val;
  for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      val = f(val, workspace[tid + stride]);
      workspace[tid] = val;
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

__device__ float block_broadcast(float val, int tid, float *workspace) {
  if (tid == 0)
    workspace[0] = val;
  __syncthreads();
  return workspace[0];
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

__device__ float4 sub(float4 a, float b) { return {a.x - b, a.y - b, a.z - b , a.w - b}; }
__device__ float4 mul(float4 a, float b) { return {a.x * b, a.y * b, a.z * b , a.w * b}; }
__device__ float4 exp(float4 a) { return {exp(a.x), exp(a.y), exp(a.z), exp(a.w)}; }
__device__ float max(float4 a) { return max(max(a.x, a.y), max(a.z, a.w)); }
__device__ float sum(float4 a) { return a.x + a.y + a.z + a.w; }

// calculate max per row
__global__ void softmax_v1_kernel_pass1(const float *input, float *output, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  input += row * N + tile_id * TILE_SIZE;
  output += row;
  extern __shared__ float workspace[];

  float max_val = -INFINITY;
  max_val = thread_reduce<max>(max_val, input, TILE_SIZE, BLOCK_SIZE, tid, N);
  max_val = block_reduce<max>(max_val, BLOCK_SIZE, tid, workspace);
  max_val = warp_reduce<max>(max_val, tid);
  if (tid == 0)
    atomicMax(output, max_val);
}

// calculate exp(x - max_row) and sum(exp(x - max_row)) per row
__global__ void softmax_v1_kernel_pass2(
  const float *input, float *output, float *max_per_row, float *denom, int M, int N, int TILE_SIZE
) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  input += row * N + tile_id * TILE_SIZE;
  output += row * N + tile_id * TILE_SIZE;
  max_per_row += row;
  denom += row;
  extern __shared__ float workspace[];

  float total = 0.0f;
  float subtractor = max_per_row[0];
  for (int idx = tid * 4; idx < TILE_SIZE; idx += BLOCK_SIZE * 4)
    if (idx < N) {
      float4 tmp = reinterpret_cast<const float4 *>(input + idx)[0];
      tmp = exp(sub(tmp, subtractor));
      reinterpret_cast<float4 *>(output + idx)[0] = tmp;
      total += sum(tmp);
    }
  total = block_reduce<add>(total, BLOCK_SIZE, tid, workspace);
  total = warp_reduce<add>(total, tid);
  if (tid == 0)
    atomicAdd(denom, total);
}

// normalize output
__global__ void softmax_v1_kernel_pass3(float *output, float *denom, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  output += row * N + tile_id * TILE_SIZE;
  denom += row;

  float scale = 1.0f / denom[0];
  for (int idx = tid * 4; idx < TILE_SIZE; idx += BLOCK_SIZE * 4)
    if (idx < N) {
      float4 tmp = reinterpret_cast<float4 *>(output + idx)[0];
      reinterpret_cast<float4 *>(output + idx)[0] = mul(tmp, scale);
    }
}

// naive softmax. workspace size = 2M
void softmax_v1(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int TILE_SIZE = BLOCK_SIZE * 4;
  const dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;

  float *max_per_row = workspace;
  float *denom = workspace + M;

  // pass 1: max per row
  // pass 2: exp(x - max) and sum
  // pass 3: normalize
  softmax_v1_kernel_pass1<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, max_per_row, M, N, TILE_SIZE);
  softmax_v1_kernel_pass2<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, max_per_row, denom, M, N, TILE_SIZE);
  softmax_v1_kernel_pass3<<<grid_size, BLOCK_SIZE>>>(output, denom, M, N, TILE_SIZE);
}

__global__ void softmax_v2_kernel_pass1(const float *input, float *workspace, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  input += row * N + tile_id * TILE_SIZE;
  float *max_global = workspace + row;
  float *normalizer_global = workspace + M + row;

  extern __shared__ float max_shared[];
  float *normalizer_shared = max_shared + BLOCK_SIZE;

  // algorithm 3 in https://arxiv.org/pdf/1805.02867
  // calculate max per row and normalizer (denominator) at the same time
  float max_val = -INFINITY;
  float normalizer = 0.0f;
  for (int idx = tid * 4; idx < TILE_SIZE; idx += BLOCK_SIZE * 4) {
    if (idx < N) {
      float4 tmp = reinterpret_cast<const float4 *>(input + idx)[0];
      float old_max_val = max_val;
      max_val = max(max_val, max(tmp));

      normalizer *= exp(old_max_val - max_val);  // correction
      normalizer += sum(exp(sub(tmp, max_val)));
    }
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
    float other_max = atomicMax(max_global, max_val);
    float new_max = max(max_val, other_max);

    // using atomicCAS is slow
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    float other_normalizer = normalizer_global[0];
    float assumed;
    do {
      assumed = other_normalizer;
      float new_normalizer = normalizer * exp(max_val - new_max) + other_normalizer * exp(other_max - new_max);
      other_normalizer = atomicCAS(reinterpret_cast<int *>(normalizer_global),
                                   __float_as_int(other_normalizer),
                                   __float_as_int(new_normalizer));
    } while (assumed != other_normalizer);
  }
}

__global__ void softmax_v2_kernel_pass2(const float *input, float *output, const float *workspace, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  input += row * N + tile_id * TILE_SIZE;
  output += row * N + tile_id * TILE_SIZE;
  float row_max  = workspace[row];
  float scale = 1.0f / workspace[M + row];

  for (int idx = tid * 4; idx < TILE_SIZE; idx += BLOCK_SIZE * 4)
    if (idx < N) {
      // recompute exp(x - row_max)
      float4 tmp = reinterpret_cast<const float4 *>(input + idx)[0];
      reinterpret_cast<float4 *>(output + idx)[0] = mul(exp(sub(tmp, row_max)), scale);
    }
}

// online softmax. workspace size = 2M.
void softmax_v2(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int TILE_SIZE = BLOCK_SIZE;
  const dim3 grid_size(cdiv(N, BLOCK_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE * 2;

  // pass 1: find max and normalizer at the same time
  // pass 2: calculate output
  softmax_v2_kernel_pass1<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, workspace, M, N, TILE_SIZE);
  softmax_v2_kernel_pass2<<<grid_size, BLOCK_SIZE>>>(input, output, workspace, M, N, TILE_SIZE);
}
