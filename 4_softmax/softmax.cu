#include <float.h>

__host__ __device__ int cdiv(int a, int b) { return (a + b - 1) / b; }
__device__ float add(float a, float b) { return a + b; }

constexpr int WARP_SIZE = 32;

template <float f(float, float)>
__device__ float thread_reduce(float val, const float *input, int N, int BLOCK_SIZE, int tid) {
  for (int idx = tid; idx < N; idx += BLOCK_SIZE)
    val = f(val, input[idx]);
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

__global__ void fill(float *data, int N, float val) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    data[idx] = val;
}

// calculate max per row
__global__ void softmax_v1_kernel_pass1(const float *input, float *max_per_row, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  const int col_offset = tile_id * TILE_SIZE;
  input += row * N + col_offset;
  max_per_row += row;
  extern __shared__ float workspace[];

  float max_val = -FLT_MAX;
  max_val = thread_reduce<max>(max_val, input, min(TILE_SIZE, N - col_offset), BLOCK_SIZE, tid);
  max_val = block_reduce<max>(max_val, BLOCK_SIZE, tid, workspace);
  max_val = warp_reduce<max>(max_val, tid);
  if (tid == 0)
    atomicMax(max_per_row, max_val);
}

// calculate exp(x - max_row) and sum(exp(x - max_row)) per row
__global__ void softmax_v1_kernel_pass2(
  const float *input, float *output, float *max_per_row, float *denom, int M, int N, int TILE_SIZE
) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  const int col_offset = tile_id * TILE_SIZE;
  input += row * N + col_offset;
  output += row * N + col_offset;
  max_per_row += row;
  denom += row;
  extern __shared__ float workspace[];

  float total = 0.0f;
  float subtractor = max_per_row[0];
  for (int idx = tid; idx < min(TILE_SIZE, N - col_offset); idx += BLOCK_SIZE) {
    float tmp = exp(input[idx] - subtractor);
    output[idx] = tmp;
    total += tmp;
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

  const int col_offset = tile_id * TILE_SIZE;
  output += row * N + col_offset;

  float scale = 1.0f / denom[row];
  for (int idx = tid; idx < min(TILE_SIZE, N - col_offset); idx += BLOCK_SIZE)
    output[idx] *= scale;
}

// naive softmax. workspace size = 2M
void softmax_v1(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int TILE_SIZE = BLOCK_SIZE * 2;
  const dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;

  float *max_per_row = workspace;
  float *denom = workspace + M;

  // setup workspace
  fill<<<cdiv(M, BLOCK_SIZE), BLOCK_SIZE>>>(max_per_row, M, -FLT_MAX);
  fill<<<cdiv(M, BLOCK_SIZE), BLOCK_SIZE>>>(denom, M, 0.0f);

  // pass 1: max per row
  // pass 2: exp(x - max) and sum
  // pass 3: normalize
  softmax_v1_kernel_pass1<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, max_per_row, M, N, TILE_SIZE);
  softmax_v1_kernel_pass2<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, max_per_row, denom, M, N, TILE_SIZE);
  softmax_v1_kernel_pass3<<<grid_size, BLOCK_SIZE>>>(output, denom, M, N, TILE_SIZE);
}

__global__ void softmax_v2_kernel_setup(float *workspace, int M) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M) {
    workspace[idx * 2] = -FLT_MAX;  // init for max
    workspace[idx * 2 + 1] = 1.0f;  // init for normalizer
  }
}

// unsafe
__device__ unsigned long long int float2_as_ull(float2 a) { return reinterpret_cast<unsigned long long int *>(&a)[0]; }
__device__ float2 ull_as_float2(unsigned long long int a) { return reinterpret_cast<float2 *>(&a)[0]; }

__global__ void softmax_v2_kernel_pass1(const float *input, float *workspace, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  const int col_offset = tile_id * TILE_SIZE;
  input += row * N + col_offset;
  workspace += row * 2;

  extern __shared__ float shmem[];
  float *max_shared = shmem;
  float *normalizer_shared = shmem + BLOCK_SIZE;

  // algorithm 3 in https://arxiv.org/pdf/1805.02867
  // calculate max per row and normalizer (denominator) at the same time
  float max_val = -FLT_MAX;
  float normalizer = 1.0f;
  for (int idx = tid; idx < min(TILE_SIZE, N - col_offset); idx += BLOCK_SIZE) {
    float tmp = input[idx];
    float old_max = max_val;
    max_val = max(max_val, tmp);
    normalizer = normalizer * exp(old_max - max_val) + exp(tmp - max_val);
  }

  max_shared[tid] = max_val;
  normalizer_shared[tid] = normalizer;
  for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      float other_max = max_shared[tid + stride];
      float other_normalizer = normalizer_shared[tid + stride];

      // equation 4, section 3.1 in https://arxiv.org/pdf/1805.02867
      float old_max = max_val;
      max_val = max(max_val, other_max);
      normalizer = normalizer * exp(old_max - max_val) + other_normalizer * exp(other_max - max_val);

      max_shared[tid] = max_val;
      normalizer_shared[tid] = normalizer;
    }
  }

  if (tid < WARP_SIZE)
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      float other_max = __shfl_down_sync(0xffffffff, max_val, stride);
      float other_normalizer = __shfl_down_sync(0xffffffff, normalizer, stride);

      float old_max = max_val;
      max_val = max(max_val, other_max);
      normalizer = normalizer * exp(old_max - max_val) + other_normalizer * exp(other_max - max_val);
    }

  if (tid == 0)
    if (gridDim.x == 1) {
      workspace[0] = max_val;
      workspace[1] = normalizer;
    } else {
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
      // using atomicCAS is slow
      float2 *addr = reinterpret_cast<float2 *>(workspace);
      float2 old = addr[0], assumed;
      do {
        assumed = old;
        float2 updated_val;
        updated_val.x = max(max_val, assumed.x);
        updated_val.y = normalizer * exp(max_val - updated_val.x) + assumed.y * exp(assumed.x - updated_val.x);
        old = ull_as_float2(atomicCAS(reinterpret_cast<unsigned long long int *>(workspace),
                                      float2_as_ull(assumed),
                                      float2_as_ull(updated_val)));
      } while (float2_as_ull(assumed) != float2_as_ull(old));
    }
}

__global__ void softmax_v2_kernel_pass2(const float *input, float *output, const float *workspace, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  const int col_offset = tile_id * TILE_SIZE;
  input += row * N + col_offset;
  output += row * N + col_offset;
  float row_max = workspace[row * 2];
  float scale = 1.0f / workspace[row * 2 + 1];

  for (int idx = tid; idx < min(TILE_SIZE, N - col_offset); idx += BLOCK_SIZE) {
    output[idx] = exp(input[idx] - row_max) * scale;  // recompute exp(x - row_max)
  }
}

// online softmax. workspace size = 2M.
void softmax_v2(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int TILE_SIZE = BLOCK_SIZE;
  // const int TILE_SIZE = N;
  const dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE * 2;

  if (grid_size.x > 1)
    softmax_v2_kernel_setup<<<cdiv(M, BLOCK_SIZE), BLOCK_SIZE>>>(workspace, M);

  // pass 1: find max and normalizer at the same time
  // pass 2: calculate output
  softmax_v2_kernel_pass1<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, workspace, M, N, TILE_SIZE);
  softmax_v2_kernel_pass2<<<grid_size, BLOCK_SIZE>>>(input, output, workspace, M, N, TILE_SIZE);
}
