#include <float.h>

__host__ __device__ int cdiv(int a, int b) { return (a + b - 1) / b; }
__device__ float add(float a, float b) { return a + b; }

template<typename B, typename A>
__device__ B bitcast(A a) {
  static_assert(sizeof(A) == sizeof(B));
  B b;
  memcpy(&b, &a, sizeof(A));
  return b;
}

constexpr int WARP_SIZE = 32;

template<typename F, typename V>
__device__ V thread_reduce(F f, V val, const float *input, int N, int BLOCK_SIZE, int tid) {
  for (int idx = tid; idx < N; idx += BLOCK_SIZE)
    val = f(val, input[idx]);
  return val;
}

template<typename F, typename V>
__device__ V block_reduce(F f, V val, int BLOCK_SIZE, int tid, V *workspace) {
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

template<typename T>
__device__ T shfl_down_sync(unsigned int mask, T var, int srcLane) {
  if constexpr (sizeof(T) == sizeof(int))
    return bitcast<T>(__shfl_down_sync(mask, bitcast<int>(var), srcLane));
  else if constexpr (sizeof(T) == sizeof(long long))
    return bitcast<T>(__shfl_down_sync(mask, bitcast<long long>(var), srcLane));
  else
    static_assert(!sizeof(T));
}

template<typename F, typename V>
__device__ V warp_reduce(F f, V val, int tid) {
  if (tid < WARP_SIZE)
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2)
      val = f(val, shfl_down_sync(0xffffffff, val, stride));
  return val;
}

template<typename T>
__device__ T block_broadcast(T val, int tid, T *workspace) {
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

__global__ void softmax_naive_kernel(const float *input, float *output, int M, int N) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int row = blockIdx.x;

  input += row * N;
  output += row * N;
  extern __shared__ float workspace[];

  // 1st pass
  float max_val = -FLT_MAX;
  max_val = thread_reduce(fmaxf, max_val, input, N, BLOCK_SIZE, tid);
  max_val = block_reduce(fmaxf, max_val, BLOCK_SIZE, tid, workspace);
  max_val = warp_reduce(fmaxf, max_val, tid);
  max_val = block_broadcast(max_val, tid, workspace);

  // 2nd pass
  float normalizer = 0.0f;
  auto f = [max_val](float a, float b) { return a + exp(b - max_val); };
  normalizer = thread_reduce(f, normalizer, input, N, BLOCK_SIZE, tid);
  normalizer = block_reduce(add, normalizer, BLOCK_SIZE, tid, workspace);
  normalizer = warp_reduce(add, normalizer, tid);
  normalizer = block_broadcast(normalizer, tid, workspace);

  // 3rd pass
  // NOTE: we can save exp(input[idx] - max_val) to output in 2nd pass
  // exchange 1 write for the computation
  float scale = 1.0f / normalizer;
  for (int idx = tid; idx < N; idx += BLOCK_SIZE)
    output[idx] = exp(input[idx] - max_val) * scale;
}

void softmax_naive(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int shmem_size = sizeof(float) * BLOCK_SIZE;
  softmax_naive_kernel<<<M, BLOCK_SIZE, shmem_size>>>(input, output, M, N);
}

__global__ void softmax_naive_split_setup_kernel(float *data, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] = -FLT_MAX;
    data[N + idx] = 0.0f;
  }
}

// calculate max per row
__global__ void softmax_naive_split_pass1_kernel(const float *input, float *max_per_row, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;

  const int col_offset = tile_id * TILE_SIZE;
  input += row * N + col_offset;
  max_per_row += row;
  extern __shared__ float workspace[];

  float max_val = -FLT_MAX;
  max_val = thread_reduce(fmaxf, max_val, input, min(TILE_SIZE, N - col_offset), BLOCK_SIZE, tid);
  max_val = block_reduce(fmaxf, max_val, BLOCK_SIZE, tid, workspace);
  max_val = warp_reduce(fmaxf, max_val, tid);
  if (tid == 0)
    if (gridDim.x == 1)
      max_per_row[0] = max_val;
    else
      atomicMax(max_per_row, max_val);
}

// calculate exp(x - max_row) and sum(exp(x - max_row)) per row
__global__ void softmax_naive_split_pass2_kernel(
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
  total = block_reduce(add, total, BLOCK_SIZE, tid, workspace);
  total = warp_reduce(add, total, tid);
  if (tid == 0)
    if (gridDim.x == 1)
      denom[0] = total;
    else
      atomicAdd(denom, total);
}

// normalize output
__global__ void softmax_naive_split_pass3_kernel(float *output, float *denom, int M, int N, int TILE_SIZE) {
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
void softmax_naive_split(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int TILE_SIZE = BLOCK_SIZE;
  const dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE;

  float *max_per_row = workspace;
  float *denom = workspace + M;

  // setup workspace
  if (grid_size.x > 1)
    softmax_naive_split_setup_kernel<<<cdiv(M, BLOCK_SIZE), BLOCK_SIZE>>>(workspace, M);

  // pass 1: max per row
  // pass 2: exp(x - max) and sum
  // pass 3: normalize
  softmax_naive_split_pass1_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, max_per_row, M, N, TILE_SIZE);
  softmax_naive_split_pass2_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, max_per_row, denom, M, N, TILE_SIZE);
  softmax_naive_split_pass3_kernel<<<grid_size, BLOCK_SIZE>>>(output, denom, M, N, TILE_SIZE);
}

__global__ void softmax_online_split_setup_kernel(float *workspace, int M) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M) {
    workspace[idx * 2] = -FLT_MAX;  // init for max
    workspace[idx * 2 + 1] = 1.0f;  // init for normalizer
  }
}

// equation 4, section 3.1 in https://arxiv.org/pdf/1805.02867
__device__ float2 online_normalizer(float2 a, float2 b) {
  float new_max = max(a.x, b.x);
  float normalizer = a.y * exp(a.x - new_max) + b.y * exp(b.x - new_max);
  return {new_max, normalizer};
}

__global__ void softmax_online_kernel(const float *input, float *output, int M, int N) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int row = blockIdx.x;
  extern __shared__ float2 shmem[];

  input += row * N;
  output += row * N;

  // pass 1
  // algorithm 3 in https://arxiv.org/pdf/1805.02867
  // calculate max per row and normalizer (denominator) at the same time
  float2 normalizer = {-FLT_MAX, 1.0f};
  auto f = [](float2 a, float b) {
    float new_max = fmaxf(a.x, b);
    float new_normalizer = a.y * exp(a.x - new_max) + exp(b - new_max);
    return float2{new_max, new_normalizer};
  };
  normalizer = thread_reduce(f, normalizer, input, N, BLOCK_SIZE, tid);
  normalizer = block_reduce(online_normalizer, normalizer, BLOCK_SIZE, tid, shmem);
  normalizer = warp_reduce(online_normalizer, normalizer, tid);
  normalizer = block_broadcast(normalizer, tid, shmem);

  // pass 2
  float subtractor = normalizer.x;
  float scale = 1.0f / normalizer.y;
  for (int idx = tid; idx < N; idx += BLOCK_SIZE)
    output[idx] = exp(input[idx] - subtractor) * scale;
}

void softmax_online(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int shmem_size = sizeof(float) * BLOCK_SIZE * 2;
  softmax_naive_kernel<<<M, BLOCK_SIZE, shmem_size>>>(input, output, M, N);
}

__global__ void softmax_online_split_pass1_kernel(const float *input, float *workspace, int M, int N, int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;
  const int row = blockIdx.y;
  extern __shared__ float2 shmem[];

  const int col_offset = tile_id * TILE_SIZE;
  input += row * N + col_offset;
  workspace += row * 2;

  float2 normalizer = {-FLT_MAX, 1.0f};
  auto f = [](float2 a, float b) {
    float new_max = fmaxf(a.x, b);
    float new_normalizer = a.y * exp(a.x - new_max) + exp(b - new_max);
    return float2{new_max, new_normalizer};
  };
  normalizer = thread_reduce(f, normalizer, input, min(TILE_SIZE, N - col_offset), BLOCK_SIZE, tid);
  normalizer = block_reduce(online_normalizer, normalizer, BLOCK_SIZE, tid, shmem);
  normalizer = warp_reduce(online_normalizer, normalizer, tid);

  if (tid == 0)
    if (gridDim.x == 1) {
      reinterpret_cast<float2 *>(workspace)[0] = normalizer;
    } else {
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
      // using atomicCAS is slow
      using ull = unsigned long long;
      ull *addr = reinterpret_cast<ull *>(workspace);
      ull old = addr[0], assumed;
      do {
        assumed = old;
        float2 new_normalizer = online_normalizer(normalizer, bitcast<float2>(assumed));
        old = atomicCAS(addr, assumed, bitcast<ull>(new_normalizer));
      } while (assumed != old);
    }
}

__global__ void softmax_online_split_pass2_kernel(const float *input, float *output, const float *workspace, int M, int N, int TILE_SIZE) {
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
void softmax_online_split(const float *input, float *output, float *workspace, int M, int N) {
  const int BLOCK_SIZE = 256;
  const int TILE_SIZE = BLOCK_SIZE;
  const dim3 grid_size(cdiv(N, TILE_SIZE), M);
  const int shmem_size = sizeof(float) * BLOCK_SIZE * 2;

  if (grid_size.x > 1)
    softmax_online_split_setup_kernel<<<cdiv(M, BLOCK_SIZE), BLOCK_SIZE>>>(workspace, M);

  // pass 1: find max and normalizer at the same time
  // pass 2: calculate output
  softmax_online_split_pass1_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, workspace, M, N, TILE_SIZE);
  softmax_online_split_pass2_kernel<<<grid_size, BLOCK_SIZE>>>(input, output, workspace, M, N, TILE_SIZE);
}
