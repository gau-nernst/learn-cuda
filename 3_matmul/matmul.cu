#include <cmath>

__host__ __device__ int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

__global__ void matmul_v1_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || col >= N)
    return;

  float total = 0.0f;

  // broadcast read from A since each warp reads the same A value
  // coalesce read from B since each warp reads consecutive B values
  for (int k = 0; k < K; k++)
    total += A[row * K + k] * B[k * N + col];

  // coalesce write to C since each warp writes consecutive C values
  C[row * N + col] = total;
}

void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K) {
  // determine optimal block size at runtime
  int block_size_total;
  int min_grid_size; // we don't need this
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_total, matmul_v1_kernel, 0, 0);

  // NOTE: blockIdx.x is the fastest changing dimension. thus, we assign column index to it
  // intuitively, block dimensions will be PyTorch's dimensions in reverse.
  // NOTE: blockDim.x must be multiple of 32 (warpSize) to ensure coalesce memory access
  dim3 block_size(32, block_size_total / 32);
  dim3 grid_size(cdiv(N, 32), cdiv(M, block_size.y));
  matmul_v1_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

// we can't use a larger block size since we are limited by 1024 threads per block
constexpr int V2_BLOCK_SIZE = 32;

__global__ void matmul_v2_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int C_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int C_row = blockIdx.y * blockDim.y + threadIdx.y;

  // we cannot return early since all threads need to synchronize
  __shared__ float A_shmem[V2_BLOCK_SIZE][V2_BLOCK_SIZE];
  __shared__ float B_shmem[V2_BLOCK_SIZE][V2_BLOCK_SIZE];
  float total = 0.0f;

  // we move block by block along K dim 
  for (int k_start = 0; k_start < K; k_start += V2_BLOCK_SIZE) {
    // load data from global memory (DDR/HBM) to shared memory (SRAM)
    // notice now each thread only loads 2 x n_blocks elements
    // coalesced memory read for both A and B
    int A_col = k_start + threadIdx.x;
    int B_row = k_start + threadIdx.y;
    A_shmem[threadIdx.y][threadIdx.x] = C_row < M && A_col < K ? A[C_row * K + A_col] : 0.0f;
    B_shmem[threadIdx.y][threadIdx.x] = B_row < K && C_col < N ? B[B_row * N + C_col] : 0.0f;

    // wait for all threads in a block to load data
    __syncthreads();

    // compute from shared memory
    for (int tile_k = 0; tile_k < V2_BLOCK_SIZE; tile_k++)
      total += A_shmem[threadIdx.y][tile_k] * B_shmem[tile_k][threadIdx.x];

    // wait to finish before moving to the next tile
    __syncthreads();
  }

  if (C_row < M && C_col < N)
    C[C_row * N + C_col] = total;
}

void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 block_size(V2_BLOCK_SIZE, V2_BLOCK_SIZE);
  dim3 grid_size(cdiv(N, V2_BLOCK_SIZE), cdiv(M, V2_BLOCK_SIZE));
  matmul_v2_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

// we want to load a tile of size (height, width) from global to shared memory
// assume there are enough threads (each thread loads 1 element)
template <int height, int width>
__device__ void load_shared_memory(const float *in, int in_max_row, int in_max_col, int row_stride,
                                   float out[height][width], int tid) {
  const int col = tid % width;
  const int row = tid / width;
  out[row][col] = row < in_max_row && col < in_max_col ? in[row * row_stride + col] : 0.0f;
}

// 1D thread tiling
// we only use 1D threadIdx so that we can partition row/column differently for A and B tiles.
// NOTE: TILE_SIZE is a multiple of TILE_SIZE_K
template <int TILE_SIZE, int TILE_SIZE_K>
__global__ void matmul_v3_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int tid = threadIdx.x;
  const int row_start = blockIdx.y * TILE_SIZE;
  const int col_start = blockIdx.x * TILE_SIZE;
  A += row_start * K;
  B += col_start;
  C += row_start * N + col_start;

  __shared__ float A_tile[TILE_SIZE][TILE_SIZE_K]; // vertical rectangle
  __shared__ float B_tile[TILE_SIZE_K][TILE_SIZE]; // horizontal rectangle

  // each block calculates a (TILE_SIZE, TILE_SIZE) tile of C
  // each block has (TILE_SIZE x TILE_SIZE_K) threads -> each thread calculates multiple elements
  // each thread calculates consecutive elements in a column [tile_row_start, tile_row_start + n_elems_per_thread)
  const int n_elems_per_thread = TILE_SIZE / TILE_SIZE_K;
  const int tile_row_start = tid / TILE_SIZE * n_elems_per_thread;
  const int tile_col = tid % TILE_SIZE;
  float total[n_elems_per_thread] = {0.0};

  for (int k_start = 0; k_start < K; k_start += TILE_SIZE_K) {
    load_shared_memory<TILE_SIZE, TILE_SIZE_K>(A, M - row_start, K - k_start, K, A_tile, tid);
    load_shared_memory<TILE_SIZE_K, TILE_SIZE>(B, K - k_start, N - col_start, N, B_tile, tid);
    __syncthreads();

    A += TILE_SIZE_K;
    B += TILE_SIZE_K * N;

    for (int tile_k = 0; tile_k < TILE_SIZE_K; tile_k++) {
      const float B_val = B_tile[tile_k][tile_col]; // cache B_val in register

      for (int out_idx = 0; out_idx < n_elems_per_thread; out_idx++)
        total[out_idx] += A_tile[tile_row_start + out_idx][tile_k] * B_val;
    }
    __syncthreads();
  }

  for (int out_idx = 0; out_idx < n_elems_per_thread; out_idx++)
    if (row_start + tile_row_start + out_idx < M && col_start + tile_col < N)
      C[(tile_row_start + out_idx) * N + tile_col] = total[out_idx];
}

void matmul_v3(const float *A, const float *B, float *C, int M, int N, int K) {
  const int TILE_SIZE = 64;
  const int TILE_SIZE_K = 8;
  dim3 block_size(TILE_SIZE * TILE_SIZE_K);
  dim3 grid_size(cdiv(N, TILE_SIZE), cdiv(M, TILE_SIZE));
  matmul_v3_kernel<TILE_SIZE, TILE_SIZE_K><<<grid_size, block_size>>>(A, B, C, M, N, K);
}
