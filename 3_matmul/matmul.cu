#include <cmath>

#define cdiv(a, b) ((a) + (b)-1) / (b)

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

template <int BLOCK_SIZE>
__global__ void matmul_v2_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  // we cannot return early since all threads need to synchronize
  __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];
  float total = 0.0f;

  for (int tile = 0; tile < cdiv(K, BLOCK_SIZE); tile++) {
    // load data from global memory (DDR/HBM) to shared memory (SRAM)
    // notice now each thread only loads 2 x n_blocks elements
    int A_col = tile * BLOCK_SIZE + threadIdx.x;
    int B_row = tile * BLOCK_SIZE + threadIdx.y;
    A_block[threadIdx.y][threadIdx.x] = row < M && A_col < K ? A[row * K + A_col] : 0.0f;
    B_block[threadIdx.y][threadIdx.x] = B_row < K && col < N ? B[B_row * N + col] : 0.0f;

    // wait for all threads in a block to load data
    __syncthreads();

    // compute from shared memory
    // there is memory bank conflict for B_block
    for (int sub_i = 0; sub_i < BLOCK_SIZE; sub_i++)
      total += A_block[threadIdx.y][sub_i] * B_block[sub_i][threadIdx.x];

    // wait to finish before moving to the next tile
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = total;
}

void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 block_size(16, 16);
  dim3 grid_size(cdiv(N, 16), cdiv(M, 16));
  matmul_v2_kernel<16><<<grid_size, block_size>>>(A, B, C, M, N, K);
}
