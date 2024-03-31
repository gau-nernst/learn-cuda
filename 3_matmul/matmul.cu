#include <cmath>

#define cdiv(a, b) ((a) + (b)-1) / (b)

__global__ void matmul_v1_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || col >= N)
    return;

  float total = 0.0f;

  for (int i = 0; i < K; i++)
    total += A[row * K + i] * B[i * N + col];

  C[row * N + col] = total;
}

void matmul_v1_launch(const float *A, const float *B, float *C, int M, int N, int K) {
  // NOTE: blockIdx.x is the fastest changing dimension. thus, we assign column index to it
  // intuitively, block dimensions will be PyTorch's dimensions in reverse.
  dim3 block_size(16, 16);
  dim3 grid_size(cdiv(N, 16), cdiv(M, 16));
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

  for (int block_k = 0; block_k < cdiv(K, BLOCK_SIZE); block_k++) {
    // load data from global memory (DDR/HBM) to shared memory (SRAM)
    // notice now each thread only loads 2 x n_blocks elements
    int A_col = block_k * BLOCK_SIZE + threadIdx.x;
    int B_row = block_k * BLOCK_SIZE + threadIdx.y;
    A_block[threadIdx.y][threadIdx.x] = row < M && A_col < K ? A[row * K + A_col] : 0.0f;
    B_block[threadIdx.y][threadIdx.x] = B_row < K && col < N ? B[B_row * N + col] : 0.0f;

    // wait for all threads in a block to load data
    __syncthreads();

    // compute from shared memory
    for (int sub_i = 0; sub_i < BLOCK_SIZE; sub_i++)
      total += A_block[threadIdx.y][sub_i] * B_block[sub_i][threadIdx.x];

    // wait to finish before moving to the next block
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = total;
}

void matmul_v2_launch(const float *A, const float *B, float *C, int M, int N, int K) {
  int BLOCK_SIZE = 32;
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size(cdiv(N, BLOCK_SIZE), cdiv(M, BLOCK_SIZE));

  switch (BLOCK_SIZE) {
  case 32:
    matmul_v2_kernel<32><<<grid_size, block_size>>>(A, B, C, M, N, K);
    break;
  }
}
