#include <cmath>

#define cdiv(a, b) ((a) + (b)-1) / (b)

__global__ void matmul_v1_kernel(const float *input1, const float *input2, float *output, int m, int n, int k) {
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (col_idx >= k || row_idx >= m)
    return;

  float total = 0.0f;

  for (int i = 0; i < n; i++)
    total += input1[row_idx * n + i] * input2[i * k + col_idx];

  output[row_idx * k + col_idx] = total;
}

void matmul_v1_launch(const float *input1, const float *input2, float *output, int m, int n, int k) {
  // NOTE: blockIdx.x is the fastest changing dimension. thus, we assign column index to it
  // intuitively, block dimensions will be PyTorch's dimensions in reverse.
  dim3 block_size(16, 16);
  dim3 grid_size(cdiv(k, 16), cdiv(m, 16));
  matmul_v1_kernel<<<grid_size, block_size>>>(input1, input2, output, m, n, k);
}

template <int BLOCK_SIZE>
__global__ void matmul_v2_kernel(const float *input1, const float *input2, float *output, int m, int n, int k) {
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  // we cannot return early since all threads need to synchronize

  __shared__ float input1_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float input2_block[BLOCK_SIZE][BLOCK_SIZE];
  float total = 0.0f;

  for (int block_n = 0; block_n < cdiv(n, BLOCK_SIZE); block_n++) {
    // load data from global memory (DDR/HBM) to shared memory (SRAM)
    // notice now each thread only loads 2 x n_blocks elements
    int input1_col = block_n * BLOCK_SIZE + threadIdx.x;
    int input2_row = block_n * BLOCK_SIZE + threadIdx.y;
    input1_block[threadIdx.y][threadIdx.x] = row_idx < m && input1_col < n ? input1[row_idx * n + input1_col] : 0.0f;
    input2_block[threadIdx.y][threadIdx.x] = input2_row < n && col_idx < k ? input2[input2_row * k + col_idx] : 0.0f;

    // wait for all threads in a block to load data
    __syncthreads();

    // compute from shared memory
    for (int sub_i = 0; sub_i < BLOCK_SIZE; sub_i++)
      total += input1_block[threadIdx.y][sub_i] * input2_block[sub_i][threadIdx.x];

    // wait to finish before moving to the next block
    __syncthreads();
  }

  if (row_idx < m && col_idx < k)
    output[row_idx * k + col_idx] = total;
}

void matmul_v2_launch(const float *input1, const float *input2, float *output, int m, int n, int k) {
  int BLOCK_SIZE = 32;
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size(cdiv(k, BLOCK_SIZE), cdiv(m, BLOCK_SIZE));

  switch (BLOCK_SIZE) {
  case 32:
    matmul_v2_kernel<32><<<grid_size, block_size>>>(input1, input2, output, m, n, k);
    break;
  }
}
