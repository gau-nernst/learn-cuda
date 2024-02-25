#include <torch/extension.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define cdiv(a, b) ((a) + (b) - 1) / (b)

__global__ void matmul_kernel_v1(const float *input1, const float *input2, float *output, int m, int n, int k) {
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (col_idx >= k || row_idx >= m)
    return;

  float total = 0.0f;

  for (int i = 0; i < n; i++)
    total += input1[row_idx * n + i] * input2[i * k + col_idx];
  
  output[row_idx * k + col_idx] = total;
}

torch::Tensor matmul_v1(torch::Tensor input1, torch::Tensor input2) {
  CHECK_INPUT(input1);
  CHECK_INPUT(input2);
  int m = input1.size(0);
  int n = input1.size(1);
  TORCH_CHECK(n == input2.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int k = input2.size(1);
  torch::Tensor output = torch::empty({m, k}, input1.options());

  // NOTE: blockIdx.x is the fastest changing dimension. thus, we assign column index to it
  // intuitively, block dimensions will be PyTorch's dimensions in reverse.
  dim3 n_threads(16, 16);
  dim3 n_blocks(cdiv(k, 16), cdiv(m, 16));
  matmul_kernel_v1<<<n_blocks, n_threads>>>(input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(), m, n, k);

  return output;
}

constexpr int BLOCK_SIZE = 32;

__global__ void matmul_kernel_v2(const float *input1, const float *input2, float *output, int m, int n, int k) {
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

torch::Tensor matmul_v2(torch::Tensor input1, torch::Tensor input2) {
  CHECK_INPUT(input1);
  CHECK_INPUT(input2);
  int m = input1.size(0);
  int n = input1.size(1);;
  TORCH_CHECK(n == input2.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int k = input2.size(1);
  torch::Tensor output = torch::empty({m, k}, input1.options());

  dim3 n_threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 n_blocks(cdiv(k, BLOCK_SIZE), cdiv(m, BLOCK_SIZE));
  matmul_kernel_v2<<<n_blocks, n_threads>>>(input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(), m, n, k);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1", &matmul_v1, "Matrix multiplication v1");
  m.def("matmul_v2", &matmul_v2, "Matrix multiplication v2");
}
