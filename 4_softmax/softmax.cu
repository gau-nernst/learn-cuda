#include <torch/extension.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define cdiv(a, b) ((a) + (b) - 1) / (b)

__global__ void softmax_kernel_v1(const float *input, float *output, int m, int n) {
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float sum_exp;
  sum_exp = 0.0f;
  __syncthreads();

  float val = expf(input[row_idx * n + col_idx]);
  atomicAdd(&sum_exp, val);
  __syncthreads();

  output[row_idx * n + col_idx] = val / sum_exp;
}

torch::Tensor softmax_v1(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);;
  torch::Tensor output = torch::empty_like(input);

  int n_threads = n;
  int n_blocks = cdiv(n, n_threads);
  softmax_kernel_v1<<<dim3(n_blocks, m), dim3(n_threads, 1)>>>(input.data_ptr<float>(), output.data_ptr<float>(), m, n);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_v1", &softmax_v1, "Softmax v1");
}
