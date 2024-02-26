#include <torch/extension.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define cdiv(a, b) ((a) + (b) - 1) / (b)

// Kahan sum to reduce errors
__global__ void sum_kernel_v1(const float *input, float *output, int m, int n) {
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx >= m)
    return;

  float sum = 0.0f;
  float error = 0.0f;

  for (int i = 0; i < n; i++) {
    float item = input[row_idx * n + i] - error;
    float new_sum = sum + item;
    error = new_sum - sum - item;
    sum = new_sum;
  }
  
  output[row_idx] = sum;
}

torch::Tensor sum_v1(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);
  torch::Tensor output = torch::empty({m}, input.options());

  int n_threads = 256;
  int n_blocks = cdiv(m, n_threads);
  sum_kernel_v1<<<n_blocks, n_threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), m, n);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_v1", &sum_v1, "Sum v1");
}
