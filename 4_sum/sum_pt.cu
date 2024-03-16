#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define cdiv(a, b) ((a) + (b) - 1) / (b)

__global__ void sum_kernel_v1(const float *input, float *output, int m, int n);

template <int BLOCK_SIZE>
__global__ void sum_kernel_v2(const float *input, float *output, int m, int n);

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

torch::Tensor sum_v2(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);
  torch::Tensor output = torch::zeros({m}, input.options());

  int tpb = 1024;
  int n_blocks = cdiv(n, tpb);
  sum_kernel_v2<1024><<<dim3(n_blocks, m), dim3(tpb, 1)>>>(input.data_ptr<float>(), output.data_ptr<float>(), m, n);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_v1", &sum_v1, "Sum v1");
  m.def("sum_v2", &sum_v2, "Sum v2");
}
