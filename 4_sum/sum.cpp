#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void sum_v1_launch(const float *input, float *output, int m, int n, int block_size);
void sum_v2_launch(const float *input, float *output, int m, int n, int block_size);
void sum_v3_launch(const float *input, float *output, int m, int n, int block_size, int coarse_factor);

torch::Tensor sum_v1(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);
  torch::Tensor output = torch::empty({m}, input.options());

  int block_size = 4;
  sum_v1_launch(input.data_ptr<float>(), output.data_ptr<float>(), m, n, block_size);
  return output;
}

torch::Tensor sum_v2(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);
  torch::Tensor output = torch::zeros({m}, input.options());

  int block_size = 512;
  sum_v2_launch(input.data_ptr<float>(), output.data_ptr<float>(), m, n, block_size);
  return output;
}

torch::Tensor sum_v3(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);
  torch::Tensor output = torch::zeros({m}, input.options());

  int block_size = 512;
  int coarse_factor = 4;
  sum_v3_launch(input.data_ptr<float>(), output.data_ptr<float>(), m, n, block_size, coarse_factor);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_v1", &sum_v1, "Sum v1");
  m.def("sum_v2", &sum_v2, "Sum v2");
  m.def("sum_v3", &sum_v3, "Sum v3");
}
