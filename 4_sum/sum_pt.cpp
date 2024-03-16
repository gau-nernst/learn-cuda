#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

#define cdiv(a, b) ((a) + (b)-1) / (b)

void sum_v1_launch(const float *input, float *output, int m, int n);
void sum_v2_launch(const float *input, float *output, int m, int n, int tpb);

torch::Tensor sum_v1(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);
  torch::Tensor output = torch::empty({m}, input.options());

  sum_v1_launch(input.data_ptr<float>(), output.data_ptr<float>(), m, n);
  return output;
}

torch::Tensor sum_v2(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);
  torch::Tensor output = torch::zeros({m}, input.options());

  sum_v2_launch(input.data_ptr<float>(), output.data_ptr<float>(), m, n, 1024);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_v1", &sum_v1, "Sum v1");
  m.def("sum_v2", &sum_v2, "Sum v2");
}
