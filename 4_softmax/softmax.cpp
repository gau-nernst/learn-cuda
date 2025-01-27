#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void softmax_v1(const float *input, float *output, float *workspace, int M, int N);
void softmax_v2(const float *input, float *output, float *workspace, int M, int N);

torch::Tensor softmax_v1_pt(torch::Tensor input) {
  CHECK_INPUT(input);
  int M = input.size(0);
  int N = input.size(1);
  torch::Tensor output = torch::empty_like(input);
  torch::Tensor workspace = torch::empty(M * 2, input.options());
  softmax_v1(input.data_ptr<float>(), output.data_ptr<float>(), workspace.data_ptr<float>(), M, N);
  return output;
}

torch::Tensor softmax_v2_pt(torch::Tensor input) {
  CHECK_INPUT(input);
  int M = input.size(0);
  int N = input.size(1);
  torch::Tensor output = torch::empty_like(input);
  torch::Tensor workspace = torch::empty(M * 2, input.options());
  softmax_v2(input.data_ptr<float>(), output.data_ptr<float>(), workspace.data_ptr<float>(), M, N);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_v1", &softmax_v1_pt, "Naive softmax");
  m.def("softmax_v2", &softmax_v2_pt, "Online softmax");
}
