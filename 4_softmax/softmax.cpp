#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void mini_softmax(const float *input, float *output, int M, int N, int BLOCK_SIZE);

torch::Tensor mini_softmax_pt(torch::Tensor input) {
  CHECK_INPUT(input);
  int M = input.size(0);
  int N = input.size(1);
  torch::Tensor output = torch::empty_like(input);

  int BLOCK_SIZE = 1024;
  mini_softmax(input.data_ptr<float>(), output.data_ptr<float>(), M, N, BLOCK_SIZE);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mini_softmax", &mini_softmax_pt, "Mini softmax"); }
