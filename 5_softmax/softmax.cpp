#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void softmax_v1_launch(const float *input, float *output, int m, int n, int block_size);

torch::Tensor softmax_v1(torch::Tensor input) {
  CHECK_INPUT(input);
  int m = input.size(0);
  int n = input.size(1);
  torch::Tensor output = torch::empty_like(input);

  int block_size = 1024;
  softmax_v1_launch(input.data_ptr<float>(), output.data_ptr<float>(), m, n, block_size);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("softmax_v1", &softmax_v1, "Softmax v1"); }
