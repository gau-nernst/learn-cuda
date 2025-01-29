#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void softmax_naive(const float *input, float *output, float *workspace, int M, int N);
void softmax_naive_split(const float *input, float *output, float *workspace, int M, int N);
void softmax_online(const float *input, float *output, float *workspace, int M, int N);
void softmax_online_split(const float *input, float *output, float *workspace, int M, int N);

template<
  void softmax(const float *input, float *output, float *workspace, int M, int N),
  bool use_workspace>
torch::Tensor softmax_pt(torch::Tensor input) {
  CHECK_INPUT(input);
  int M = input.size(0);
  int N = input.size(1);
  torch::Tensor output = torch::empty_like(input);
  float *workspace = use_workspace ? torch::empty(M * 2, input.options()).data_ptr<float>() : nullptr;
  softmax(input.data_ptr<float>(), output.data_ptr<float>(), workspace, M, N);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_naive", &softmax_pt<softmax_naive, false>, "Naive softmax");
  m.def("softmax_naive_split", &softmax_pt<softmax_naive_split, true>, "Naive softmax split");
  m.def("softmax_online", &softmax_pt<softmax_online, false>, "Online softmax");
  m.def("softmax_online_split", &softmax_pt<softmax_online_split, true>, "Online softmax split");
}
