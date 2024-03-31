#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void matmul_v1_launch(const float *input1, const float *input2, float *output, int m, int n, int k);
void matmul_v2_launch(const float *input1, const float *input2, float *output, int m, int n, int k);

template <int version> torch::Tensor matmul(torch::Tensor input1, torch::Tensor input2) {
  CHECK_INPUT(input1);
  CHECK_INPUT(input2);
  int m = input1.size(0);
  int n = input1.size(1);
  TORCH_CHECK(n == input2.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int k = input2.size(1);
  torch::Tensor output = torch::empty({m, k}, input1.options());

  switch (version) {
  case 1:
    matmul_v1_launch(input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(), m, n, k);
  case 2:
    matmul_v2_launch(input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(), m, n, k);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1", &matmul<1>, "Matrix multiplication v1");
  m.def("matmul_v2", &matmul<2>, "Matrix multiplication v2");
}
