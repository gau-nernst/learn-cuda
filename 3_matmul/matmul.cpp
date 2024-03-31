#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void matmul_v1_launch(const float *A, const float *B, float *C, int M, int N, int K);
void matmul_v2_launch(const float *A, const float *B, float *C, int M, int N, int K);

template <int version> torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, A.options());

  switch (version) {
  case 1:
    matmul_v1_launch(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  case 2:
    matmul_v2_launch(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  }
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1", &matmul<1>, "Matrix multiplication v1");
  m.def("matmul_v2", &matmul<2>, "Matrix multiplication v2");
}
