#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K);
void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K);
void matmul_v3(const float *A, const float *B, float *C, int M, int N, int K);

torch::Tensor matmul_v1_pt(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, A.options());
  matmul_v1(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  return C;
}

torch::Tensor matmul_v2_pt(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, A.options());
  matmul_v2(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  return C;
}

torch::Tensor matmul_v3_pt(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, A.options());
  matmul_v3(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1", &matmul_v1_pt, "Matrix multiplication v1");
  m.def("matmul_v2", &matmul_v2_pt, "Matrix multiplication v2");
  m.def("matmul_v3", &matmul_v3_pt, "Matrix multiplication v2");
}
