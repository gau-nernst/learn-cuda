#include <torch/extension.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

typedef void MatmulFn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K);

MatmulFn matmul_v0;
MatmulFn matmul_v1;
MatmulFn matmul_v2;

template <MatmulFn matmul_fn> torch::Tensor matmul_pt(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B.t());
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, A.options());
  // torch::Tensor C = torch::zeros({M, N}, A.options());  // for correctness check, use this
  matmul_fn(
    reinterpret_cast<nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
    reinterpret_cast<nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
    reinterpret_cast<nv_bfloat16 *>(C.data_ptr<at::BFloat16>()),
    M, N, K);
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v0", &matmul_pt<matmul_v0>, "Matrix multiplication v0");
  m.def("matmul_v1", &matmul_pt<matmul_v1>, "Matrix multiplication v1");
  m.def("matmul_v2", &matmul_pt<matmul_v2>, "Matrix multiplication v2");
}
