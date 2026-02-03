#include <torch/extension.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

typedef void MatmulFn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K);

MatmulFn matmul_v1;
MatmulFn matmul_v2;
MatmulFn matmul_v3;
MatmulFn matmul_v4;
MatmulFn matmul_v5;
MatmulFn matmul_v6;
MatmulFn matmul_v7;
MatmulFn matmul_v8;

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
  m.def("matmul_v1", &matmul_pt<matmul_v1>, "Matrix multiplication v1");
  m.def("matmul_v2", &matmul_pt<matmul_v2>, "Matrix multiplication v2");
  m.def("matmul_v3", &matmul_pt<matmul_v3>, "Matrix multiplication v3");
  m.def("matmul_v4", &matmul_pt<matmul_v4>, "Matrix multiplication v4");
  m.def("matmul_v5", &matmul_pt<matmul_v5>, "Matrix multiplication v5");
  m.def("matmul_v6", &matmul_pt<matmul_v6>, "Matrix multiplication v6");
  m.def("matmul_v7", &matmul_pt<matmul_v7>, "Matrix multiplication v7");
  m.def("matmul_v8", &matmul_pt<matmul_v8>, "Matrix multiplication v8");
}
