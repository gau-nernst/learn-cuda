#include <torch/library.h>
#include <ATen/ATen.h>
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

template <MatmulFn matmul_fn>
at::Tensor matmul_pt(const at::Tensor& A, const at::Tensor& B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B.t());
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  auto options = A.options();
  at::Tensor C = at::empty({M, N}, options);
  // at::Tensor C = at::zeros({M, N}, options);  // for correctness check, use this
  matmul_fn(
    reinterpret_cast<nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
    reinterpret_cast<nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
    reinterpret_cast<nv_bfloat16 *>(C.data_ptr<at::BFloat16>()),
    M, N, K);
  return C;
}

TORCH_LIBRARY(my_module, m) {
  m.def("matmul_v1(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v1>);
  m.def("matmul_v2(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v2>);
  m.def("matmul_v3(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v3>);
  m.def("matmul_v4(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v4>);
  m.def("matmul_v5(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v5>);
  m.def("matmul_v6(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v6>);
  m.def("matmul_v7(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v7>);
  m.def("matmul_v8(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v8>);
}
