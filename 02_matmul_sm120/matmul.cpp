#include <torch/library.h>
#include <ATen/ATen.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

typedef void MatmulBF16Fn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K);
typedef void MatmulINT8Fn(const int8_t *A, const int8_t *B, int *C, int M, int N, int K);

MatmulBF16Fn matmul_v0_bf16;
MatmulBF16Fn matmul_v1_bf16;
MatmulBF16Fn matmul_v2_bf16;
MatmulBF16Fn matmul_v3_bf16;

MatmulINT8Fn matmul_v0_int8;
MatmulINT8Fn matmul_v1_int8;
MatmulINT8Fn matmul_v2_int8;
MatmulINT8Fn matmul_v3_int8;

template <MatmulBF16Fn matmul_bf16_fn, MatmulINT8Fn matmul_int8_fn>
at::Tensor matmul_pt(const at::Tensor& A, const at::Tensor& B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B.t());
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  if (A.dtype() == at::kBFloat16) {
    auto options = A.options();
    at::Tensor C = at::empty({M, N}, options);
    // at::Tensor C = at::zeros({M, N}, options);  // for correctness check, use this
    matmul_bf16_fn(
      reinterpret_cast<nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
      reinterpret_cast<nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
      reinterpret_cast<nv_bfloat16 *>(C.data_ptr<at::BFloat16>()),
      M, N, K);
    return C;
  }
  else if (A.dtype() == at::kChar) {
    auto options = A.options().dtype(at::kInt);
    at::Tensor C = at::empty({M, N}, options);
    // at::Tensor C = at::zeros({M, N}, options);  // for correctness check, use this
    matmul_int8_fn(A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), C.data_ptr<int>(), M, N, K);
    return C;
  }
  else {
    TORCH_CHECK(false);
  }
}

TORCH_LIBRARY(my_module, m) {
  m.def("matmul_v0(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v0_bf16, matmul_v0_int8>);
  m.def("matmul_v1(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v1_bf16, matmul_v1_int8>);
  m.def("matmul_v2(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v2_bf16, matmul_v2_int8>);
  m.def("matmul_v3(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v3_bf16, matmul_v3_int8>);
}
