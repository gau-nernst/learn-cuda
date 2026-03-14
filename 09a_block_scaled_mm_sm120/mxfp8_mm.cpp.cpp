#include <cuda_bf16.h>
#include <cstdint>

#include <torch/library.h>
#include <ATen/ATen.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

typedef void MxFp8MmFn(const char *A, const char *B, const char *scale_A, const char *scale_B, nv_bfloat16 *C, int M, int N, int K);

MxFp8MmFn mxfp8_mm_v1;
MxFp8MmFn mxfp8_mm_v2;
MxFp8MmFn mxfp8_mm_v2b;
MxFp8MmFn mxfp8_mm_v3;

template <MxFp8MmFn mxfp8_mm_fn>
at::Tensor mxfp8_mm(
  const at::Tensor &A,
  const at::Tensor &B,
  const at::Tensor &SFA,
  const at::Tensor &SFB
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B.t());
  CHECK_INPUT(SFA);
  CHECK_INPUT(SFB);
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  at::Tensor C = at::empty({M, N}, A.options().dtype(at::kBFloat16));
  // at::Tensor C = at::zeros({M, N}, A.options().dtype(at::kBFloat16));  // for correctness check, use this
  mxfp8_mm_fn(
    reinterpret_cast<const char *>(A.data_ptr()),
    reinterpret_cast<const char *>(B.data_ptr()),
    reinterpret_cast<const char *>(SFA.data_ptr()),
    reinterpret_cast<const char *>(SFB.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
    M, N, K);
  return C;
}

TORCH_LIBRARY(my_module, m) {
  m.def("mxfp8_mm_v1(Tensor A, Tensor B, Tensor SFA, Tensor SFB) -> Tensor", &mxfp8_mm<mxfp8_mm_v1>);
  m.def("mxfp8_mm_v2(Tensor A, Tensor B, Tensor SFA, Tensor SFB) -> Tensor", &mxfp8_mm<mxfp8_mm_v2>);
  m.def("mxfp8_mm_v2b(Tensor A, Tensor B, Tensor SFA, Tensor SFB) -> Tensor", &mxfp8_mm<mxfp8_mm_v2b>);
  m.def("mxfp8_mm_v3(Tensor A, Tensor B, Tensor SFA, Tensor SFB) -> Tensor", &mxfp8_mm<mxfp8_mm_v3>);
}
