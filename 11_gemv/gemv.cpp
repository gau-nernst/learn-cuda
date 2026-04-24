#include <torch/library.h>
#include <ATen/ATen.h>
#include <cuda_bf16.h>

typedef void GemvFn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int N, int K);

GemvFn cuda_v1;
GemvFn cuda_persistent_v1;

template <GemvFn gemv_fn>
at::Tensor gemv(const at::Tensor& A, const at::Tensor& B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  auto C = at::empty({M, N}, A.options());
  //auto C = at::zeros({M, N}, A.options());  // for correctness check, use this
  gemv_fn(
    reinterpret_cast<const nv_bfloat16 *>(A.data_ptr()),
    reinterpret_cast<const nv_bfloat16 *>(B.data_ptr()),
    reinterpret_cast<      nv_bfloat16 *>(C.data_ptr()),
    N, K
  );
  return C;
}

TORCH_LIBRARY(my_gemv, m) {
  m.def("cuda_v1(Tensor A, Tensor B) -> Tensor", &gemv<cuda_v1>);
  m.def("cuda_persistent_v1(Tensor A, Tensor B) -> Tensor", &gemv<cuda_persistent_v1>);
}
