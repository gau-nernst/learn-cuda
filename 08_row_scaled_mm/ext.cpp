#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

template <typename input_type>
using ScaledMmFn = void(const input_type *A,
                        const input_type *B,
                        const float *scale_A,
                        const float *scale_B,
                        nv_bfloat16 *C,
                        int M, int N, int K);

using Int8ScaledMmFn = ScaledMmFn<int8_t>;
using Fp8ScaledMmFn = ScaledMmFn<__nv_fp8_e4m3>;

Int8ScaledMmFn row_scaled_mm_v1;
Fp8ScaledMmFn row_scaled_mm_v1;

template <Int8ScaledMmFn int8_mm_fn, Fp8ScaledMmFn fp8_mm_fn>
at::Tensor row_scaled_mm(const at::Tensor &A,
                         const at::Tensor &B,
                         const at::Tensor &scale_A,
                         const at::Tensor &scale_B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B.t());
  CHECK_INPUT(scale_A);
  CHECK_INPUT(scale_B);
  TORCH_CHECK(A.size(1) == B.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  at::Tensor C = at::empty({M, N}, A.options().dtype(at::kBFloat16));
  // at::Tensor C = at::zeros({M, N}, A.options().dtype(at::kBFloat16));  // for correctness check, use this

  auto scale_A_ptr = reinterpret_cast<const float *>(scale_A.data_ptr());
  auto scale_B_ptr = reinterpret_cast<const float *>(scale_B.data_ptr());
  auto C_ptr = reinterpret_cast<nv_bfloat16 *>(C.data_ptr());

  if (A.dtype() == at::kChar)
    int8_mm_fn(
      reinterpret_cast<const int8_t *>(A.data_ptr()),
      reinterpret_cast<const int8_t *>(B.data_ptr()),
      scale_A_ptr, scale_B_ptr, C_ptr, M, N, K);
  else if (A.dtype() == at::kFloat8_e4m3fn)
    fp8_mm_fn(
      reinterpret_cast<const __nv_fp8_e4m3 *>(A.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3 *>(B.data_ptr()),
      scale_A_ptr, scale_B_ptr, C_ptr, M, N, K);

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_scaled_mm_v1", &row_scaled_mm<row_scaled_mm_v1, row_scaled_mm_v1>);
}
