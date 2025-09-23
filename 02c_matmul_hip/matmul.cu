// this file can't be compiled as .cpp

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_bf16.h>

using MatmulFunc = void(const __hip_bfloat16 *, const __hip_bfloat16 *, __hip_bfloat16 *, int, int, int, hipStream_t);

MatmulFunc matmul_v1;

template <MatmulFunc matmul_raw>
at::Tensor matmul(const at::Tensor& A, const at::Tensor& B) {
  TORCH_CHECK(A.stride(1) == 1);
  TORCH_CHECK(B.stride(0) == 1);
  TORCH_CHECK(A.size(1) == B.size(0));

  const int M = A.size(0);
  const int N = B.size(1);
  const int K = A.size(1);

  at::Tensor C = at::empty({M, N}, A.options());

  auto A_gmem = reinterpret_cast<const __hip_bfloat16 *>(A.data_ptr());
  auto B_gmem = reinterpret_cast<const __hip_bfloat16 *>(B.data_ptr());
  auto C_gmem = reinterpret_cast<__hip_bfloat16 *>(C.data_ptr());
  hipStream_t stream = at::cuda::getCurrentCUDAStream();

  matmul_raw(A_gmem, B_gmem, C_gmem, M, N, K, stream);

  return C;
}

TORCH_LIBRARY(hip_matmul, m) {
  m.def("matmul_v1(Tensor A, Tensor B) -> Tensor");
  m.impl("matmul_v1", at::kCUDA, &matmul<matmul_v1>);
}
