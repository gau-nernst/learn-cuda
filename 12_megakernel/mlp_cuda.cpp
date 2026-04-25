#include <torch/library.h>
#include <ATen/ATen.h>

at::Tensor mlp_gemv_cuda_v1(const at::Tensor& x, const at::Tensor& norm, const at::Tensor& w13, const at::Tensor& w2);

TORCH_LIBRARY(my_mlp, m) {
  m.def("mlp_gemv_cuda_v1(Tensor x, Tensor norm, Tensor w13, Tensor w2) -> Tensor", &mlp_gemv_cuda_v1);
}
