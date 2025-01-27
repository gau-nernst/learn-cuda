#include <torch/extension.h>

using namespace pybind11::literals;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

void sum_v1(const float *input, float *output, int M, int N, int BLOCK_SIZE);
void sum_v2(const float *input, float *output, int M, int N, int BLOCK_SIZE);
void sum_v3(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v4a(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v4b(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v4c(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v5(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v6(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);

std::tuple<torch::Tensor, int, int> _setup(torch::Tensor input) {
  CHECK_INPUT(input);
  int M = input.size(0);
  int N = input.size(1);
  return std::tuple(torch::empty({M}, input.options()), M, N);
}

torch::Tensor sum_v1_pt(torch::Tensor input, int BLOCK_SIZE) {
  auto [output, M, N] = _setup(input);
  sum_v1(input.data_ptr<float>(), output.data_ptr<float>(), M, N, BLOCK_SIZE);
  return output;
}

torch::Tensor sum_v2_pt(torch::Tensor input, int BLOCK_SIZE) {
  auto [output, M, N] = _setup(input);
  sum_v2(input.data_ptr<float>(), output.data_ptr<float>(), M, N, BLOCK_SIZE);
  return output;
}

torch::Tensor sum_v3_pt(torch::Tensor input, int TILE_SIZE, int BLOCK_SIZE) {
  auto [output, M, N] = _setup(input);
  sum_v3(input.data_ptr<float>(), output.data_ptr<float>(), M, N, TILE_SIZE, BLOCK_SIZE);
  return output;
}

torch::Tensor sum_v4a_pt(torch::Tensor input, int TILE_SIZE, int BLOCK_SIZE) {
  auto [output, M, N] = _setup(input);
  sum_v4a(input.data_ptr<float>(), output.data_ptr<float>(), M, N, TILE_SIZE, BLOCK_SIZE);
  return output;
}

torch::Tensor sum_v4b_pt(torch::Tensor input, int TILE_SIZE, int BLOCK_SIZE) {
  auto [output, M, N] = _setup(input);
  sum_v4b(input.data_ptr<float>(), output.data_ptr<float>(), M, N, TILE_SIZE, BLOCK_SIZE);
  return output;
}

torch::Tensor sum_v4c_pt(torch::Tensor input, int TILE_SIZE, int BLOCK_SIZE) {
  auto [output, M, N] = _setup(input);
  sum_v4c(input.data_ptr<float>(), output.data_ptr<float>(), M, N, TILE_SIZE, BLOCK_SIZE);
  return output;
}

torch::Tensor sum_v5_pt(torch::Tensor input, int TILE_SIZE, int BLOCK_SIZE) {
  auto [output, M, N] = _setup(input);
  sum_v5(input.data_ptr<float>(), output.data_ptr<float>(), M, N, TILE_SIZE, BLOCK_SIZE);
  return output;
}

torch::Tensor sum_v6_pt(torch::Tensor input, int TILE_SIZE, int BLOCK_SIZE) {
  auto [output, M, N] = _setup(input);
  sum_v6(input.data_ptr<float>(), output.data_ptr<float>(), M, N, TILE_SIZE, BLOCK_SIZE);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_v1", &sum_v1_pt, "Sum v1", "input"_a, "BLOCK_SIZE"_a=1);
  m.def("sum_v2", &sum_v2_pt, "Sum v2", "input"_a, "BLOCK_SIZE"_a=128);
  m.def("sum_v3", &sum_v3_pt, "Sum v3", "input"_a, "TILE_SIZE"_a=2048, "BLOCK_SIZE"_a=256);
  m.def("sum_v4a", &sum_v4a_pt, "Sum v4a", "input"_a, "TILE_SIZE"_a=4096, "BLOCK_SIZE"_a=128);
  m.def("sum_v4b", &sum_v4b_pt, "Sum v4b", "input"_a, "TILE_SIZE"_a=4096, "BLOCK_SIZE"_a=128);
  m.def("sum_v4c", &sum_v4c_pt, "Sum v4c", "input"_a, "TILE_SIZE"_a=4096, "BLOCK_SIZE"_a=128);
  m.def("sum_v5", &sum_v5_pt, "Sum v5", "input"_a, "TILE_SIZE"_a=2048, "BLOCK_SIZE"_a=128);
  m.def("sum_v6", &sum_v6_pt, "Sum v6", "input"_a, "TILE_SIZE"_a=8192, "BLOCK_SIZE"_a=128);
}
