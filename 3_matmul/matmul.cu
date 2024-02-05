#include <torch/extension.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

__global__ void matmul_kernel_v1(const float *input1, const float *input2, float *output, int m, int n, int k) {
  const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_x >= k || idx_y >= m)
    return;

  float total = 0.0f;
  for (int i = 0; i < n; i++)
    total += input1[idx_y * n + i] * input2[i * k + idx_x];
  output[idx_y * k + idx_x] = total;
}

torch::Tensor matmul_v1(torch::Tensor input1, torch::Tensor input2) {
  CHECK_INPUT(input1);
  CHECK_INPUT(input2);
  int m = input1.size(0);
  int n = input1.size(1);;
  TORCH_CHECK(n == input2.size(0), "dim1 of input2 should be equal to dim2 of input1");
  int k = input2.size(1);
  torch::Tensor output = torch::empty({m, k}, input1.options());

  dim3 n_threads(16, 16);
  dim3 n_blocks(cdiv(m, 16), cdiv(k, 16));
  matmul_kernel_v1<<<n_blocks, n_threads>>>(input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(), m, n, k);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1", &matmul_v1, "Matrix multiplication v1");
}
