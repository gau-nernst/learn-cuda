#include <torch/extension.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

__global__ void box_blur_kernel_v1(const float *input, int kernel_size, float *output, int width, int height) {
  const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_x >= width || idx_y >= height)
    return;

  int radius = (kernel_size - 1) / 2;
  float scale = 1.0f / (kernel_size * kernel_size);
  float total = 0.0f;

  for (int i = max(0, idx_x - radius); i < min(idx_x + radius + 1, width); i++)
    for (int j = max(0, idx_y - radius); j < min(idx_y + radius + 1, height); j++)
      total += input[j * width + i] * scale;

  output[idx_y * width + idx_x] = total;
}

torch::Tensor box_blur_v1(torch::Tensor input, int kernel_size) {
  CHECK_INPUT(input);
  TORCH_CHECK(kernel_size > 0 && kernel_size % 2, "kernel_size must be positive and odd");
  int height = input.size(1);
  int width = input.size(2);
  torch::Tensor output = torch::empty_like(input);

  dim3 n_threads(16, 16);
  dim3 n_blocks(cdiv(width, 16), cdiv(height, 16));
  box_blur_kernel_v1<<<n_blocks, n_threads>>>(input.data_ptr<float>(), kernel_size, output.data_ptr<float>(), width, height);

  return output;
}

__global__ void box_blur_kernel_v2_row(const float *input, int kernel_size, float *output, int width, int height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= height)
    return;

  int radius = (kernel_size - 1) / 2;
  float scale = 1.0f / kernel_size;
  float total = 0.0f;

  // initialize. assume kernel size is smaller than input
  for (int i = 0; i < radius; i++) {
    total += input[idx * width + i] * scale;
  }
  for (int i = 0; i < width; i++) {
    if (i + radius < width)
      total += input[idx * width + i + radius] * scale;
    output[idx * width + i] = total;
    if (i - radius >= 0)
      total -= input[idx * width + i - radius] * scale;
  }
}

__global__ void box_blur_kernel_v2_col(const float *input, int kernel_size, float *output, int width, int height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= width)
    return;

  int radius = (kernel_size - 1) / 2;
  float scale = 1.0f / kernel_size;
  float total = 0.0f;

  // initialize. assume kernel size is smaller than input
  for (int i = 0; i < radius; i++) {
    total += input[i * width + idx] * scale;
  }
  for (int i = 0; i < width; i++) {
    if (i + radius < height)
      total += input[(i + radius) * width + idx] * scale;
    output[i * width + idx] = total;
    if (i - radius >= 0)
      total -= input[(i - radius) * width + idx] * scale;
  }
}

torch::Tensor box_blur_v2(torch::Tensor input, int kernel_size) {
  CHECK_INPUT(input);
  TORCH_CHECK(kernel_size > 0 && kernel_size % 2, "kernel_size must be positive and odd");
  int height = input.size(1);
  int width = input.size(2);
  torch::Tensor output1 = torch::empty_like(input);
  torch::Tensor output2 = torch::empty_like(input);

  int n_threads = 256;
  box_blur_kernel_v2_row<<<cdiv(height, n_threads), n_threads>>>(input.data_ptr<float>(), kernel_size, output1.data_ptr<float>(), width, height);
  box_blur_kernel_v2_col<<<cdiv(width, n_threads), n_threads>>>(output1.data_ptr<float>(), kernel_size, output2.data_ptr<float>(), width, height);

  return output2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("box_blur_v1", &box_blur_v1, "Box blur v1");
  m.def("box_blur_v2", &box_blur_v2, "Box blur v2");
}
