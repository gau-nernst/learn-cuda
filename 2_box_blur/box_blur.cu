#include <torch/extension.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

__global__ void box_blur_kernel_v1(const float *input, float *output, int width, int height, int kernel_size) {
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;

  if (col_idx >= width || row_idx >= height)
    return;

  int radius = (kernel_size - 1) / 2;
  float total = 0.0f;

  for (int j = max(0, row_idx - radius); j < min(row_idx + radius + 1, height); j++)
    for (int i = max(0, col_idx - radius); i < min(col_idx + radius + 1, width); i++)
      total += input[batch_idx * width * height + j * width + i];

  output[batch_idx * width * height + row_idx * width + col_idx] = total / (kernel_size * kernel_size);
}

torch::Tensor box_blur_v1(torch::Tensor input, int kernel_size) {
  CHECK_INPUT(input);
  TORCH_CHECK(kernel_size > 0 && kernel_size % 2, "kernel_size must be positive and odd");
  TORCH_CHECK(input.dim() == 3, "Input must have exactly 3 dimensions");
  int bsize = input.size(0);
  int height = input.size(1);
  int width = input.size(2);
  torch::Tensor output = torch::empty_like(input);

  dim3 n_threads(16, 16, 1);
  dim3 n_blocks(cdiv(width, 16), cdiv(height, 16), bsize);
  box_blur_kernel_v1<<<n_blocks, n_threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), width, height, kernel_size);

  return output;
}

__global__ void box_blur_kernel_v2_row(const float *input, float *output, int width, int height, int kernel_size) {
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (row_idx >= height)
    return;

  int radius = (kernel_size - 1) / 2;
  float total = 0.0f;

  // initialize. assume kernel size is smaller than input
  for (int i = 0; i < radius; i++) {
    total += input[batch_idx * width * height + row_idx * width + i];
  }
  for (int i = 0; i < width; i++) {
    if (i + radius < width)
      total += input[batch_idx * width * height + row_idx * width + i + radius];
    output[batch_idx * width * height + row_idx * width + i] = total / kernel_size;
    if (i - radius >= 0)
      total -= input[batch_idx * width * height + row_idx * width + i - radius];
  }
}

__global__ void box_blur_kernel_v2_col(const float *input, float *output, int width, int height, int kernel_size) {
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (col_idx >= width)
    return;

  int radius = (kernel_size - 1) / 2;
  float total = 0.0f;

  // initialize. assume kernel size is smaller than input
  for (int i = 0; i < radius; i++) {
    total += input[batch_idx * width * height + i * width + col_idx];
  }
  for (int i = 0; i < height; i++) {
    if (i + radius < height)
      total += input[batch_idx * width * height + (i + radius) * width + col_idx];
    output[batch_idx * width * height + i * width + col_idx] = total / kernel_size;
    if (i - radius >= 0)
      total -= input[batch_idx * width * height + (i - radius) * width + col_idx];
  }
}

torch::Tensor box_blur_v2(torch::Tensor input, int kernel_size) {
  CHECK_INPUT(input);
  TORCH_CHECK(kernel_size > 0 && kernel_size % 2, "kernel_size must be positive and odd");
  TORCH_CHECK(input.dim() == 3, "Input must have exactly 3 dimensions");
  int bsize = input.size(0);
  int height = input.size(1);
  int width = input.size(2);
  torch::Tensor output1 = torch::empty_like(input);
  torch::Tensor output2 = torch::empty_like(input);

  int n_threads = 256;
  box_blur_kernel_v2_row<<<dim3(cdiv(height, n_threads), bsize), n_threads>>>(input.data_ptr<float>(), output1.data_ptr<float>(), width, height, kernel_size);
  box_blur_kernel_v2_col<<<dim3(cdiv(width, n_threads), bsize), n_threads>>>(output1.data_ptr<float>(), output2.data_ptr<float>(), width, height, kernel_size);

  return output2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("box_blur_v1", &box_blur_v1, "Box blur v1");
  m.def("box_blur_v2", &box_blur_v2, "Box blur v2");
}
