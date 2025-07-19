#include <cuda.h>
#include <iostream>

void sum_v1(const float *input, float *output, int M, int N, int BLOCK_SIZE);
void sum_v2(const float *input, float *output, int M, int N, int BLOCK_SIZE);
void sum_v3(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v4a(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v4b(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v4c(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v5(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);
void sum_v6(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE);

int main() {
  // Size of the input data
  const int M = 64, N = 32000;

  // Allocate memory for input and output on host
  float *h_input = new float[M * N];
  float *h_output = new float[M];

  // Initialize input data on host
  for (int i = 0; i < M * N; i++)
    h_input[i] = 1.0f; // Example: Initialize all elements to 1

  for (int i = 0; i < M; i++)
    h_output[0] = 0.0f;

  // Allocate memory for input and output on device
  float *d_input;
  float *d_output;

  cudaMalloc(&d_input, sizeof(float) * M * N);
  cudaMalloc(&d_output, sizeof(float) * M);

  // Copy data from host to device
  cudaMemcpy(d_output, h_output, sizeof(float) * M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, h_input, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  // Launch the kernel
  const int TILE_SIZE = 8192;
  const int BLOCK_SIZE = 128;
  sum_v6(d_input, d_output, M, N, TILE_SIZE, BLOCK_SIZE);

  // Copy result back to host
  cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  // Print the result
  std::cout << "Sum is " << *h_output << std::endl;

  // Cleanup
  delete[] h_input;
  delete h_output;
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
