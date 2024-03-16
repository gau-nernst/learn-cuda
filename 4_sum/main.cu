#include <cuda.h>
#include <iostream>

void sum_v1_launch(const float *input, float *output, int m, int n);
void sum_v2_launch(const float *input, float *output, int m, int n, int tpb);

int main() {
  // Size of the input data
  const int size = 100000;
  const int bytes = size * sizeof(float);

  // Allocate memory for input and output on host
  float *h_input = new float[size];
  float *h_output = new float;

  // Initialize input data on host
  for (int i = 0; i < size; i++) {
    h_input[i] = 1.0f; // Example: Initialize all elements to 1
  }

  // Allocate memory for input and output on device
  float *d_input;
  float *d_output;

  cudaMalloc(&d_input, bytes);
  cudaMalloc(&d_output, sizeof(float));

  // Copy data from host to device
  float zero = 0.0f;
  cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

  // Launch the kernel
  int tpb = 1024;
  sum_v2_launch(d_input, d_output, 1, size, tpb);

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
