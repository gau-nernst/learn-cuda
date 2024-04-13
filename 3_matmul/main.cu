#include <cuda.h>
#include <iostream>

void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K);
void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K);

int main() {
  // Size of the input data
  const int N = 4096;

  // Allocate memory for input and output on host
  float *A = new float[N * N];
  float *B = new float[N * N];
  float *C = new float[N * N];

  // Initialize input data on host
  for (int i = 0; i < N * N; i++) {
    A[i] = 1.0f; // Example: Initialize all elements to 1
    B[i] = 1.0f; // Example: Initialize all elements to 1
  }

  // Allocate memory for input and output on device
  float *d_A;
  float *d_B;
  float *d_C;

  cudaMalloc(&d_A, N * N * sizeof(float));
  cudaMalloc(&d_B, N * N * sizeof(float));
  cudaMalloc(&d_C, N * N * sizeof(float));

  // Copy data from host to devic
  cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel
  // matmul_v1(d_A, d_B, d_C, N, N, N);
  matmul_v2(d_A, d_B, d_C, N, N, N);

  // Copy result back to host
  cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Check results
  for (int col = 0; col < N; col++)
    for (int row = 0; row < N; row++)
      if (C[row * N + col] != N)
        std::cout << "Wrong result at (" << row << ", " << col << ")" << std::endl;

  // Cleanup
  delete[] A;
  delete[] B;
  delete[] C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
