#include <cuda.h>
#include <iostream>

typedef void MatmulFn(const float *A, const float *B, float *C, int M, int N, int K);

MatmulFn matmul_v1;
MatmulFn matmul_v2;
MatmulFn matmul_v3;
MatmulFn matmul_v4;
MatmulFn matmul_v5;
MatmulFn matmul_v6a;
MatmulFn matmul_v6b;

MatmulFn *matmul_table[] = {
  0,
  matmul_v1,
  matmul_v2,
  matmul_v3,
  matmul_v4,
  matmul_v5,
  matmul_v6a,
  matmul_v6b,
};

int main(int argc, char *argv[]) {
  const int N_MATMUL = sizeof(matmul_table) / sizeof(matmul_table[0]) - 1;
  int choice = N_MATMUL;
  if (argc >= 2) {
    choice = argv[1][0] - '0';
    if (choice > N_MATMUL)
      choice = N_MATMUL;
  }

  // Size of the input data
  const int N = 4096;

  // Allocate memory for input and output on host
  float *A = new float[N * N];
  float *B = new float[N * N];
  float *C = new float[N * N];

  // Initialize input data on host
  for (int i = 0; i < N * N; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
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
  matmul_table[choice](d_A, d_B, d_C, N, N, N);

  // Copy result back to host
  cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Check results
  for (int row = 0; row < N; row++)
    for (int col = 0; col < N; col++) {
      float val = C[row * N + col];
      if (val != N * 2)
        std::cout << "Wrong result " << val << " at (" << row << ", " << col << ")" << std::endl;
  }

  // Cleanup
  delete[] A;
  delete[] B;
  delete[] C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
