#include <iostream>
#include <cuda.h>

template <int BLOCK_SIZE>
__global__ void sum_kernel_v3(const float *input, float *output, int m, int n);

int main() {
    // Size of the input data
    const int size = 100000;
    const int bytes = size * sizeof(float);

    // Allocate memory for input and output on host
    float* h_input = new float[size];
    float* h_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f; // Example: Initialize all elements to 1
    }

    // Allocate memory for input and output on device
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    // Copy data from host to device
    float zero = 0.0f;
    cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    const int tpb = 512;
    // int numBlocks = (size + tpb - 1) / tpb;
    // sum_kernel_v2<tpb><<<numBlocks, tpb>>>(d_input, d_output, 1, size);

    int numBlocks = (size + tpb * 2 - 1) / (tpb * 2);
    sum_kernel_v3<tpb><<<numBlocks, tpb>>>(d_input, d_output, 1, size);

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
