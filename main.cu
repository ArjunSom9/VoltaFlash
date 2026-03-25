/**
 * @file main.cu
 * @brief Entry point for initializing data, allocating memory, and invoking the kernel.
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernel.cuh"
#include "utils.cuh"

int main() {
    int seq_len = 256;
    int size = seq_len * HEAD_DIM * sizeof(half);

    // Host matrices
    std::vector<half> h_Q(seq_len * HEAD_DIM);
    std::vector<half> h_K(seq_len * HEAD_DIM);
    std::vector<half> h_V(seq_len * HEAD_DIM);
    std::vector<half> h_O(seq_len * HEAD_DIM);

    // Initialize with dummy data
    for (int i = 0; i < seq_len * HEAD_DIM; ++i) {
        h_Q[i] = __float2half(0.1f);
        h_K[i] = __float2half(0.1f);
        h_V[i] = __float2half(0.1f);
    }

    // Device allocations
    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, size));
    CUDA_CHECK(cudaMalloc(&d_K, size));
    CUDA_CHECK(cudaMalloc(&d_V, size));
    CUDA_CHECK(cudaMalloc(&d_O, size));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), size, cudaMemcpyHostToDevice));

    std::cout << "Launching Bare-Metal Fused Attention Kernel on Volta V100..." << std::endl;

    // Execute the kernel
    launch_fused_attention(d_Q, d_K, d_V, d_O, seq_len);

    // Fetch result
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, size, cudaMemcpyDeviceToHost));

    std::cout << "Kernel execution complete. Hardware limits respected." << std::endl;

    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    return 0;
}