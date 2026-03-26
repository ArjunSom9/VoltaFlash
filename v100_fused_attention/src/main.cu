/**
 * @file main.cu
 * @brief Host entry point for initializing tensors, verifying the V100 hardware, 
 * and invoking the bare-metal fused attention kernel.
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fused_attention.cuh"
#include "host_utils.cuh"

int main() {
    // ------------------------------------------------------------------------
    // 1. Hardware Substrate Verification
    // ------------------------------------------------------------------------
    // Ensure the device matches the micro-architectural constraints of the blueprint.
    initialize_and_verify_device();

    // ------------------------------------------------------------------------
    // 2. Configuration & Memory Allocation
    // ------------------------------------------------------------------------
    const int seq_len = 256; // Must be a multiple of BLOCK_Q/BLOCK_K
    const size_t matrix_elements = seq_len * HEAD_DIM;
    const size_t matrix_bytes = matrix_elements * sizeof(half);

    std::cout << "[INFO] Sequence Length     : " << seq_len << std::endl;
    std::cout << "[INFO] Head Dimension      : " << HEAD_DIM << std::endl;
    std::cout << "[INFO] Memory Allocation   : " << (matrix_bytes * 4) / 1024 << " KB total" << std::endl;

    // Host matrices (Pinned memory can be used here for faster transfers if needed)
    std::vector<half> h_Q(matrix_elements);
    std::vector<half> h_K(matrix_elements);
    std::vector<half> h_V(matrix_elements);
    std::vector<half> h_O(matrix_elements);

    // Initialize with deterministic dummy data for initial validation
    for (int i = 0; i < matrix_elements; ++i) {
        h_Q[i] = __float2half(0.123f);
        h_K[i] = __float2half(0.456f);
        h_V[i] = __float2half(0.789f);
    }

    // Device allocations
    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, matrix_bytes));

    // ------------------------------------------------------------------------
    // 3. Data Transfer & Execution
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), matrix_bytes, cudaMemcpyHostToDevice));

    std::cout << "[INFO] Launching Fused Attention Kernel..." << std::endl;
    
    // Invoke the host launcher defined in src/fused_attention_kernel.cu
    // This function handles grid/block calculations and dynamic SMEM config.
    launch_fused_attention(d_Q, d_K, d_V, d_O, seq_len);

    // Synchronize and check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------------
    // 4. Verification & Cleanup
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, matrix_bytes, cudaMemcpyDeviceToHost));

    std::cout << "[SUCCESS] Kernel execution complete. Hardware limits respected." << std::endl;

    // Post-execution check for illegal memory accesses or hardware exceptions
    CUDA_CHECK(cudaGetLastError());

    // Cleanup resources
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return 0;
}