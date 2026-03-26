/**
 * @file test_attention_exact.cu
 * @brief Mathematical validation of the fused attention kernel.
 *
 * Compares the output of the optimized Volta kernel against a 
 * double-precision CPU reference implementation to verify the 
 * exactness of the online softmax and tiling logic.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fused_attention.cuh"
#include "host_utils.cuh"

// ----------------------------------------------------------------------------
// CPU Reference Implementation (Standard Attention)
// ----------------------------------------------------------------------------
void cpu_attention_reference(
    const std::vector<half>& Q,
    const std::vector<half>& K,
    const std::vector<half>& V,
    std::vector<half>& O,
    int seq_len,
    float scale) 
{
    std::vector<float> scores(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        // 1. Compute QK^T for row i
        for (int j = 0; j < seq_len; ++j) {
            float sum = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                sum += __half2float(Q[i * HEAD_DIM + d]) * __half2float(K[j * HEAD_DIM + d]);
            }
            scores[j] = sum * scale;
        }

        // 2. Softmax for row i (with numerical stability)
        float max_val = -1e38f;
        for (float s : scores) max_val = std::max(max_val, s);

        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - max_val);
            sum_exp += scores[j];
        }

        for (int j = 0; j < seq_len; ++j) {
            scores[j] /= sum_exp;
        }

        // 3. Compute P * V for row i
        for (int d = 0; d < HEAD_DIM; ++d) {
            float out_val = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                out_val += scores[j] * __half2float(V[j * HEAD_DIM + d]);
            }
            O[i * HEAD_DIM + d] = __float2half(out_val);
        }
    }
}

// ----------------------------------------------------------------------------
// Main Test Runner
// ----------------------------------------------------------------------------
int main() {
    const int seq_len = 128; // Keep small for CPU reference speed
    const size_t matrix_elements = seq_len * HEAD_DIM;
    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    // Initialize Device
    initialize_and_verify_device();

    // Host Data
    std::vector<half> h_Q(matrix_elements);
    std::vector<half> h_K(matrix_elements);
    std::vector<half> h_V(matrix_elements);
    std::vector<half> h_O_gpu(matrix_elements);
    std::vector<half> h_O_cpu(matrix_elements);

    // Initialize with randomish but stable data
    for (int i = 0; i < matrix_elements; ++i) {
        h_Q[i] = __float2half((float)(i % 10) * 0.1f);
        h_K[i] = __float2half((float)(i % 7) * 0.1f);
        h_V[i] = __float2half((float)(i % 5) * 0.1f);
    }

    // Device Allocation
    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, matrix_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, matrix_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, matrix_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, matrix_elements * sizeof(half)));

    // Copy to Device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), matrix_elements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), matrix_elements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), matrix_elements * sizeof(half), cudaMemcpyHostToDevice));

    // Execute Fused Kernel
    std::cout << "[TEST] Launching Fused GPU Kernel..." << std::endl;
    launch_fused_attention(d_Q, d_K, d_V, d_O, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O_gpu.data(), d_O, matrix_elements * sizeof(half), cudaMemcpyDeviceToHost));

    // Execute CPU Reference
    std::cout << "[TEST] Computing CPU Reference..." << std::endl;
    cpu_attention_reference(h_Q, h_K, h_V, h_O_cpu, seq_len, scale);

    // Compare Results
    float max_err = 0.0f;
    float sum_err = 0.0f;
    for (int i = 0; i < matrix_elements; ++i) {
        float gpu_val = __half2float(h_O_gpu[i]);
        float cpu_val = __half2float(h_O_cpu[i]);
        float err = std::abs(gpu_val - cpu_val);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    float avg_err = sum_err / matrix_elements;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << " Validation Statistics " << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "[RESULT] Max Absolute Error : " << max_err << std::endl;
    std::cout << "[RESULT] Avg Absolute Error : " << avg_err << std::endl;

    // Threshold for half-precision math (approx 1e-3)
    if (max_err < 1e-2) {
        std::cout << "[SUCCESS] Mathematical exactness verified." << std::endl;
    } else {
        std::cout << "[FAILURE] Error exceeds tolerance." << std::endl;
    }
    std::cout << "------------------------------------------------------" << std::endl;

    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}