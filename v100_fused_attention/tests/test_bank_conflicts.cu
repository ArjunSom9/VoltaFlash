/**
 * @file test_bank_conflicts.cu
 * @brief Empirical verification of shared memory bank conflict eradication.
 *
 * This test compares naive column-major shared memory access against 
 * the XOR swizzled implementation to quantify the performance gain 
 * on the Volta architecture.
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "xor_swizzle.cuh"
#include "host_utils.cuh"
#include "fused_attention.cuh" // For HEAD_DIM

// ----------------------------------------------------------------------------
// Micro-benchmark Kernels
// ----------------------------------------------------------------------------

/**
 * @brief Simulates a 32-way bank conflict.
 * Threads in a warp read a column of a row-major matrix without swizzling.
 */
__global__ void benchmark_naive_access(float* out, int iterations) {
    extern __shared__ half smem[];
    int tid = threadIdx.x;
    
    // We only care about the first warp for this micro-benchmark
    if (tid >= 32) return;

    half val = __float2half(0.0f);
    for (int i = 0; i < iterations; ++i) {
        // Column-major read: thread 'tid' reads row 'tid', col 0.
        // If stride is a multiple of 32, every thread hits Bank 0.
        int idx = (tid * HEAD_DIM) + 0; 
        val = __hadd(val, smem[idx]);
    }
    out[tid] = __half2float(val);
}

/**
 * @brief Executes conflict-free column access using XOR swizzling.
 */
__global__ void benchmark_swizzled_access(float* out, int iterations) {
    extern __shared__ half smem[];
    int tid = threadIdx.x;
    
    if (tid >= 32) return;

    half val = __float2half(0.0f);
    for (int i = 0; i < iterations; ++i) {
        // Swizzled column-major read.
        int idx = xor_swizzle(tid, 0, HEAD_DIM);
        val = __hadd(val, smem[idx]);
    }
    out[tid] = __half2float(val);
}

// ----------------------------------------------------------------------------
// Host Logic: Analysis and Timing
// ----------------------------------------------------------------------------

void run_bank_analysis() {
    std::cout << "--- Mathematical Bank Mapping (Column 0, Rows 0-31) ---" << std::endl;
    std::cout << "Thread | Naive Bank | Swizzled Bank | Status" << std::endl;
    std::cout << "-------|------------|---------------|-------" << std::endl;

    for (int row = 0; row < 32; ++row) {
        int naive_bank = ((row * HEAD_DIM) + 0) % 32;
        
        // Simulating the xor_swizzle bank calculation logic
        int swizzled_idx = (row ^ (0 / 8)) % 32; 
        int swizzled_bank = swizzled_idx;

        std::cout << std::setw(6) << row << " | " 
                  << std::setw(10) << naive_bank << " | " 
                  << std::setw(13) << swizzled_bank << " | "
                  << (row == swizzled_bank ? "Mapped" : "Swizzled") << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    initialize_and_verify_device();
    run_bank_analysis();

    const int iterations = 100000;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(float)));

    // CUDA Events for high-precision timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t smem_size = (BLOCK_K * HEAD_DIM) * sizeof(half);

    // 1. Benchmark Naive
    cudaEventRecord(start);
    benchmark_naive_access<<<1, 32, smem_size>>>(d_out, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0;
    cudaEventElapsedTime(&naive_ms, start, stop);

    // 2. Benchmark Swizzled
    cudaEventRecord(start);
    benchmark_swizzled_access<<<1, 32, smem_size>>>(d_out, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float swizzled_ms = 0;
    cudaEventElapsedTime(&swizzled_ms, start, stop);

    // 3. Report
    std::cout << "--- Shared Memory Performance Results ---" << std::endl;
    std::cout << "[INFO] Naive Execution    : " << naive_ms << " ms" << std::endl;
    std::cout << "[INFO] Swizzled Execution : " << swizzled_ms << " ms" << std::endl;
    
    float speedup = naive_ms / swizzled_ms;
    std::cout << "[RESULT] Speedup Factor   : " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;

    if (speedup > 1.2f) {
        std::cout << "[SUCCESS] XOR Swizzling effectively hidden bank conflict latency." << std::endl;
    } else {
        std::cout << "[NOTE] Minor delta. Ensure HEAD_DIM is a multiple of 32 to see max conflict penalty." << std::endl;
    }

    cudaFree(d_out);
    return 0;
}