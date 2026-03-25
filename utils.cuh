/**
 * @file utils.cuh
 * @brief Inline PTX assembly and device utility functions.
 */

#pragma once

#include <cuda_fp16.h>
#include <iostream>

// ----------------------------------------------------------------------------
// Utility: Error Checking Macro
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ----------------------------------------------------------------------------
// Device Functions: XOR Swizzling and PTX
// ----------------------------------------------------------------------------

/**
 * Section 3.3: Implementing the Mathematical XOR Swizzling Algorithm
 * Eliminates shared memory bank conflicts by scrambling the data placement
 * using bitwise XOR operations on the row and column indices.
 */
__device__ __forceinline__ int xor_swizzle(int row, int col, int stride) {
    // 16-bit elements -> 8 elements per 16-byte bank line
    const int vec_len = 8; 
    int logical_bank = col / vec_len;
    // XOR row with logical bank, modulo 32 physical banks
    int physical_bank = (row ^ logical_bank) % 32;
    int offset = (col % vec_len);
    return (row * stride) + (physical_bank * vec_len) + offset;
}

/**
 * Section 3.1 & 3.2: Transcending the CUDA WMMA API / Inline PTX
 * Wraps the Volta-specific m8n8k4 matrix-multiply-and-accumulate operation.
 * Inputs: A (fp16), B (fp16) packed into 32-bit registers.
 * Accumulators: C and D (fp32) stored in 8 discrete 32-bit registers.
 */
__device__ __forceinline__ void mma_sync_m8n8k4_fp16_fp32(
    float D[8], const uint32_t A[1], const uint32_t B[1], const float C[8]) 
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8}, {%9}, "
        "{%10, %11, %12, %13, %14, %15, %16, %17};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), 
          "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(B[0]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), 
          "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7])
    );
}