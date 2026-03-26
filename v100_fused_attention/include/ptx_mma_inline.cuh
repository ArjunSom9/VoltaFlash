/**
 * @file ptx_mma_inline.cuh
 * @brief Inline PTX assembly wrappers for Volta-specific Tensor Core instructions.
 *
 * This file implements the foundational m8n8k4 matrix-multiply-and-accumulate 
 * (MMA) operation. By bypassing the high-level WMMA API, we achieve granular 
 * control over register-level data mapping, which is critical for the 
 * mathematical stability of the online softmax algorithm.
 */

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

/**
 * @brief Executes a warp-synchronous m8n8k4 MMA operation.
 * * Target Architecture: sm_70 (NVIDIA Volta / Tesla V100)
 * * Precision: 
 * - Inputs (A, B): 16-bit Floating Point (FP16)
 * - Accumulators (C, D): 32-bit Floating Point (FP32)
 * * Fragment Topology:
 * - Matrix A (8x4): Each warp-lane (thread) holds one 32-bit register containing 
 * two packed __half elements.
 * - Matrix B (4x8): Each warp-lane holds one 32-bit register containing 
 * two packed __half elements.
 * - Matrix C/D (8x8): Each warp-lane holds eight 32-bit float registers.
 *
 * @param D Output accumulator fragment (8x8 result).
 * @param A Input fragment from matrix A (8x4).
 * @param B Input fragment from matrix B (4x8).
 * @param C Input accumulator fragment (initial values).
 */
__device__ __forceinline__ void mma_sync_m8n8k4_fp16_fp32(
    float D[8], 
    const uint32_t A[1], 
    const uint32_t B[1], 
    const float C[8]) 
{
    /**
     * PTX Instruction: mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
     * * The '.sync' qualifier ensures warp-wide synchronization.
     * The '.aligned' qualifier indicates that all threads in the warp 
     * execute this instruction with the same arguments.
     */
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

/**
 * @brief Utility for packing two 16-bit floats into a single 32-bit register.
 * Used to satisfy the operand requirements of the mma.sync instruction.
 */
__device__ __forceinline__ uint32_t pack_half2(half a, half b) {
    uint32_t res;
    asm("mov.b32 %0, {%1, %2};" : "=r"(res) : "h"(__half_as_short(a)), "h"(__half_as_short(b)));
    return res;
}