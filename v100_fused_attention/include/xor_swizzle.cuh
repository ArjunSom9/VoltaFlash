/**
 * @file xor_swizzle.cuh
 * @brief Implementation of the Mathematical XOR Swizzling Algorithm.
 *
 * This header provides the logic to categorically eliminate shared memory 
 * bank conflicts on the Volta architecture. By scrambling the physical 
 * mapping of two-dimensional matrix tiles into linear memory, we ensure 
 * that warp-wide requests for columns or sub-blocks are evenly distributed 
 * across the 32 physical memory banks.
 */

#pragma once

#include <cuda_runtime.h>

/**
 * @brief Computes a swizzled memory offset for shared memory access.
 * * Instead of a standard row-major indexing (row * stride + col), this 
 * function applies a bitwise XOR transformation to the physical bank index.
 * * Logic:
 * 1. Identify the logical bank based on the vector length (8 elements for fp16).
 * 2. XOR the row index with the logical bank index.
 * 3. Map the result to one of the 32 physical banks.
 * * @param row The logical row index in the tile.
 * @param col The logical column index in the tile.
 * @param stride The row stride (typically HEAD_DIM or block width).
 * @return The swizzled linear offset into the shared memory array.
 */
__device__ __forceinline__ int xor_swizzle(int row, int col, int stride) {
    // For 16-bit elements (half), a 16-byte bank line contains 8 elements.
    const int elements_per_bank_line = 8; 
    
    // Calculate the logical bank index (which "chunk" of 8 elements this column falls into)
    int logical_bank = col / elements_per_bank_line;
    
    // Apply the XOR transformation to derive the conflict-free physical bank.
    // This ensures that for any given column, different rows map to different physical banks.
    int physical_bank = (row ^ logical_bank) % 32;
    
    // Calculate the internal offset within the bank line (0-7)
    int offset_within_line = (col % elements_per_bank_line);
    
    // Final linear address calculation
    return (row * stride) + (physical_bank * elements_per_bank_line) + offset_within_line;
}

/**
 * @brief Specialized swizzle for 32-bit (float) accumulator writebacks if needed.
 * * While input matrices (Q, K, V) are fp16, intermediate scores or output
 * fragments may be fp32. This variant adjusts the vector length for 4-byte elements.
 */
__device__ __forceinline__ int xor_swizzle_f32(int row, int col, int stride) {
    // For 32-bit elements (float), a 16-byte bank line contains 4 elements.
    const int elements_per_bank_line = 4;
    int logical_bank = col / elements_per_bank_line;
    int physical_bank = (row ^ logical_bank) % 32;
    int offset_within_line = (col % elements_per_bank_line);
    
    return (row * stride) + (physical_bank * elements_per_bank_line) + offset_within_line;
}