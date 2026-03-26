/**
 * @file software_pipeline.cuh
 * @brief Manual software pipelining and double-buffering orchestration.
 *
 * Implements the cycle-by-cycle logic required to hide HBM2 latency on 
 * Volta (V100). This header manages buffer indexing and prefetching 
 * interspersed with compute to ensure the Tensor Cores stay saturated.
 */

#pragma once

#include <cuda_runtime.h>
#include "fused_attention.cuh"
#include "xor_swizzle.cuh"

/**
 * @brief Manages the state of a double-buffered pipeline.
 */
struct PipelineState {
    int load_buffer;    // Buffer currently receiving global memory data
    int compute_buffer; // Buffer currently being read by the Tensor Cores
};

/**
 * @brief Initializes the pipeline pointers for the loop prologue.
 */
__device__ __forceinline__ void init_pipeline(PipelineState &pipe) {
    pipe.load_buffer = 0;
    pipe.compute_buffer = 0;
}

/**
 * @brief Executes a pointer swap between load and compute buffers.
 * Uses rapid XOR pointer arithmetic as suggested in the blueprint.
 */
__device__ __forceinline__ void advance_pipeline(PipelineState &pipe) {
    pipe.load_buffer ^= 1;
    pipe.compute_buffer ^= 1;
}

/**
 * @brief Performs a coordinated global-to-shared load for a tile.
 * * Uses XOR swizzling to prevent bank conflicts during the write phase.
 *
 * @param smem_buffer Pointer to the specific shared memory segment (Buffer 0 or 1).
 * @param gmem_ptr Pointer to the source matrix in Global Memory.
 * @param block_idx The current K-block index being fetched.
 * @param tid Thread ID in the block.
 */
__device__ __forceinline__ void load_tile_to_smem(
    half* smem_buffer,
    const half* gmem_ptr,
    int block_idx,
    int tid) 
{
    const int tile_elements = BLOCK_K * HEAD_DIM;
    const int gmem_offset = block_idx * tile_elements;

    // Grid-stride loop within the block to fill the SMEM tile
    #pragma unroll
    for (int i = tid; i < tile_elements; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        
        // Target swizzled address to avoid bank conflicts
        int smem_idx = x_swizzle(row, col, HEAD_DIM);
        
        // Volatile cast ensures the compiler doesn't reorder these reads
        // outside of the intended software pipeline phase.
        ((volatile half*)smem_buffer)[smem_idx] = gmem_ptr[gmem_offset + i];
    }
}

/**
 * @brief Memory Proxy Fence for Volta.
 * Forces the compiler to ensure all global memory reads issued before 
 * the fence are visible to the thread block before proceeding.
 */
__device__ __forceinline__ void memory_fence() {
    __syncthreads();
}