/**
 * @file fused_attention.cuh
 * @brief Core declarations and architectural constants for the bare-metal 
 * fused attention kernel targeting the NVIDIA Tesla V100 (Volta).
 *
 * This header serves as the foundational contract between the host launcher 
 * and the device execution, defining the strict tiling dimensions required 
 * to keep the working set entirely within the 128 KB SMEM/L1 cache limit.
 */

#pragma once

#include <cuda_fp16.h>

// ============================================================================
// Architectural and Tiling Constants (Formerly config.h)
// ============================================================================

/** * Dimension of each attention head. 
 * Fixed to 64 to perfectly align with 16-bit float memory transactions. 
 */
#define HEAD_DIM 64

/** * Query block size (Sequence length processed per block). 
 * Balanced against the 65,536 register-per-SM physical limit to prevent spills.
 */
#define BLOCK_Q  64

/** * Key/Value block size. 
 * Defines the inner loop tiling dimension.
 */
#define BLOCK_K  64

/** Number of threads in a Volta execution warp. */
#define WARP_SIZE 32

// ----------------------------------------------------------------------------
// Volta PTX m8n8k4 Tensor Core Sub-matrix Dimensions
// ----------------------------------------------------------------------------
#define MMA_M 8
#define MMA_N 8
#define MMA_K 4

// ============================================================================
// Kernel Declarations
// ============================================================================

/**
 * @brief Computes exact fused self-attention on Volta V100.
 * * @param Q Pointer to Query matrix in global memory (fp16).
 * @param K Pointer to Key matrix in global memory (fp16).
 * @param V Pointer to Value matrix in global memory (fp16).
 * @param O Pointer to Output matrix in global memory (fp16).
 * @param seq_len The sequence length of the attention operation.
 * @param scale The scaling factor applied to Q*K^T before softmax.
 */
__global__ void fused_attention_v100_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const float scale
);

// ============================================================================
// Host Launcher Declarations
// ============================================================================

/**
 * @brief Configures grid/block dimensions and dynamic shared memory, 
 * then launches the fused attention kernel.
 * * @param d_Q Device pointer to Query matrix.
 * @param d_K Device pointer to Key matrix.
 * @param d_V Device pointer to Value matrix.
 * @param d_O Device pointer to Output matrix.
 * @param seq_len Sequence length.
 */
void launch_fused_attention(
    half* d_Q, 
    half* d_K, 
    half* d_V, 
    half* d_O, 
    int seq_len
);