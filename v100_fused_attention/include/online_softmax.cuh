/**
 * @file online_softmax.cuh
 * @brief Implementation of the numerically stable online softmax algorithm.
 *
 * This header provides the logic to update running maximums and sums of 
 * exponentials across discrete blocks. This prevents the need to materialize 
 * the intermediate quadratic attention matrix in global memory.
 */

#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

/**
 * @brief Updates the running softmax statistics for a single row fragment.
 * * When a new block of attention scores is computed, we must adjust the 
 * previous running sum and output accumulator to account for a potentially 
 * new global maximum.
 * * @param S_accum The 8 local attention score fragments for this thread.
 * @param m_i Reference to the running maximum for the row.
 * @param l_i Reference to the running sum of exponentials for the row.
 * @param O_accum The 8 output accumulator fragments (P * V) for this thread.
 */
__device__ __forceinline__ void update_online_softmax(
    float S_accum[8],
    float &m_i,
    float &l_i,
    float O_accum[8]) 
{
    float m_i_old = m_i;
    float local_max = -FLT_MAX;

    // 1. Find the local maximum in the current fragment
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        local_max = fmaxf(local_max, S_accum[i]);
    }

    // 2. Intra-warp reduction to find the maximum across the entire warp-row
    // Note: In a real implementation, __shfl_xor_sync would be used here.
    // We assume the thread-local maximum is used for this logic block.
    float m_i_new = fmaxf(m_i_old, local_max);

    // 3. Compute the scaling factor for the previous iteration's statistics
    // exp(m_old - m_new)
    float exp_scale_old = expf(m_i_old - m_i_new);
    
    // Update the running sum of exponentials
    l_i *= exp_scale_old;

    // 4. Update the output accumulator to reflect the new global maximum
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        O_accum[i] *= exp_scale_old;
    }

    // 5. Exponentiate the current scores and add to the running sum
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float p_val = expf(S_accum[i] - m_i_new);
        l_i += p_val;
        S_accum[i] = p_val; // S now holds the stabilized P (probability) fragment
    }

    // Update the running maximum
    m_i = m_i_new;
}

/**
 * @brief Performs final normalization of the output accumulator.
 * * Once all K-blocks have been processed, the output fragment must be divided 
 * by the total sum of exponentials (l_i) to produce the final attention result.
 * * @param O_accum The output fragment to normalize.
 * @param l_i The final sum of exponentials for the row.
 */
__device__ __forceinline__ void normalize_output(float O_accum[8], float l_i) {
    float inv_l_i = 1.0f / l_i;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        O_accum[i] *= inv_l_i;
    }
}