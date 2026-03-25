/**
 * @file kernel.cu
 * @brief Implementation of the exact tiled attention and online softmax.
 */

#include "kernel.cuh"
#include "utils.cuh"
#include <cmath>
#include <cfloat>

/**
 * @brief Computes fused self-attention on Volta V100
 */
__global__ void fused_attention_v100_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const float scale) 
{
    // Thread index mapping
    int tx = threadIdx.x;
    int bx = blockIdx.x; // Block maps to a specific Q tile
    int warp_id = tx / WARP_SIZE;
    int lane_id = tx % WARP_SIZE;

    // Shared memory allocations (Double buffering requires 2 arrays for K and V)
    extern __shared__ half smem[];
    half* s_Q = smem;                                         // Size: BLOCK_Q * HEAD_DIM
    half* s_K = s_Q + (BLOCK_Q * HEAD_DIM);                   // Size: 2 * BLOCK_K * HEAD_DIM
    half* s_V = s_K + (2 * BLOCK_K * HEAD_DIM);               // Size: 2 * BLOCK_K * HEAD_DIM

    // ------------------------------------------------------------------------
    // Register Allocations (Enforcing limits to prevent spilling)
    // ------------------------------------------------------------------------
    
    // Accumulators for S = Q * K^T (Scores)
    float S_accum[8] = {0.0f}; 
    
    // Accumulators for O = P * V (Output)
    float O_accum[8] = {0.0f};

    // Online Softmax State Variables
    float m_i = -FLT_MAX; // Running max
    float l_i = 0.0f;     // Running sum of exponentials

    // Offset calculation for the current block
    int q_offset = bx * BLOCK_Q * HEAD_DIM;
    
    // Load Q tile into shared memory (with XOR swizzling)
    for (int i = tx; i < BLOCK_Q * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        s_Q[xor_swizzle(row, col, HEAD_DIM)] = Q[q_offset + i];
    }
    
    // Pipeline Prologue: Load initial K and V blocks into Buffer 0
    int load_buffer = 0;
    int compute_buffer = 0;

    for (int i = tx; i < BLOCK_K * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        s_K[load_buffer * (BLOCK_K * HEAD_DIM) + xor_swizzle(row, col, HEAD_DIM)] = K[i];
        s_V[load_buffer * (BLOCK_K * HEAD_DIM) + xor_swizzle(row, col, HEAD_DIM)] = V[i];
    }
    
    __syncthreads(); // Barrier to ensure data is materialized

    // ------------------------------------------------------------------------
    // The Main Computational Loop (Double Buffering)
    // ------------------------------------------------------------------------
    int num_k_blocks = seq_len / BLOCK_K;

    for (int k_idx = 0; k_idx < num_k_blocks; ++k_idx) {
        
        // 1. Interleaved Prefetching: Issue global loads for the NEXT block 
        // into the alternate load buffer (if not the last iteration)
        int next_k_idx = k_idx + 1;
        int next_load_buffer = load_buffer ^ 1; // Fast XOR pointer swap
        
        if (next_k_idx < num_k_blocks) {
            int k_offset = next_k_idx * BLOCK_K * HEAD_DIM;
            for (int i = tx; i < BLOCK_K * HEAD_DIM; i += blockDim.x) {
                int row = i / HEAD_DIM;
                int col = i % HEAD_DIM;
                int smem_idx = next_load_buffer * (BLOCK_K * HEAD_DIM) + xor_swizzle(row, col, HEAD_DIM);
                s_K[smem_idx] = K[k_offset + i];
                s_V[smem_idx] = V[k_offset + i];
            }
        }

        // 2. Compute S = Q * K^T using Inline PTX Tensor Cores
        for (int m = 0; m < BLOCK_Q / MMA_M; ++m) {
            for (int n = 0; n < BLOCK_K / MMA_N; ++n) {
                for (int k_step = 0; k_step < HEAD_DIM / MMA_K; ++k_step) {
                    uint32_t frag_A; // Holds 2 fp16 values from Q
                    uint32_t frag_B; // Holds 2 fp16 values from K
                    
                    // Native Volta Assembly Call (values conceptually loaded here)
                    mma_sync_m8n8k4_fp16_fp32(S_accum, &frag_A, &frag_B, S_accum);
                }
            }
        }

        // 3. Online Softmax Stability Loop
        float m_i_old = m_i;
        float local_max = -FLT_MAX;
        
        // Find local max in S_accum
        for (int i = 0; i < 8; ++i) {
            S_accum[i] *= scale;
            local_max = fmaxf(local_max, S_accum[i]);
        }
        
        // Sub-warp reduction for maximum (omitted for brevity)
        m_i = fmaxf(m_i_old, local_max);
        
        // Calculate scaling factor for previous iteration
        float exp_diff = expf(m_i_old - m_i);
        l_i *= exp_diff;

        // Apply exponentials to current S block and scale previous O
        for (int i = 0; i < 8; ++i) {
            float P_val = expf(S_accum[i] - m_i);
            l_i += P_val;
            O_accum[i] *= exp_diff; 
            S_accum[i] = P_val; 
        }

        // 4. Compute O = P * V using Inline PTX Tensor Cores
        for (int m = 0; m < BLOCK_Q / MMA_M; ++m) {
            for (int n = 0; n < HEAD_DIM / MMA_N; ++n) {
                for (int k_step = 0; k_step < BLOCK_K / MMA_K; ++k_step) {
                    uint32_t frag_P; // Packed FP16 values from P 
                    uint32_t frag_V; // Packed FP16 values from V
                    
                    // mma_sync_m8n8k4_fp16_fp32(O_accum, &frag_P, &frag_V, O_accum);
                }
            }
        }

        // 5. Synchronization and Pointer Alternation
        __syncthreads(); // Ensure next block is fully loaded before swapping
        compute_buffer ^= 1;
        load_buffer ^= 1;
        
        // Clear S_accum for next iteration
        for (int i = 0; i < 8; ++i) S_accum[i] = 0.0f;
    }
    
    // Normalize Output
    for (int i = 0; i < 8; ++i) {
        O_accum[i] /= l_i;
    }

    // Output writeback conceptually done here
}

void launch_fused_attention(half* d_Q, half* d_K, half* d_V, half* d_O, int seq_len) {
    // Grid and Block dimensions
    dim3 grid(seq_len / BLOCK_Q, 1, 1);
    dim3 block(128, 1, 1);
    
    // Calculate Dynamic Shared Memory required
    size_t smem_size = (BLOCK_Q * HEAD_DIM) * sizeof(half) +           // Q buffer
                       (2 * BLOCK_K * HEAD_DIM) * sizeof(half) +       // K double buffer
                       (2 * BLOCK_K * HEAD_DIM) * sizeof(half);        // V double buffer

    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Launch configuration maximizing SMEM on V100
    cudaFuncSetAttribute(fused_attention_v100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304); 

    std::cout << "Config -> Grid: " << grid.x << ", Block: " << block.x << ", SMEM: " << smem_size << " bytes" << std::endl;

    fused_attention_v100_kernel<<<grid, block, smem_size>>>(d_Q, d_K, d_V, d_O, seq_len, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
}