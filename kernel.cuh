/**
 * @file kernel.cuh
 * @brief Declarations for the fused attention kernel and host launch functions.
 */

#pragma once

#include <cuda_fp16.h>
#include "config.h"

// Kernel declaration
__global__ void fused_attention_v100_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const float scale
);

// Host launcher declaration
void launch_fused_attention(
    half* d_Q, 
    half* d_K, 
    half* d_V, 
    half* d_O, 
    int seq_len
);