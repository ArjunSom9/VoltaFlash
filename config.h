/**
 * @file config.h
 * @brief Tiling and architectural constants for the Volta V100 GPU.
 */

#pragma once

// ----------------------------------------------------------------------------
// Tiling and Architectural Constants
// ----------------------------------------------------------------------------
#define HEAD_DIM 64
#define BLOCK_Q  64
#define BLOCK_K  64
#define WARP_SIZE 32

// Volta m8n8k4 specific parameters for Tensor Cores
#define MMA_M 8
#define MMA_N 8
#define MMA_K 4