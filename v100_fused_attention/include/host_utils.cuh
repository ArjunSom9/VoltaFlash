/**
 * @file host_utils.cuh
 * @brief Host-side utilities for device initialization, architecture verification, 
 * and robust error handling.
 */

#pragma once

#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Utility: Error Checking Macro
// ----------------------------------------------------------------------------
/**
 * @brief Wraps CUDA API calls to intercept and decode runtime errors.
 * Routes the error code, file name, and line number to the implementation.
 */
#define CUDA_CHECK(call) check_cuda_error((call), __FILE__, __LINE__)

/**
 * @brief Out-of-line implementation for CUDA error checking.
 * * @param err The CUDA error code returned by the API call.
 * @param file The source file where the macro was invoked.
 * @param line The line number where the macro was invoked.
 */
void check_cuda_error(cudaError_t err, const char* file, int line);

/**
 * @brief Queries the active CUDA device, prints its micro-architectural 
 * properties, and verifies it meets the stringent requirements (sm_70) 
 * for the bare-metal fused attention kernel.
 */
void initialize_and_verify_device();