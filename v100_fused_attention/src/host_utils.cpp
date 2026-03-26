/**
 * @file host_utils.cpp
 * @brief Implementation of host-side utilities for CUDA error management 
 * and hardware substrate verification.
 */

#include "host_utils.cuh"
#include <iostream>
#include <cstdlib>

void check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "\n======================================================\n";
        std::cerr << "[FATAL] CUDA Runtime Error Detonated!\n";
        std::cerr << "======================================================\n";
        std::cerr << "Error Code  : " << err << "\n";
        std::cerr << "Description : " << cudaGetErrorString(err) << "\n";
        std::cerr << "Location    : " << file << ":" << line << "\n";
        std::cerr << "======================================================\n";
        
        // Terminate execution immediately to prevent undefined device states
        exit(EXIT_FAILURE);
    }
}

void initialize_and_verify_device() {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

    std::cout << "======================================================" << std::endl;
    std::cout << " Hardware Substrate Verification " << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "[INFO] Target GPU          : " << props.name << std::endl;
    std::cout << "[INFO] Compute Capability  : sm_" << props.major << props.minor << std::endl;
    std::cout << "[INFO] SM Count            : " << props.multiProcessorCount << std::endl;
    std::cout << "[INFO] Max SMEM / Block    : " << (props.sharedMemPerBlock / 1024) << " KB" << std::endl;
    std::cout << "[INFO] Max SMEM / SM       : " << (props.sharedMemPerMultiprocessor / 1024) << " KB" << std::endl;
    std::cout << "[INFO] Register File / SM  : " << (props.regsPerMultiprocessor / 1024) << " K Registers" << std::endl;

    // Verify Volta Architecture constraint (sm_70)
    if (props.major != 7 || props.minor != 0) {
        std::cerr << "\n[WARNING] Architecture Mismatch!" << std::endl;
        std::cerr << "This bare-metal kernel is heavily micro-optimized for the NVIDIA Tesla V100." << std::endl;
        std::cerr << "Register allocation, XOR swizzling, and PTX m8n8k4 instructions expect sm_70." << std::endl;
        std::cerr << "Executing on sm_" << props.major << props.minor << " may result in register spilling or bank conflicts.\n" << std::endl;
    } else {
        std::cout << "[INFO] Architecture Match  : Validated (NVIDIA Volta)" << std::endl;
    }
    
    // Verify Shared Memory Capacity
    if (props.sharedMemPerMultiprocessor < 98304) { // 96 KB
        std::cerr << "\n[ERROR] Insufficient Shared Memory!" << std::endl;
        std::cerr << "The fused attention kernel requires at least 96KB of configurable SMEM per SM." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "======================================================\n" << std::endl;
}