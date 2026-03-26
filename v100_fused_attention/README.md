# VoltaFlash: Bare-Metal Fused Attention Kernel

## Overview
VoltaFlash is a manual architecture and optimization of a fused, scaled dot-product attention kernel written exclusively in raw CUDA C++. It is meticulously hand-tuned for the NVIDIA Tesla V100 PCIe GPU. 

While modern attention implementations extract their performance from hardware features intrinsic to the Ampere and Hopper architectures, the Volta architecture lacks these modern conveniences. Consequently, achieving memory-efficient, IO-aware exact attention on a V100 mandates that the engineer manually emulate asynchronous data movement through intricate software pipelining, execute complex shared memory swizzling, and orchestrate explicit register-level thread mapping.

## Core Optimizations

* **Algorithmic Tiling & Online Softmax**: By retaining the intermediate attention scores and probability matrices entirely in fast on-chip Static Random-Access Memory and registers, the kernel reduces global memory reads and writes from a quadratic complexity to a linear complexity.
* **Inline PTX Tensor Cores**: To achieve maximum performance, the project bypasses the high-level WMMA Application Programming Interface entirely. It utilizes inline Parallel Thread Execution assembly instructions to grant granular, cycle-by-cycle control over register allocation and arithmetic execution.
* **XOR Shared Memory Swizzling**: To categorically eliminate bank conflicts, the project implements a sophisticated, mathematically rigorous data swizzling strategy. The implementation utilizes a bitwise Exclusive-OR transformation during the transitional phase from global memory to shared memory.
* **Software Pipelining & Double Buffering**: Because the Volta architecture entirely lacks native asynchronous memory capabilities, a manual software pipeline using double buffering is constructed to explicitly emulate this asynchronous behavior.

## Project Structure

* `Makefile`: Build configuration for compiling the multi-file CUDA project targeting the Volta Architecture (`sm_70`).
* `config.h`: Defines crucial tiling parameters (e.g., `HEAD_DIM 64`) and Volta `m8n8k4` architectural constants.
* `kernel.cu` / `kernel.cuh`: Contains the core `fused_attention_v100_kernel` implementation, managing shared memory allocations, the online softmax stability loop, and interleaved prefetching.
* `utils.cuh`: Houses device utility functions, including the `xor_swizzle` logic and the native Volta assembly call (`mma_sync_m8n8k4_fp16_fp32`).
* `main.cu`: The host entry point that initializes the query, key, and value matrices, allocates device memory, and launches the kernel.

## Build and Execution

**Prerequisites:** NVIDIA CUDA Toolkit and a Volta-architecture GPU (e.g., Tesla V100).

To compile the project, simply run:
```bash
make