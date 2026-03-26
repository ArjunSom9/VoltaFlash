# VoltaFlash: Bare-Metal Fused Attention for NVIDIA Tesla V100

## 1. Executive Overview
**VoltaFlash** is a high-performance, manual architecture and optimization of a fused, scaled dot-product attention kernel written exclusively in raw CUDA C++. While contemporary machine learning relies heavily on domain-specific compilers like Triton, MLIR, and OpenXLA, achieving absolute peak hardware utilization—the theoretical roofline—still necessitates profound expertise in bare-metal GPU engineering.

The selection of the **NVIDIA Tesla V100 (Volta architecture, sm_70)** represents a strategic constraint. Unlike modern Ampere or Hopper architectures, Volta lacks native asynchronous memory copies and Tensor Memory Accelerators. This implementation demonstrates mastery of micro-architectural constraints by manually emulating hardware asynchrony through software pipelining, XOR swizzling, and inline PTX assembly.

---

## 2. The Hardware Substrate: NVIDIA Tesla V100 (GV100)
The software design is inextricably linked to the underlying silicon to eliminate execution bottlenecks.

### 2.1. Microarchitecture Specifications
* **Streaming Multiprocessors (SM)**: 80 (PCIe variant).
* **Processing Cores per SM**: 64 FP32, 64 INT32, 32 FP64, and 8 Tensor Cores.
* **Independent Thread Scheduling**: Volta maintains per-thread scheduling resources (program counter and call stack), permitting sub-warp divergence and reconvergence. This requires strict memory consistency models and explicit synchronization directives to avoid race conditions.

### 2.2. Memory Hierarchy and Ridge Point
| Memory Tier | Capacity Profile | Bandwidth / Latency |
| :--- | :--- | :--- |
| **Global Memory (HBM2)** | 16 GB or 32 GB | 900 GB/s / ~Hundreds of clock cycles |
| **L2 Cache** | 6 MB Shared | Accessible across all SMs |
| **Unified L1 / Shared Memory** | 128 KB per SM | 20-30 clock cycles (96 KB configurable SMEM) |
| **Register File** | 256 KB per SM | ~1 clock cycle (65,536 32-bit registers) |

The **Mathematical Ridge Point** for the V100 is calculated at **~138.8 FLOPs/byte**. Any kernel with an arithmetic intensity below this is memory-bound; our goal is to exceed this point to become compute-bound.

---

## 3. Core Technical Pillars

### 3.1. Algorithmic Tiling & Online Softmax
Standard attention mechanisms have quadratic memory complexity ($O(N^2)$) relative to sequence length. VoltaFlash partitions input matrices into blocks sized to fit within the 128 KB unified L1/Shared Memory. 
* **Online Softmax**: By maintaining running statistical variables (running maximum and sum of exponentials) in dedicated thread registers, the kernel reduces global memory transactions from quadratic to linear complexity ($O(N)$).
* **Register Management**: The kernel meticulously balances register pressure against warp occupancy to prevent "register spilling" into slow HBM memory.

### 3.2. Inline PTX & m8n8k4 Tensor Core Mechanics
To bypass the opaque fragment layouts of the high-level CUDA WMMA API, this project utilizes **Inline Parallel Thread Execution (PTX)** assembly.
* **Instruction**: Native `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`.
* **Thread Mapping**: Each physical thread is strictly assigned two 32-bit elements to hold in its localized registers.
* **Granular Control**: Bypassing APIs allows auxiliary operations, such as the exponential scaling required by online softmax, to be executed directly on physical registers before memory store-back.

### 3.3. XOR Shared Memory Swizzling
Accessing row-major matrices in a column-wise fashion to feed Tensor Cores causes catastrophic 32-way bank conflicts.
* **The Algorithm**: VoltaFlash applies a bitwise XOR transformation: `physical_bank = (row ^ (column / vector_length)) % 32`.
* **Impact**: This nonlinear mapping ensures that requested memory addresses are perfectly distributed across all 32 physical banks, guaranteeing conflict-free, peak-bandwidth transactions.

### 3.4. Software Pipelining (Double Buffering)
Since Volta lacks native asynchronous memory hardware, we manually construct a software pipeline:
1.  **Prologue**: Global memory reads load the initial block into the first shared memory buffer.
2.  **Computational Loop**: The warp computes using the current buffer while concurrently prefetching the next block into a secondary buffer.
3.  **Latency Hiding**: Global memory fetch latency (~hundreds of cycles) is hidden behind the massive clock cycle requirements of Tensor Core matrix multiplications.

---

## 4. Project Structure
The repository is organized to isolate micro-architectural concerns from host-side orchestration:

```text
v100_fused_attention/
├── src/
│   ├── main.cu                # Host entry, memory allocation, and hardware check.
│   ├── fused_attention_kernel.cu # Core implementation of the tiled pipeline.
│   └── host_utils.cpp         # Error handling and substrate verification.
├── include/
│   ├── fused_attention.cuh    # Kernel declarations and tiling constants.
│   ├── online_softmax.cuh     # Running max/sum stability logic.
│   ├── ptx_mma_inline.cuh     # Raw m8n8k4 assembly wrappers.
│   ├── xor_swizzle.cuh        # Conflict-free memory indexing.
│   └── software_pipeline.cuh  # Prefetching and buffer management.
├── tests/
│   ├── test_attention_exact.cu # Validation against double-precision CPU reference.
│   └── test_bank_conflicts.cu  # Empirical verification of swizzling efficacy.
├── profiling/
│   ├── nsight_compute_metrics.txt # NCU hardware counter configuration.
│   ├── roofline_model_analysis.py # Visualization of arithmetic intensity.
│   └── hardware_counters_log.csv  # Logged profiling data.
├── integration/
│   └── xla_cost_model_validation.py # Benchmarking against XLA HLO estimates.
├── Makefile                   # Targeted build system for sm_70.
└── README.md                  # Comprehensive documentation.
```

---

## 5. Build and Execution
**Prerequisites**: NVIDIA CUDA Toolkit 9.0+ and a Volta-architecture GPU (e.g., Tesla V100).

### 5.1. Compilation
To build the primary kernel and host launcher:
```bash
make
```

### 5.2. Validation
To run mathematical equivalence and bank conflict tests:
```bash
make tests
./bin/test_attention_exact
./bin/test_bank_conflicts
```

### 5.3. Profiling
To generate hardware telemetry for Roofline Analysis:
```bash
ncu --metrics `cat profiling/nsight_compute_metrics.txt | grep -v "#" | tr '\n' ','` ./bin/v100_fused_attention
python3 profiling/roofline_model_analysis.py
```

---

## 6. Macro-Ecosystem Integration
This project provides actionable evidence for optimizing hyperscale AI infrastructure:
* **OpenXLA Cost Models**: The analytical rigor applied here directly translates to improving XLA's High-Level Optimizer (HLO) cost models by accurately predicting device time based on register usage and occupancy.
* **Hardware Evolution**: Mastery of manual software pipelining on Volta prepares engineers for native asynchronous paradigms in emerging H100 (Hopper) and Blackwell architectures.