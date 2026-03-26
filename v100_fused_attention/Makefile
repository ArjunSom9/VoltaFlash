# ============================================================================
# Bare-Metal Fused Attention Kernel for NVIDIA Tesla V100
# ============================================================================

# Compiler and Core Flags
NVCC = nvcc
# Targeting sm_70 for Volta V100, -O3 for maximum optimization, -I for includes
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14 -I./include -Xcompiler -Wall

# Optional: Add -lineinfo for Nsight Compute profiling to map assembly to source lines
PROFILE_FLAGS = -lineinfo

# Directory Structure
SRC_DIR = src
INC_DIR = include
TEST_DIR = tests
BIN_DIR = bin

# Source Files and Headers
MAIN_SOURCES = $(SRC_DIR)/main.cu \
               $(SRC_DIR)/fused_attention_kernel.cu \
               $(SRC_DIR)/host_utils.cpp

# Utilize wildcard to track all headers so modifications trigger recompilation
HEADERS = $(wildcard $(INC_DIR)/*.cuh) $(wildcard $(INC_DIR)/*.h)

# Executable Targets
TARGET = $(BIN_DIR)/v100_fused_attention
TEST_EXACT = $(BIN_DIR)/test_attention_exact
TEST_BANK = $(BIN_DIR)/test_bank_conflicts

# ============================================================================
# Build Rules
# ============================================================================

# Phony targets to prevent conflicts with files of the same name
.PHONY: all clean tests setup profile_build

# Default target
all: setup $(TARGET)

# Creates the binary output directory
setup:
	@mkdir -p $(BIN_DIR)

# Rule for building the main application
$(TARGET): $(MAIN_SOURCES) $(HEADERS)
	@echo "[BUILD] Compiling Main Fused Attention Kernel for sm_70..."
	$(NVCC) $(NVCC_FLAGS) $(MAIN_SOURCES) -o $@
	@echo "[SUCCESS] Main executable generated at $@"

# Target specifically for profiling (adds lineinfo for PC sampling in Nsight Compute)
profile_build: setup
	@echo "[BUILD] Compiling Main Kernel with Profiling Telemetry..."
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) $(MAIN_SOURCES) -o $(TARGET)_profile
	@echo "[SUCCESS] Profiling executable generated at $(TARGET)_profile"

# ============================================================================
# Test Build Rules
# ============================================================================

# Build all tests
tests: setup $(TEST_EXACT) $(TEST_BANK)

# Rule for the Exact Mathematical Equivalence Test
# Needs the main kernel source to compare against standard naive attention
$(TEST_EXACT): $(TEST_DIR)/test_attention_exact.cu $(SRC_DIR)/fused_attention_kernel.cu $(SRC_DIR)/host_utils.cpp $(HEADERS)
	@echo "[BUILD] Compiling Exact Attention Equivalence Test..."
	$(NVCC) $(NVCC_FLAGS) $< $(SRC_DIR)/fused_attention_kernel.cu $(SRC_DIR)/host_utils.cpp -o $@

# Rule for the XOR Bank Conflict Emulation Test
# Generally standalone, testing only the xor_swizzle logic against shared memory arrays
$(TEST_BANK): $(TEST_DIR)/test_bank_conflicts.cu $(SRC_DIR)/host_utils.cpp $(HEADERS)
	@echo "[BUILD] Compiling Bank Conflict & Swizzle Test..."
	$(NVCC) $(NVCC_FLAGS) $< $(SRC_DIR)/host_utils.cpp -o $@

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "[CLEAN] Removing build artifacts..."
	rm -rf $(BIN_DIR)
	@echo "[SUCCESS] Clean complete."