# Makefile for compiling the multi-file CUDA project targeting Volta Architecture

NVCC = nvcc
# Targeting sm_70 for NVIDIA Tesla V100 (Volta)
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14

# Source files and targets
TARGET = v100_fused_attention
SOURCES = main.cu kernel.cu
HEADERS = config.h utils.cuh kernel.cuh

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)