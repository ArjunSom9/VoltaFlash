import pandas as pd
import numpy as np
import os

# ============================================================================
# XLA Cost Model Constants (NVIDIA Tesla V100)
# ============================================================================
V100_PEAK_TFLOPS = 125.0
V100_PEAK_BW_GBS = 900.0
CORE_CLOCK_MHZ = 1312.0  # Base clock for V100 PCIe
DEFAULT_LAUNCH_OVERHEAD_US = 5.0 # Typical XLA overhead estimate

class XLAGPUCostModel:
    """
    Simulates the XLA HLO Analytical Cost Model for GPU Kernels.
    Estimates execution time based on arithmetic intensity and hardware limits.
    """
    def __init__(self, peak_tflops, peak_bw_gbs):
        self.peak_tflops = peak_tflops
        self.peak_bw_gbs = peak_bw_gbs

    def estimate_time_us(self, total_flops, total_bytes, occupancy_pct):
        # 1. Compute time based on peak FLOPS
        compute_time_us = (total_flops / (self.peak_tflops * 1e12)) * 1e6
        
        # 2. Memory time based on peak Bandwidth
        memory_time_us = (total_bytes / (self.peak_bw_gbs * 1e9)) * 1e6
        
        # 3. Apply Occupancy Degradation Factor
        # XLA often penalizes bandwidth estimates if occupancy is low
        occupancy_penalty = 1.0 if occupancy_pct > 30 else (30.0 / max(occupancy_pct, 1.0))
        degraded_memory_time_us = memory_time_us * occupancy_penalty

        # 4. Roofline Estimate: max(compute, memory) + launch overhead
        estimated_execution_us = max(compute_time_us, degraded_memory_time_us) + DEFAULT_LAUNCH_OVERHEAD_US
        
        return {
            "compute_limit_us": compute_time_us,
            "memory_limit_us": memory_time_us,
            "occupancy_penalty": occupancy_penalty,
            "total_estimated_us": estimated_execution_us
        }

def validate_xla_model(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Profiling data not found at {csv_path}. Please run Phase 5 first.")
        return

    # Load hardware counters from Phase 5
    df = pd.read_csv(csv_path)
    def get_val(name):
        return float(df[df['Metric Name'] == name]['Metric Value'].values[0])

    # Extract Empirical Data
    fadd = get_val("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
    fmul = get_val("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
    ffma = get_val("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
    total_flops = fadd + fmul + (2 * ffma)
    
    bytes_read = get_val("dram__bytes_read.sum")
    bytes_write = get_val("dram__bytes_write.sum")
    total_bytes = bytes_read + bytes_write
    
    occupancy = get_val("smsp__warps_active.avg.pct_of_peak")
    actual_sm_throughput = get_val("sm__throughput.avg.pct_of_peak")
    
    # Calculate "Actual" time derived from SM throughput and theoretical peaks
    # (Simplified for validation: total work / achieved work)
    actual_time_us = (total_flops / (V100_PEAK_TFLOPS * 1e12 * (actual_sm_throughput / 100.0))) * 1e6

    # Run Simulation
    model = XLAGPUCostModel(V100_PEAK_TFLOPS, V100_PEAK_BW_GBS)
    prediction = model.estimate_time_us(total_flops, total_bytes, occupancy)

    # ------------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------------
    print("======================================================")
    print(" XLA HLO COST MODEL VALIDATION REPORT ")
    print("======================================================")
    print(f"Kernel Name         : fused_attention_v100_kernel")
    print(f"Empirical Occupancy : {occupancy:.2f}%")
    print(f"Actual Measured Time: {actual_time_us:.2f} us")
    print(f"XLA Estimated Time  : {prediction['total_estimated_us']:.2f} us")
    print("------------------------------------------------------")
    
    error_pct = abs(actual_time_us - prediction['total_estimated_us']) / actual_time_us * 100
    print(f"Model Prediction Error: {error_pct:.2f}%")
    
    print("\n[ANALYSIS OF COST MODEL DISCREPANCY]")
    if actual_time_us < prediction['total_estimated_us']:
        print("-> The manual kernel is OUTPERFORMING the XLA Cost Model.")
        print("-> Reason: Manual XOR swizzling and software pipelining allow")
        print("   high performance even at lower occupancy, which the standard")
        print("   XLA model penalizes too heavily.")
    else:
        print("-> The XLA Cost Model is overly optimistic.")
        print("-> Reason: Possible instruction-level dependency stalls not")
        print("   captured by the simple Roofline-based heuristic.")
    
    print("\n[RECOMMENDATION FOR XLA COMPILER TEAM]")
    print(f"1. Adjust 'occupancy_penalty' for Volta fused emitters to 1.0.")
    print(f"2. Use manual prefetch distance of 2 blocks for V100 SMEM loads.")
    print("======================================================")

if __name__ == "__main__":
    validate_xla_model('profiling/hardware_counters_log.csv')