import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# Hardware Constants: NVIDIA Tesla V100 (Volta GV100)
# ============================================================================
PEAK_BW_GBS = 900.0      # HBM2 Peak Bandwidth (GB/s)
PEAK_COMPUTE_TFLOPS = 125.0  # Tensor Core Peak (TFLOPS)
RIDGE_POINT = (PEAK_COMPUTE_TFLOPS * 1000.0) / PEAK_BW_GBS  # ~138.8 FLOPs/byte

def analyze_performance(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found at {csv_path}")
        return

    # Load Nsight Compute CSV data
    df = pd.read_csv(csv_path)
    
    # Helper to extract metric values
    def get_val(name):
        try:
            return float(df[df['Metric Name'] == name]['Metric Value'].values[0])
        except IndexError:
            print(f"[WARNING] Metric {name} missing from CSV.")
            return 0.0

    # 1. Calculate Total Floating Point Operations (FLOPs)
    # FFMA (Fused Multiply-Add) counts as 2 operations
    fadd = get_val("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
    fmul = get_val("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
    ffma = get_val("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
    total_flops = fadd + fmul + (2 * ffma)

    # 2. Calculate Total DRAM Traffic (Bytes)
    bytes_read = get_val("dram__bytes_read.sum")
    bytes_write = get_val("dram__bytes_write.sum")
    total_bytes = bytes_read + bytes_write

    # 3. Derive Roofline Coordinates
    # Arithmetic Intensity (AI) = FLOPs / Byte
    empirical_ai = total_flops / total_bytes if total_bytes > 0 else 0
    
    # Achieved Performance
    # We use the SM Throughput % to estimate achieved TFLOPS relative to peak
    sm_throughput_pct = get_val("sm__throughput.avg.pct_of_peak")
    achieved_tflops = (sm_throughput_pct / 100.0) * PEAK_COMPUTE_TFLOPS

    # ------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    
    # Create the Roofline boundaries
    ai_range = np.logspace(-1, 4, 1000)
    memory_roof = PEAK_BW_GBS * ai_range / 1000.0 # Convert GB/s * AI to TFLOPS
    compute_roof = np.full_like(ai_range, PEAK_COMPUTE_TFLOPS)
    
    # Plot the Roofline (The "Roof")
    actual_roof = np.minimum(memory_roof, compute_roof)
    plt.loglog(ai_range, actual_roof, label='V100 Theoretical Roofline', color='black', lw=2)
    
    # Plot the Ridge Point
    plt.axvline(x=RIDGE_POINT, color='red', linestyle='--', alpha=0.5, label=f'Ridge Point ({RIDGE_POINT:.1f})')

    # Plot our Kernel Point
    plt.scatter([empirical_ai], [achieved_tflops], color='blue', s=100, edgecolors='black', zorder=5, label='Fused Attention Kernel')
    plt.text(empirical_ai * 1.2, achieved_tflops, f'AI: {empirical_ai:.2f}\nPerf: {achieved_tflops:.1f} TFLOPS', fontweight='bold')

    # Formatting
    plt.title('Empirical Roofline Analysis: Fused Attention on Tesla V100', fontsize=14)
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Performance (TFLOPS)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='lower right')
    plt.ylim(0.1, 200)
    plt.xlim(0.1, 10000)

    # Print Summary Analysis
    print("======================================================")
    print(" Performance Analysis Summary ")
    print("======================================================")
    print(f"Total FLOPs         : {total_flops / 1e9:.2f} GFLOPs")
    print(f"Total DRAM Traffic  : {total_bytes / 1e6:.2f} MB")
    print(f"Empirical AI        : {empirical_ai:.2f} FLOPs/byte")
    print(f"Achieved Performance: {achieved_tflops:.2f} TFLOPS")
    print("------------------------------------------------------")
    
    if empirical_ai > RIDGE_POINT:
        print("[STATUS] COMPUTE-BOUND: The kernel is utilizing the Tensor Cores effectively.")
        print("[ANALYSIS] Optimization goal achieved. Memory latency is hidden.")
    else:
        print("[STATUS] MEMORY-BOUND: The kernel is limited by HBM2 bandwidth.")
        print("[ANALYSIS] Increase tiling size or check software pipelining logic.")
    print("======================================================")

    plt.savefig('profiling/v100_roofline_model.png')
    print("\n[SUCCESS] Roofline plot saved to profiling/v100_roofline_model.png")
    plt.show()

if __name__ == "__main__":
    analyze_performance('profiling/hardware_counters_log.csv')