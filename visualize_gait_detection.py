#!/usr/bin/env python3
"""
Visualize gait cycle detection algorithm.
Shows COM velocity signal, smoothing, peak/valley detection, and cycle boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from pathlib import Path


rcParams['figure.facecolor'] = 'white'
rcParams['font.size'] = 10


def parse_kinematics_file(filepath):
    """Parse kinematics data file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    header_info = {}
    for line in lines:
        if line.startswith('# Total frames:'):
            header_info['total_frames'] = int(line.split(':')[1].strip())
        elif line.startswith('# Control Hz:'):
            header_info['control_hz'] = int(line.split(':')[1].strip())
        elif line.startswith('#'):
            continue
        else:
            break
    
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    data = np.array([[float(x) for x in line.split()] for line in data_lines])
    
    return data, header_info


def detect_gait_cycles_verbose(data, hz=30, min_cycle_frames=10):
    """
    Detect gait cycles with detailed output for visualization.
    """
    com_vel_y_idx = 118
    signal = data[:, com_vel_y_idx]
    
    # Smoothing
    smoothed = np.convolve(signal, np.ones(5) / 5, mode='same')
    
    # Peak/valley detection
    peaks = []
    valleys = []
    
    for i in range(2, len(smoothed) - 2):
        if (smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and 
            smoothed[i] > smoothed[i-2] and smoothed[i] > smoothed[i+2]):
            peaks.append(i)
        elif (smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1] and 
              smoothed[i] < smoothed[i-2] and smoothed[i] < smoothed[i+2]):
            valleys.append(i)
    
    # Create cycles
    cycles = []
    if len(valleys) >= 2:
        for i in range(len(valleys) - 1):
            start = valleys[i]
            end = valleys[i + 1]
            cycle_length = end - start
            if cycle_length >= min_cycle_frames:
                cycles.append((start, end))
    elif len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]
            cycle_length = end - start
            if cycle_length >= min_cycle_frames:
                cycles.append((start, end))
    
    return signal, smoothed, peaks, valleys, cycles


def plot_gait_cycle_detection(data, hz=30, output_file=None):
    """
    Create comprehensive visualization of gait cycle detection.
    """
    print("Detecting gait cycles...")
    signal, smoothed, peaks, valleys, cycles = detect_gait_cycles_verbose(data, hz=hz)
    
    print(f"Found {len(peaks)} peaks, {len(valleys)} valleys")
    print(f"Detected {len(cycles)} complete gait cycles")
    print(f"Cycle lengths range: {min(end-start for start, end in cycles)}-{max(end-start for start, end in cycles)} frames")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Full signal overview
    ax1 = plt.subplot(3, 2, 1)
    frames = np.arange(len(signal))
    ax1.plot(frames, signal, label='Raw signal', color='lightblue', linewidth=1.5)
    ax1.plot(frames, smoothed, label='Smoothed (5-frame MA)', color='darkblue', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('COM y velocity')
    ax1.set_title('1. COM Y Velocity Signal', fontweight='bold', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 2. Peaks and valleys
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(frames, smoothed, label='Smoothed signal', color='gray', linewidth=1.5, alpha=0.6)
    ax2.scatter(peaks, [smoothed[p] for p in peaks], color='red', s=100, marker='^', 
               label=f'Peaks (n={len(peaks)})', zorder=5)
    ax2.scatter(valleys, [smoothed[v] for v in valleys], color='green', s=100, marker='v',
               label=f'Valleys (n={len(valleys)})', zorder=5)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Signal value')
    ax2.set_title('2. Peak and Valley Detection', fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # 3. Cycle boundaries (first 300 frames)
    ax3 = plt.subplot(3, 2, 3)
    max_frames = min(300, len(signal))
    ax3.plot(frames[:max_frames], smoothed[:max_frames], color='black', linewidth=1.5)
    
    # Color background for different cycles
    colors = plt.cm.Set3(np.linspace(0, 1, len(cycles)))
    for cycle_idx, (start, end) in enumerate(cycles):
        if start >= max_frames:
            break
        end_plot = min(end, max_frames)
        ax3.axvspan(start, end_plot, alpha=0.3, color=colors[cycle_idx % len(colors)])
    
    # Mark cycle boundaries
    cycle_starts = [c[0] for c in cycles if c[0] < max_frames]
    ax3.vlines(cycle_starts, smoothed[:max_frames].min(), smoothed[:max_frames].max(), 
              colors='red', linestyles='--', linewidth=2, alpha=0.6, label='Cycle boundaries')
    
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('COM y velocity')
    ax3.set_title('3. Gait Cycle Segmentation (First 300 frames)', fontweight='bold', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # 4. Cycle length distribution
    ax4 = plt.subplot(3, 2, 4)
    cycle_lengths = [end - start for start, end in cycles]
    ax4.hist(cycle_lengths, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(cycle_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cycle_lengths):.1f}')
    ax4.set_xlabel('Cycle length (frames)')
    ax4.set_ylabel('Count')
    ax4.set_title('4. Cycle Length Distribution', fontweight='bold', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=9)
    
    # 5. Peak-to-valley time
    ax5 = plt.subplot(3, 2, 5)
    if len(peaks) > 0 and len(valleys) > 0:
        # For each valley, find nearest peak and calculate distance
        pv_distances = []
        for v_idx, valley in enumerate(valleys[:-1]):
            next_valley = valleys[v_idx + 1]
            # Find peak between valleys
            peaks_between = [p for p in peaks if valley < p < next_valley]
            if peaks_between:
                peak = peaks_between[0]
                v_to_p = peak - valley
                p_to_v = next_valley - peak
                pv_distances.append((v_to_p, p_to_v))
        
        if pv_distances:
            v_to_p_list = [x[0] for x in pv_distances]
            p_to_v_list = [x[1] for x in pv_distances]
            
            x_pos = np.arange(len(v_to_p_list))
            width = 0.35
            ax5.bar(x_pos - width/2, v_to_p_list, width, label='Valley→Peak', color='green', alpha=0.7)
            ax5.bar(x_pos + width/2, p_to_v_list, width, label='Peak→Valley', color='red', alpha=0.7)
            ax5.set_xlabel('Cycle index')
            ax5.set_ylabel('Frames')
            ax5.set_title('5. Phase Duration (Valley→Peak vs Peak→Valley)', fontweight='bold', fontsize=11)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Algorithm summary
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    summary_text = f"""
GAIT CYCLE DETECTION ALGORITHM SUMMARY

Data Information:
  • Total frames: {len(signal)}
  • Sampling rate: {hz} Hz
  • Duration: {len(signal)/hz:.2f} seconds
  • Signal used: COM y velocity (index 118)

Processing Steps:
  1. Smoothing: 5-frame moving average
  2. Peak detection: local maxima (±2 frames)
  3. Valley detection: local minima (±2 frames)
  4. Cycle definition: valley → next valley

Results:
  • Peaks detected: {len(peaks)}
  • Valleys detected: {len(valleys)}
  • Complete cycles: {len(cycles)}
  • Avg cycle length: {np.mean(cycle_lengths) if cycle_lengths else 0:.1f} frames
  • Min cycle length: {min(cycle_lengths) if cycle_lengths else 0} frames
  • Max cycle length: {max(cycle_lengths) if cycle_lengths else 0} frames

Physiological Interpretation:
  • Each cycle ≈ {np.mean(cycle_lengths)/hz if cycle_lengths else 0:.2f} seconds
  • Gait speed ≈ {np.mean(cycle_lengths)/hz if cycle_lengths else 0:.2f} sec/cycle

Next Steps:
  • Extract muscle EMG for each cycle
  • Interpolate to uniform 200 frames
  • Normalize activation (0-1)
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    
    return fig, cycles


def main():
    parser = argparse.ArgumentParser(description='Visualize gait cycle detection')
    parser.add_argument('input_file', help='Input kinematics text file')
    parser.add_argument('--output', '-o', help='Output PNG file')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}")
        return 1
    
    print(f"Loading: {args.input_file}")
    data, header_info = parse_kinematics_file(args.input_file)
    
    print(f"Data shape: {data.shape}")
    print(f"Control Hz: {header_info['control_hz']}")
    
    output_file = args.output or f"{Path(args.input_file).stem}_gait_cycles_detection.png"
    
    fig, cycles = plot_gait_cycle_detection(data, hz=header_info['control_hz'], output_file=output_file)
    
    # Print cycle info
    print(f"\nDetected {len(cycles)} gait cycles:")
    for i, (start, end) in enumerate(cycles[:10]):
        print(f"  Cycle {i:2d}: frames {start:3d}-{end:3d} (length={end-start:2d})")
    if len(cycles) > 10:
        print(f"  ... and {len(cycles)-10} more cycles")
    
    plt.show()
    
    return 0


if __name__ == '__main__':
    exit(main())
