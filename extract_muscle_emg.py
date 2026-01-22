#!/usr/bin/env python3
"""
Extract muscle EMG signals from kinematics data and organize by gait cycles.

This script extracts specific muscles' activation levels from kinematics txt files,
segments them by gait cycles using phase information, and outputs as NPZ files.

Target muscles (right side):
- Gastrocnemius medialis (Medial Head)
- Tibialis anterior
- Soleus
- Vastus medialis
- Vastus lateralis
- Rectus femoris
- Biceps femoris
- Semitendinosus
- Gracilis
- Gluteus medius
- Right external oblique

Usage:
    python extract_muscle_emg.py
    (will prompt you to select a txt file)
    
Or:
    python extract_muscle_emg.py kinematics_data/phased_2450.txt --output muscle_emg.npz --target-length 30
"""

import numpy as np
import os
import argparse
from pathlib import Path
import glob


def parse_kinematics_file(filepath):
    """
    Parse the kinematics data file with phase information.
    
    Returns:
        tuple: (muscle_names, data, header_info, has_phase)
            - muscle_names: list of muscle names
            - data: (n_frames, n_features) array with all data (including phase if present)
            - header_info: dict with metadata
            - has_phase: bool indicating if phase column exists
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header_info = {}
    muscle_names = []
    has_phase = False
    
    for line in lines:
        if line.startswith('# Total frames:'):
            header_info['total_frames'] = int(line.split(':')[1].strip())
        elif line.startswith('# Control Hz:'):
            header_info['control_hz'] = int(line.split(':')[1].strip())
        elif line.startswith('# Time step:'):
            header_info['time_step'] = float(line.split(':')[1].split()[0])
        elif line.startswith('# Number of Muscles:'):
            header_info['num_muscles'] = int(line.split(':')[1].strip())
        elif line.startswith('# Muscle names:'):
            names_str = line.split(':')[1].strip()
            muscle_names = [m.strip() for m in names_str.split(',')]
        elif line.startswith('# Format:') and 'phase' in line.lower():
            has_phase = True
        elif line.startswith('#'):
            continue
        else:
            break
    
    # Parse data (skip header lines)
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    data = []
    expected_cols = None
    
    for line_idx, line in enumerate(data_lines):
        values = [float(x) for x in line.split()]
        
        # Check consistency
        if expected_cols is None:
            expected_cols = len(values)
        elif len(values) != expected_cols:
            print(f"Warning: Line {line_idx} has {len(values)} columns, expected {expected_cols}")
            continue
        
        data.append(values)
    
    data = np.array(data)
    
    return muscle_names, data, header_info, has_phase


def find_muscle_indices(muscle_names, target_muscles):
    """
    Find indices of target muscles in the muscle names list.
    If a muscle has multiple heads/parts on one side, return all of them.
    
    Args:
        muscle_names: list of all muscle names
        target_muscles: list of target muscle names to find (exact or partial matches)
        
    Returns:
        dict: {target_name: [list_of_indices], ...}
        dict: {target_name: [list_of_actual_names], ...}
    """
    indices = {}
    found_map = {}
    
    for target in target_muscles:
        matching_indices = []
        matching_names = []
        
        # Try exact match first
        exact_matches = [i for i, name in enumerate(muscle_names) if name == target]
        if exact_matches:
            matching_indices = exact_matches
            matching_names = [muscle_names[i] for i in exact_matches]
        else:
            # Try partial match (case-insensitive)
            for i, name in enumerate(muscle_names):
                if target.lower() in name.lower():
                    matching_indices.append(i)
                    matching_names.append(name)
        
        if matching_indices:
            indices[target] = matching_indices
            found_map[target] = matching_names
    
    missing = set(target_muscles) - set(indices.keys())
    if missing:
        print(f"Warning: Could not find muscles: {missing}")
    
    return indices, found_map


def detect_gait_cycles_by_phase(data, phase_diff_threshold=0.5):
    """
    Detect gait cycles using phase information from the last column.
    A complete gait cycle is detected when phase decreases significantly (phase reset).
    
    Args:
        data: (n_frames, n_features) array, last column is phase
        phase_diff_threshold: threshold for phase difference to detect reset (default 0.5)
                              When phase[i] - phase[i-1] < -phase_diff_threshold, it's a reset
        
    Returns:
        list: [(start_idx, end_idx), ...] for each detected cycle
    """
    phase = data[:, -1]  # Last column is phase
    
    cycles = []
    cycle_start = 0
    
    for i in range(1, len(phase)):
        # Calculate phase difference
        phase_diff = phase[i] - phase[i-1]
        
        # Detect phase reset: significant decrease in phase value
        if phase_diff < -phase_diff_threshold:
            cycle_end = i - 1
            if cycle_end > cycle_start:  # Valid cycle
                cycles.append((cycle_start, cycle_end))
            cycle_start = i
    
    # Add the last cycle if it exists
    if cycle_start < len(phase) - 1:
        cycles.append((cycle_start, len(phase) - 1))
    
    return cycles


def detect_gait_cycles(data, hz=30, min_cycle_frames=10):
    """
    Detect gait cycles by finding oscillations in COM acceleration or velocity.
    This is a fallback method when phase information is not available.
    
    Args:
        data: (n_frames, n_features) array
        hz: control frequency in Hz
        min_cycle_frames: minimum frames for a valid cycle
        
    Returns:
        list: [(start_idx, end_idx), ...] for each detected cycle
    """
    # The format is: frame_index(0) time(1) pos(2-57) vel(58-113) com_x(114) com_y(115) com_z(116) com_vel_x(117) com_vel_y(118) com_vel_z(119) muscle(120-423)
    # Use COM y velocity for cycle detection (forward motion causes oscillation)
    
    com_vel_y_idx = 118  # COM y velocity
    
    if data.shape[1] <= com_vel_y_idx:
        print(f"Warning: Data has {data.shape[1]} columns, COM velocity index {com_vel_y_idx} out of range")
        return []
    
    signal = data[:, com_vel_y_idx]
    
    # Smooth the signal to avoid detecting noise
    smoothed = np.convolve(signal, np.ones(5) / 5, mode='same')
    
    # Find peaks and valleys in smoothed signal
    # A gait cycle typically has the structure: valley -> peak -> valley
    # (i.e., deceleration -> acceleration -> deceleration)
    peaks = []
    valleys = []
    
    for i in range(2, len(smoothed) - 2):
        # Use wider comparison window to filter noise
        if (smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and 
            smoothed[i] > smoothed[i-2] and smoothed[i] > smoothed[i+2]):
            peaks.append(i)
        elif (smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1] and 
              smoothed[i] < smoothed[i-2] and smoothed[i] < smoothed[i+2]):
            valleys.append(i)
    
    # Create cycles from consecutive valleys (valley-to-valley is one gait cycle)
    cycles = []
    
    if len(valleys) >= 2:
        for i in range(len(valleys) - 1):
            start = valleys[i]
            end = valleys[i + 1]
            cycle_length = end - start
            if cycle_length >= min_cycle_frames:
                cycles.append((start, end))
    elif len(peaks) >= 2:
        # Fallback: use peaks if valleys not available
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]
            cycle_length = end - start
            if cycle_length >= min_cycle_frames:
                cycles.append((start, end))
    else:
        # Last fallback: use fixed-length sliding windows
        print(f"Warning: Only found {len(peaks)} peaks and {len(valleys)} valleys, using fixed-length cycles")
        cycle_length = max(40, hz * 1)  # Nominal gait cycle ~40-60 frames
        for start in range(0, len(signal) - cycle_length, cycle_length // 2):
            end = min(start + cycle_length, len(signal) - 1)
            if end - start >= min_cycle_frames:
                cycles.append((start, end))
    
    return cycles


def extract_gait_cycle_data(data, muscle_indices, cycle_pairs, target_length=200):
    """
    Extract muscle activation data for each gait cycle and interpolate to target length.
    If a muscle has multiple heads/parts, take the average.
    
    Args:
        data: (n_frames, n_features) array
        muscle_indices: {muscle_name: [list_of_data_column_indices], ...}
        cycle_pairs: [(start, end), ...] frame indices for each cycle
        target_length: desired number of frames per cycle
        
    Returns:
        numpy array: (n_cycles, target_length, n_muscles)
    """
    n_cycles = len(cycle_pairs)
    n_muscles = len(muscle_indices)
    
    # Get muscle names (ordered)
    muscle_names = list(muscle_indices.keys())
    
    # Initialize output array
    output = np.zeros((n_cycles, target_length, n_muscles))
    
    for cycle_idx, (start, end) in enumerate(cycle_pairs):
        cycle_data = data[start:end+1, :]  # (n_frames_in_cycle, n_features)
        
        # Interpolate to target length
        current_length = cycle_data.shape[0]
        
        if current_length < 2:
            print(f"Warning: Cycle {cycle_idx} too short, skipping")
            continue
        
        # Linear interpolation for each muscle
        original_indices = np.linspace(0, 1, current_length)
        target_indices = np.linspace(0, 1, target_length)
        
        for muscle_idx, muscle_name in enumerate(muscle_names):
            # Get all column indices for this muscle
            col_indices = muscle_indices[muscle_name]
            
            if len(col_indices) == 1:
                # Single muscle - use directly
                signal = cycle_data[:, col_indices[0]]
            else:
                # Multiple heads/parts - take average
                signals = [cycle_data[:, idx] for idx in col_indices]
                signal = np.mean(signals, axis=0)
            
            output[cycle_idx, :, muscle_idx] = np.interp(
                target_indices, 
                original_indices, 
                signal
            )
    
    return output, muscle_names


def normalize_emg_data(data, method='minmax'):
    """
    Normalize EMG data to 0-1 range.
    注意：两种方法都对每个肌肉独立归一化，不是跨肌肉的全局归一化
    
    Args:
        data: (n_cycles, n_frames, n_muscles) array
        method: normalization method
            - 'minmax': (x - min) / (max - min) for each muscle globally across all cycles
            - 'per_cycle': normalize each muscle independently within each cycle
            
    Returns:
        numpy array: normalized data with same shape
        dict: normalization parameters for each muscle (only for 'minmax' method)
    """
    n_cycles, n_frames, n_muscles = data.shape
    normalized_data = np.zeros_like(data)
    norm_params = {}
    
    if method == 'minmax':
        # Global min-max normalization per muscle (across all cycles)
        for muscle_idx in range(n_muscles):
            muscle_data = data[:, :, muscle_idx]  # (n_cycles, n_frames)
            min_val = np.min(muscle_data)
            max_val = np.max(muscle_data)
            
            # Avoid division by zero
            if max_val - min_val < 1e-10:
                # If all values are the same, set to 0.5
                normalized_data[:, :, muscle_idx] = 0.5
                norm_params[muscle_idx] = {'min': min_val, 'max': max_val, 'scale': 0}
            else:
                normalized_data[:, :, muscle_idx] = (muscle_data - min_val) / (max_val - min_val)
                norm_params[muscle_idx] = {'min': min_val, 'max': max_val, 'scale': max_val - min_val}
    
    elif method == 'per_cycle':
        # Normalize each muscle independently within each cycle
        for cycle_idx in range(n_cycles):
            for muscle_idx in range(n_muscles):
                muscle_cycle_data = data[cycle_idx, :, muscle_idx]  # (n_frames,)
                min_val = np.min(muscle_cycle_data)
                max_val = np.max(muscle_cycle_data)
                
                if max_val - min_val < 1e-10:
                    normalized_data[cycle_idx, :, muscle_idx] = 0.5
                else:
                    normalized_data[cycle_idx, :, muscle_idx] = (muscle_cycle_data - min_val) / (max_val - min_val)
    
    return normalized_data, norm_params


def apply_phase_offset(data, phase_offset):
    """
    Apply phase offset to gait cycle data by circular shifting.
    
    Args:
        data: (n_cycles, n_frames, n_muscles) array
        phase_offset: phase offset as fraction of cycle (0-1), or int frames
        
    Returns:
        numpy array: phase-shifted data with same shape
    """
    n_cycles, n_frames, n_muscles = data.shape
    
    # Convert phase offset (0-1) to number of frames
    if isinstance(phase_offset, float) and 0 <= phase_offset <= 1:
        shift_frames = int(phase_offset * n_frames)
    else:
        shift_frames = int(phase_offset) % n_frames
    
    shifted_data = np.zeros_like(data)
    
    for cycle_idx in range(n_cycles):
        for muscle_idx in range(n_muscles):
            shifted_data[cycle_idx, :, muscle_idx] = np.roll(
                data[cycle_idx, :, muscle_idx],
                shift_frames
            )
    
    return shifted_data


def select_txt_file():
    """
    Let user select a txt file from kinematics_data directory.
    
    Returns:
        str: path to selected file, or None if no files found or cancelled
    """
    # Look for txt files in current directory and kinematics_data directory
    search_paths = [
        '*.txt',
        'kinematics_data/*.txt',
        '../kinematics_data/*.txt',
    ]
    
    txt_files = []
    for pattern in search_paths:
        txt_files.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    txt_files = sorted(list(set(txt_files)))
    
    if not txt_files:
        print("Error: No txt files found in current or kinematics_data directory!")
        return None
    
    print("\n可用的txt文件:")
    print("=" * 60)
    for i, filepath in enumerate(txt_files, 1):
        filesize = os.path.getsize(filepath) / 1024  # KB
        print(f"{i:2d}. {filepath:40s} ({filesize:8.1f} KB)")
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"\n请选择文件编号 (1-{len(txt_files)}) 或按Enter键取消: ").strip()
            
            if not choice:
                print("已取消")
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(txt_files):
                selected_file = txt_files[idx]
                print(f"\n已选择: {selected_file}")
                return selected_file
            else:
                print(f"请输入1到{len(txt_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n已取消")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract muscle EMG signals from kinematics data using phase information'
    )
    parser.add_argument(
        'input_file',
        nargs='?',  # Make it optional
        help='Input kinematics text file (will prompt if not provided)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output NPZ filename (default: auto-generated from input filename)'
    )
    parser.add_argument(
        '--phase-offset',
        type=float,
        default=0.0,
        help='Gait phase offset (0-1 as fraction of cycle, default: 0.0)'
    )
    parser.add_argument(
        '--target-length',
        type=int,
        default=200,
        help='Target number of frames per cycle (default: 200)'
    )
    parser.add_argument(
        '--min-cycle-frames',
        type=int,
        default=10,
        help='Minimum frames for a valid cycle (default: 10, only used for non-phase detection)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize EMG data to 0-1 range'
    )
    parser.add_argument(
        '--norm-method',
        choices=['minmax', 'per_cycle'],
        default='minmax',
        help='Normalization method (default: minmax)'
    )
    parser.add_argument(
        '--phase-threshold',
        type=float,
        default=0.5,
        help='Phase difference threshold for cycle detection (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Select file if not provided
    if args.input_file is None:
        args.input_file = select_txt_file()
        if args.input_file is None:
            return
    
    # Check if file exists
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        return
    
    # Auto-generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = f"{input_path.stem}_muscle_emg.npz"
    
    # Target muscles - right side only
    # These will be matched to actual muscle names in the file
    # If multiple parts exist (e.g., Medial Head, Lateral Head), they will be averaged
    target_muscles = [
        'R_Gastrocnemius',          # Will match all R_Gastrocnemius* variants
        'R_Tibialis_Anterior',       # Will match R_Tibialis_Anterior and variants
        'R_Soleus',                  # Will match R_Soleus and R_Soleus1
        'R_Vastus_Medialis',         # Will match R_Vastus_Medialis* (1, 2, etc.)
        'R_Vastus_Lateralis',        # Will match R_Vastus_Lateralis* variants
        'R_Rectus_Femoris',          # Will match R_Rectus_Femoris and R_Rectus_Femoris1
        'R_Bicep_Femoris',           # Will match all R_Bicep_Femoris* (Longus, Short, Short1)
        'R_Semitendinosus',          # Will match R_Semitendinosus
        'R_Gracilis',                # Will match R_Gracilis
        'R_Gluteus_Medius',          # Will match R_Gluteus_Medius* (1, 2, 3, etc.)
    ]
    
    print(f"Loading kinematics data from: {args.input_file}")
    
    # Parse input file
    muscle_names, data, header_info, has_phase = parse_kinematics_file(args.input_file)
    
    print(f"Loaded {len(muscle_names)} muscles, {data.shape[0]} frames")
    print(f"Header info: {header_info}")
    print(f"Phase column detected: {has_phase}")
    
    # Find target muscle indices
    print("\nSearching for target muscles...")
    muscle_indices, found_map = find_muscle_indices(muscle_names, target_muscles)
    
    for target, actual_names in sorted(found_map.items()):
        indices = muscle_indices[target]
        if len(indices) == 1:
            print(f"  {target}")
            print(f"    -> {actual_names[0]} (index {indices[0]})")
        else:
            print(f"  {target}")
            print(f"    -> Found {len(indices)} parts, will average:")
            for name, idx in zip(actual_names, indices):
                print(f"       • {name} (index {idx})")
    
    # Detect gait cycles using phase or fallback method
    print(f"\nDetecting gait cycles...")
    if has_phase:
        print(f"Using phase-based detection (phase difference threshold: {args.phase_threshold})...")
        cycle_pairs = detect_gait_cycles_by_phase(data, phase_diff_threshold=args.phase_threshold)
    else:
        print(f"Phase column not found, using COM velocity-based detection (min {args.min_cycle_frames} frames)...")
        cycle_pairs = detect_gait_cycles(data, hz=header_info['control_hz'], 
                                        min_cycle_frames=args.min_cycle_frames)
    
    print(f"Found {len(cycle_pairs)} gait cycles")
    for i, (start, end) in enumerate(cycle_pairs[:5]):
        print(f"  Cycle {i}: frames {start}-{end} ({end-start+1} frames)")
    if len(cycle_pairs) > 5:
        print(f"  ... and {len(cycle_pairs)-5} more cycles")
    
    if len(cycle_pairs) == 0:
        print("Error: No gait cycles detected!")
        return
    
    # Extract muscle data for each cycle
    print(f"\nExtracting muscle data (target length: {args.target_length} frames)...")
    cycle_data, extracted_muscles = extract_gait_cycle_data(
        data, muscle_indices, cycle_pairs, 
        target_length=args.target_length
    )
    
    print(f"Extracted {len(cycle_data)} gait cycles")
    print(f"Extracted data shape: {cycle_data.shape}")
    print(f"Extracted muscles: {extracted_muscles}")
    print(f"Data range before normalization: [{np.min(cycle_data):.6f}, {np.max(cycle_data):.6f}]")
    
    # Normalize if requested
    norm_params = None
    if args.normalize:
        print(f"\nNormalizing to 0-1 range (method: {args.norm_method})...")
        cycle_data, norm_params = normalize_emg_data(cycle_data, method=args.norm_method)
        print(f"Data range after normalization: [{np.min(cycle_data):.6f}, {np.max(cycle_data):.6f}]")
        if norm_params:
            print("Normalization parameters (per muscle):")
            for muscle_idx, params in norm_params.items():
                muscle_name = extracted_muscles[muscle_idx]
                print(f"  {muscle_name}: min={params['min']:.6f}, max={params['max']:.6f}")
    
    # Apply phase offset if specified
    if args.phase_offset != 0:
        print(f"\nApplying phase offset: {args.phase_offset}")
        cycle_data = apply_phase_offset(cycle_data, args.phase_offset)
    
    # Save to NPZ file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_path}")
    
    # Prepare base save dictionary
    def prepare_save_dict(data, norm_params=None):
        save_dict = {
            'emg_data': data,
            'muscle_names': np.array(extracted_muscles),
            'cycle_pairs': np.array(cycle_pairs),
            'header_info': str(header_info),
            'phase_offset': args.phase_offset,
            'target_length': args.target_length
        }
        
        if args.normalize:
            save_dict['normalized'] = True
            save_dict['norm_method'] = args.norm_method
            if norm_params:
                for muscle_idx, params in norm_params.items():
                    save_dict[f'norm_params_muscle_{muscle_idx}'] = str(params)
        
        return save_dict
    
    # Save single file with merged cycles
    save_dict = prepare_save_dict(cycle_data, norm_params)
    np.savez(str(output_path), **save_dict)
    
    print("Done!")
    print(f"Output shape: (n_cycles={cycle_data.shape[0]}, n_frames={cycle_data.shape[1]}, n_muscles={cycle_data.shape[2]})")


if __name__ == '__main__':
    main()
