#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两组仿真数据的关节角度
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 配置matplotlib使用英文标签避免中文乱码
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_sim_data(filepath):
    """加载仿真数据"""
    data = []
    dof_names = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 解析DOF名称 - 从"# DOF names:"行直接提取
    for line in lines:
        if '# DOF names:' in line:
            # 直接从这一行提取DOF名称
            names_part = line.split('# DOF names:')[1].strip()
            dof_names = [name.strip() for name in names_part.split(',')]
            break
    
    # 读取数据
    for line in lines:
        if not line.startswith('#') and line.strip():
            values = line.strip().split()
            if len(values) > 2:  # 确保有足够的列
                data.append([float(v) for v in values])
    
    data = np.array(data)
    print(f"Simulation data shape: {data.shape}")
    print(f"Number of DOFs: {len(dof_names)}")
    if len(dof_names) > 0:
        print(f"First 5 DOFs: {dof_names[:5]}")
    
    # 提取: frame_index, time, positions(56), velocities(56), COM(6)
    result = {
        'frame': data[:, 0],
        'time': data[:, 1],
        'positions': data[:, 2:58],
        'velocities': data[:, 58:114],
        'com': data[:, 114:],
        'dof_names': dof_names
    }
    
    return result

def find_dof_index(dof_names, pattern):
    """查找包含特定模式的DOF索引"""
    indices = []
    for i, name in enumerate(dof_names):
        if pattern.lower() in name.lower():
            indices.append((i, name))
    return indices

def resample_data(x, y, target_length):
    """重采样数据到目标长度"""
    x_new = np.linspace(x[0], x[-1], target_length)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new

def plot_comparison(sim1_data, sim2_data, joint_name, sim1_indices, sim2_indices, 
                   save_path, axes_names=['X', 'Y', 'Z'], label1='Sim1', label2='Sim2'):
    """绘制两组仿真数据的关节角度对比图"""
    n_axes = len(sim1_indices)
    
    fig, axs = plt.subplots(n_axes, 1, figsize=(12, 4*n_axes))
    if n_axes == 1:
        axs = [axs]
    
    fig.suptitle(f'{joint_name} Angle Comparison ({label1} vs {label2})', fontsize=16, fontweight='bold')
    
    for i, (sim1_idx, sim2_idx) in enumerate(zip(sim1_indices, sim2_indices)):
        ax = axs[i]
        
        # 仿真数据1
        if sim1_idx is not None:
            sim1_values = sim1_data['positions'][:, sim1_idx]
            sim1_time = sim1_data['time']
            sim1_name = sim1_data['dof_names'][sim1_idx]
            sim1_values_deg = np.degrees(sim1_values)
        else:
            sim1_values_deg = np.array([])
            sim1_time = np.array([])
            sim1_name = "Not Found"
        
        # 仿真数据2
        if sim2_idx is not None:
            sim2_values = sim2_data['positions'][:, sim2_idx]
            sim2_time = sim2_data['time']
            sim2_name = sim2_data['dof_names'][sim2_idx]
            sim2_values_deg = np.degrees(sim2_values)
        else:
            sim2_values_deg = np.array([])
            sim2_time = np.array([])
            sim2_name = "Not Found"
        
        # 绘制
        if len(sim1_time) > 0:
            ax.plot(sim1_time, sim1_values_deg, 'b-', label=f'{label1} ({sim1_name})', 
                    linewidth=2, alpha=0.7)
        
        if len(sim2_time) > 0:
            ax.plot(sim2_time, sim2_values_deg, 'r-', label=f'{label2} ({sim2_name})', 
                    linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.set_title(f'{joint_name} - {axes_names[i]}-axis', fontsize=13)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    # 文件路径 - 对比两组仿真数据
    sim1_file = 'kinematics_data/kinematics_20251208_175626_0000.txt'  # 第一组仿真（无外骨骼）
    sim2_file = 'kinematics_data/kinematics_20251209_164939_0000.txt'  # 第二组仿真（有外骨骼1000N）
    output_dir = Path('comparison_plots_lower_limbs')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading simulation data...")
    sim1_data = load_sim_data(sim1_file)
    sim2_data = load_sim_data(sim2_file)
    
    # 过滤仿真数据：只取0-5秒的数据（相对于开始时间）
    for sim_data in [sim1_data, sim2_data]:
        start_time = sim_data['time'][0]
        relative_start = start_time + 0.0
        relative_end = start_time + 5.0
        time_mask = (sim_data['time'] >= relative_start) & (sim_data['time'] <= relative_end)
        sim_data['time'] = sim_data['time'][time_mask]
        sim_data['positions'] = sim_data['positions'][time_mask]
        sim_data['velocities'] = sim_data['velocities'][time_mask]
        sim_data['com'] = sim_data['com'][time_mask]
        sim_data['frame'] = sim_data['frame'][time_mask]
        # 将时间轴归零：从0开始
        sim_data['time'] = sim_data['time'] - sim_data['time'][0]
    
    print(f"\nSimulation 1 data shape: {sim1_data['positions'].shape}")
    print(f"Simulation 1 time range: {sim1_data['time'][0]:.2f}s - {sim1_data['time'][-1]:.2f}s")
    print(f"Simulation 2 data shape: {sim2_data['positions'].shape}")
    print(f"Simulation 2 time range: {sim2_data['time'][0]:.2f}s - {sim2_data['time'][-1]:.2f}s")
    
    # 定义对比项 - 左侧下肢所有关节（从骨盆到脚趾）
    comparisons = [
        {
            'name': 'Pelvis',
            'sim_patterns': ['Pelvis_rot_x', 'Pelvis_rot_y', 'Pelvis_rot_z'],
            'axes': ['X (Anterior/Posterior Tilt)', 'Y (Obliquity)', 'Z (Rotation)']
        },
        {
            'name': 'Pelvis_Position',
            'sim_patterns': ['Pelvis_pos_x', 'Pelvis_pos_y', 'Pelvis_pos_z'],
            'axes': ['X (Forward/Backward)', 'Y (Up/Down)', 'Z (Left/Right)']
        },
        {
            'name': 'Left_Hip',
            'sim_patterns': ['FemurL_x', 'FemurL_y', 'FemurL_z'],
            'axes': ['X (Flexion/Extension)', 'Y (Abduction/Adduction)', 'Z (Internal/External Rotation)']
        },
        {
            'name': 'Left_Knee',
            'sim_patterns': ['TibiaL'],
            'axes': ['Flexion/Extension']
        },
        {
            'name': 'Left_Ankle',
            'sim_patterns': ['TalusL_x', 'TalusL_y', 'TalusL_z'],
            'axes': ['X (Dorsiflexion/Plantarflexion)', 'Y (Inversion/Eversion)', 'Z (Rotation)']
        },
        {
            'name': 'Left_FootPinky',
            'sim_patterns': ['FootPinkyL'],
            'axes': ['MTP Flexion/Extension']
        },
        {
            'name': 'Left_FootThumb',
            'sim_patterns': ['FootThumbL'],
            'axes': ['MTP Flexion/Extension']
        },
    ]
    
    print("\nGenerating comparison plots...")
    
    for comp in comparisons:
        print(f"\nProcessing: {comp['name']}")
        
        # 查找仿真数据1中对应的DOF
        sim1_indices = []
        for pattern in comp['sim_patterns']:
            found = find_dof_index(sim1_data['dof_names'], pattern)
            if found:
                sim1_indices.append(found[0][0])
                print(f"  Sim1 found DOF: {found[0][1]}")
            else:
                sim1_indices.append(None)
                print(f"  Sim1 DOF not found: {pattern}")
        
        # 查找仿真数据2中对应的DOF
        sim2_indices = []
        for pattern in comp['sim_patterns']:
            found = find_dof_index(sim2_data['dof_names'], pattern)
            if found:
                sim2_indices.append(found[0][0])
                print(f"  Sim2 found DOF: {found[0][1]}")
            else:
                sim2_indices.append(None)
                print(f"  Sim2 DOF not found: {pattern}")
        
        # 生成图像
        save_path = output_dir / f"{comp['name'].replace(' ', '_')}.png"
        plot_comparison(
            sim1_data, 
            sim2_data, 
            comp['name'],
            sim1_indices,
            sim2_indices,
            save_path,
            comp['axes'],
            label1='No Exoskeleton',
            label2='Exoskeleton 1000N'
        )
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nDone!")

if __name__ == '__main__':
    main()
