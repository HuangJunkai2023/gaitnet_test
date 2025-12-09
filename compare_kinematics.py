#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比真实病人数据和仿真数据的关节角度
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

def load_real_data(filepath):
    """加载真实病人数据 (C3D导出的TSV格式)"""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # 跳过前5行的头信息，第6行开始是数据
    for line in lines[5:]:
        values = line.strip().split('\t')
        if values and values[0]:  # 确保不是空行
            try:
                # 将所有数值转换为float，跳过第一列(ITEM编号)
                row = []
                for v in values[1:]:
                    try:
                        val = float(v)
                        row.append(val)
                    except:
                        pass  # 跳过无法转换的值
                if len(row) > 0:
                    data.append(row)
            except Exception as e:
                print(f"Warning: Skipping line - {e}")
                continue
    
    # 检查每行长度
    row_lengths = [len(row) for row in data]
    print(f"Row lengths: min={min(row_lengths)}, max={max(row_lengths)}, rows={len(data)}")
    
    # 找出最长的行长度，并padding短行
    max_len = max(row_lengths)
    padded_data = []
    for row in data:
        if len(row) < max_len:
            row = row + [np.nan] * (max_len - len(row))
        padded_data.append(row)
    
    data_array = np.array(padded_data, dtype=float)
    print(f"Real data shape: {data_array.shape}")
    return data_array

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

def plot_comparison(real_data, sim_data, joint_name, real_indices, sim_indices, 
                   save_path, axes_names=['X', 'Y', 'Z']):
    """绘制关节角度对比图"""
    n_axes = len(real_indices)
    
    fig, axs = plt.subplots(n_axes, 1, figsize=(12, 4*n_axes))
    if n_axes == 1:
        axs = [axs]
    
    fig.suptitle(f'{joint_name} Angle Comparison (Real vs Simulation)', fontsize=16, fontweight='bold')
    
    for i, (real_idx, sim_idx) in enumerate(zip(real_indices, sim_indices)):
        ax = axs[i]
        
        # 真实数据
        real_values = real_data[:, real_idx]
        real_time = np.arange(len(real_values)) / 100.0  # 假设100Hz采样
        
        # 仿真数据
        if sim_idx is not None:
            sim_values = sim_data['positions'][:, sim_idx]
            sim_time = sim_data['time']
            sim_name = sim_data['dof_names'][sim_idx]
            
            # 弧度转度
            sim_values_deg = np.degrees(sim_values)
        else:
            sim_values_deg = np.array([])
            sim_time = np.array([])
            sim_name = "Not Found"
        
        # 重采样真实数据以匹配仿真数据长度（简化对比）
        if len(sim_time) > 0:
            real_time_resampled, real_values_resampled = resample_data(
                real_time, real_values, len(sim_time)
            )
        else:
            real_time_resampled = real_time
            real_values_resampled = real_values
        
        # 绘制
        ax.plot(real_time, real_values, 'b-', label=f'Real Data ({axes_names[i]}-axis)', 
                linewidth=1.5, alpha=0.7)
        
        if len(sim_time) > 0:
            ax.plot(sim_time, sim_values_deg, 'r-', label=f'Simulation Data ({sim_name})', 
                    linewidth=2, alpha=0.8)
        
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
    # 文件路径
    real_data_file = '03.txt'
    sim_data_file = 'kinematics_data/kinematics_20251208_175626_0000.txt'
    output_dir = Path('comparison_plots')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    real_data = load_real_data(real_data_file)
    sim_data = load_sim_data(sim_data_file)
    
    # 过滤仿真数据：只取0-5秒的数据（相对于开始时间）
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
    
    print(f"Real data shape: {real_data.shape}")
    print(f"Simulation data shape (filtered): {sim_data['positions'].shape}")
    print(f"Simulation time range: {sim_data['time'][0]:.2f}s - {sim_data['time'][-1]:.2f}s (filtered 0-5s from start)")
    print(f"Number of simulation DOFs: {len(sim_data['dof_names'])}")
    
    # 根据03.txt的列顺序（从第5行的ITEM行可知）:
    # 列索引（从0开始，因为跳过了第一列ITEM）:
    # Left_Hip: X(0), Y(1), Z(2)
    # Right_Hip: X(23), Y(24), Z(25)
    # Left_Knee: X(6), Y(7), Z(8)
    # Right_Knee: X(29), Y(30), Z(31)
    # Left_Ankle: X(12), Y(13), Z(14)
    # Right_Ankle: X(35), Y(36), Z(37)
    # Left_Pelvic: X(18), Y(19), Z(20)
    # Right_Pelvic: X(41), Y(42), Z(43)
    # Left_Foot_Pitch: X(21)
    # Right_Foot_Pitch: X(44)
    
    # 定义对比项 - 交叉对比：仿真左脚 vs 真实右脚，仿真右脚 vs 真实左脚
    comparisons = [
        {
            'name': 'Hip_SimLeft_vs_RealRight',
            'real_indices': [23, 24, 25],  # 真实右髋
            'sim_patterns': ['FemurL_x', 'FemurL_y', 'FemurL_z'],  # 仿真左髋
            'axes': ['X', 'Y', 'Z']
        },
        {
            'name': 'Hip_SimRight_vs_RealLeft',
            'real_indices': [0, 1, 2],  # 真实左髋
            'sim_patterns': ['FemurR_x', 'FemurR_y', 'FemurR_z'],  # 仿真右髋
            'axes': ['X', 'Y', 'Z']
        },
        {
            'name': 'Knee_SimLeft_vs_RealRight',
            'real_indices': [29, 30, 31],  # 真实右膝
            'sim_patterns': ['TibiaL', 'TibiaL', 'TibiaL'],  # 仿真左膝（单自由度）
            'axes': ['X', 'Y', 'Z']
        },
        {
            'name': 'Knee_SimRight_vs_RealLeft',
            'real_indices': [6, 7, 8],  # 真实左膝
            'sim_patterns': ['TibiaR', 'TibiaR', 'TibiaR'],  # 仿真右膝（单自由度）
            'axes': ['X', 'Y', 'Z']
        },
        {
            'name': 'Ankle_SimLeft_vs_RealRight',
            'real_indices': [35, 36, 37],  # 真实右踝
            'sim_patterns': ['TalusL_x', 'TalusL_y', 'TalusL_z'],  # 仿真左踝
            'axes': ['X', 'Y', 'Z']
        },
        {
            'name': 'Ankle_SimRight_vs_RealLeft',
            'real_indices': [12, 13, 14],  # 真实左踝
            'sim_patterns': ['TalusR_x', 'TalusR_y', 'TalusR_z'],  # 仿真右踝
            'axes': ['X', 'Y', 'Z']
        },
        {
            'name': 'Pelvic_SimLeft_vs_RealRight',
            'real_indices': [41, 42, 43],  # 真实右骨盆
            'sim_patterns': ['Pelvis_rot_x', 'Pelvis_rot_y', 'Pelvis_rot_z'],
            'axes': ['X', 'Y', 'Z']
        },
        {
            'name': 'Pelvic_SimRight_vs_RealLeft',
            'real_indices': [18, 19, 20],  # 真实左骨盆
            'sim_patterns': ['Pelvis_rot_x', 'Pelvis_rot_y', 'Pelvis_rot_z'],
            'axes': ['X', 'Y', 'Z']
        },
    ]
    
    print("\nGenerating comparison plots...")
    
    for comp in comparisons:
        print(f"\nProcessing: {comp['name']}")
        
        # 查找仿真数据中对应的DOF
        sim_indices = []
        for pattern in comp['sim_patterns']:
            found = find_dof_index(sim_data['dof_names'], pattern)
            if found:
                sim_indices.append(found[0][0])
                print(f"  Found simulation DOF: {found[0][1]}")
            else:
                sim_indices.append(None)
                print(f"  Simulation DOF not found: {pattern}")
        
        # 生成图像
        save_path = output_dir / f"{comp['name'].split('(')[0].strip().replace(' ', '_')}.png"
        plot_comparison(
            real_data, 
            sim_data, 
            comp['name'],
            comp['real_indices'],
            sim_indices,
            save_path,
            comp['axes']
        )
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nDone!")

if __name__ == '__main__':
    main()
