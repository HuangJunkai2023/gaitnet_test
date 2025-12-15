#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比三组仿真数据的左侧髋、膝、踝关节角度
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 配置matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

def load_sim_data(filepath):
    """加载仿真数据"""
    data = []
    dof_names = []
    muscle_names = []
    num_dofs = 0
    num_muscles = 0
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 解析DOF名称和肌肉名称
    for line in lines:
        if '# DOF names:' in line:
            names_part = line.split('# DOF names:')[1].strip()
            dof_names = [name.strip() for name in names_part.split(',')]
        elif '# Muscle names:' in line:
            names_part = line.split('# Muscle names:')[1].strip()
            muscle_names = [name.strip() for name in names_part.split(',')]
        elif '# Number of DOFs:' in line:
            num_dofs = int(line.split(':')[1].strip())
        elif '# Number of Muscles:' in line:
            num_muscles = int(line.split(':')[1].strip())
    
    # 读取数据
    for line in lines:
        if not line.startswith('#') and line.strip():
            values = line.strip().split()
            if len(values) > 2:
                data.append([float(v) for v in values])
    
    data = np.array(data)
    print(f"Loaded {filepath}")
    print(f"  Shape: {data.shape}, DOFs: {num_dofs}, Muscles: {num_muscles}")
    
    # 数据格式: frame_index time pos[0..n] vel[0..n] com_x com_y com_z com_vel_x com_vel_y com_vel_z muscle_act[0..m]
    result = {
        'frame': data[:, 0],
        'time': data[:, 1],
        'positions': data[:, 2:2+num_dofs],
        'velocities': data[:, 2+num_dofs:2+2*num_dofs],
        'com': data[:, 2+2*num_dofs:2+2*num_dofs+6],
        'muscle_activations': data[:, 2+2*num_dofs+6:] if num_muscles > 0 else None,
        'dof_names': dof_names,
        'muscle_names': muscle_names
    }
    
    return result

def find_dof_index(dof_names, pattern):
    """查找包含特定模式的DOF索引"""
    for i, name in enumerate(dof_names):
        if pattern.lower() in name.lower():
            return i
    return None

def find_muscle_index(muscle_names, pattern):
    """查找包含特定模式的肌肉索引"""
    for i, name in enumerate(muscle_names):
        if pattern.lower() in name.lower():
            return i
    return None

def plot_muscle_activation_comparison(datasets, labels, muscle_name, muscle_patterns, axes_names, save_path):
    """绘制三组数据的肌肉激活对比图，并显示平均激活强度统计"""
    n_axes = len(muscle_patterns)
    colors = ['blue', 'red', 'green']
    
    fig, axs = plt.subplots(n_axes, 1, figsize=(14, 4*n_axes))
    if n_axes == 1:
        axs = [axs]
    
    fig.suptitle(f'{muscle_name} Muscle Activation Comparison (3 Datasets)', fontsize=16, fontweight='bold')
    
    # 用于存储统计信息
    stats_text = []
    
    for i, (pattern, axis_name) in enumerate(zip(muscle_patterns, axes_names)):
        ax = axs[i]
        
        # 存储每个数据集的平均值
        mean_values = []
        
        for j, (data, label, color) in enumerate(zip(datasets, labels, colors)):
            if data['muscle_activations'] is None:
                print(f"  Warning: No muscle data in {label}")
                continue
                
            muscle_idx = find_muscle_index(data['muscle_names'], pattern)
            
            if muscle_idx is not None:
                values = data['muscle_activations'][:, muscle_idx]
                time = data['time']
                muscle_full_name = data['muscle_names'][muscle_idx]
                
                # 计算统计信息
                mean_act = np.mean(values)
                std_act = np.std(values)
                max_act = np.max(values)
                mean_values.append(mean_act)
                
                ax.plot(time, values, color=color, label=f'{label} ({muscle_full_name})', 
                       linewidth=2.5, alpha=0.8)
            else:
                print(f"  Warning: {pattern} not found in {label}")
                mean_values.append(None)
        
        # 生成统计信息文本
        if len(mean_values) >= 3 and all(v is not None for v in mean_values):
            stats = f"{axis_name} 平均激活:\n"
            stats += f"  {labels[0]}: {mean_values[0]:.4f}\n"
            stats += f"  {labels[1]}: {mean_values[1]:.4f} ({(mean_values[1]/mean_values[0]-1)*100:+.1f}%)\n"
            stats += f"  {labels[2]}: {mean_values[2]:.4f} ({(mean_values[2]/mean_values[0]-1)*100:+.1f}%)"
            stats_text.append(stats)
            
            # 在图上添加文本框
            ax.text(0.98, 0.97, stats, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Activation (0-1)', fontsize=12)
        ax.set_title(f'{muscle_name} - {axis_name}', fontsize=13)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    # 打印统计信息到控制台
    if stats_text:
        print(f"  Statistics:")
        for stat in stats_text:
            for line in stat.split('\n'):
                print(f"    {line}")
    
    plt.close()

def plot_three_way_comparison(datasets, labels, joint_name, dof_patterns, axes_names, save_path):
    """绘制三组数据的关节角度对比图"""
    n_axes = len(dof_patterns)
    colors = ['blue', 'red', 'green']
    
    fig, axs = plt.subplots(n_axes, 1, figsize=(14, 4*n_axes))
    if n_axes == 1:
        axs = [axs]
    
    fig.suptitle(f'{joint_name} Angle Comparison (3 Datasets)', fontsize=16, fontweight='bold')
    
    for i, (pattern, axis_name) in enumerate(zip(dof_patterns, axes_names)):
        ax = axs[i]
        
        for j, (data, label, color) in enumerate(zip(datasets, labels, colors)):
            dof_idx = find_dof_index(data['dof_names'], pattern)
            
            if dof_idx is not None:
                values = data['positions'][:, dof_idx]
                time = data['time']
                dof_name = data['dof_names'][dof_idx]
                values_deg = np.degrees(values)
                
                ax.plot(time, values_deg, color=color, label=f'{label} ({dof_name})', 
                       linewidth=2.5, alpha=0.8)
            else:
                print(f"  Warning: {pattern} not found in {label}")
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.set_title(f'{joint_name} - {axis_name}', fontsize=13)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 4])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    # 三组数据文件
    files = [
        'kinematics_data/off.txt',
        'kinematics_data/20n.txt',
        'kinematics_data/100n.txt',
    ]
    
    labels = [
        'No Exoskeleton',
        'Exo 20N', 
        'Exo 100N'
    ]
    
    output_dir = Path('comparison_plots_three_datasets')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading simulation data...")
    datasets = []
    time_shifts = [-0.0, -0.35, -0.25]  # 时间偏移: off左移0.1s, 20n左移0.75s, 100n左移0.1s
    
    for file, shift in zip(files, time_shifts):
        data = load_sim_data(file)
        # 归一化时间从0开始，然后应用时间偏移
        data['time'] = data['time'] - data['time'][0] + shift
        
        # 过滤掉时间小于0或大于4s的数据点
        valid_mask = (data['time'] >= 0) & (data['time'] <= 4.0)
        data['time'] = data['time'][valid_mask]
        data['positions'] = data['positions'][valid_mask]
        data['velocities'] = data['velocities'][valid_mask]
        data['com'] = data['com'][valid_mask]
        data['frame'] = data['frame'][valid_mask]
        if data['muscle_activations'] is not None:
            data['muscle_activations'] = data['muscle_activations'][valid_mask]
        
        datasets.append(data)
        print(f"  Applied time shift: {shift:+.2f}s, Valid points: {len(data['time'])}")
    
    print("\nGenerating comparison plots...")
    
    # 定义要对比的关节
    comparisons = [
        {
            'name': 'Left_Hip',
            'patterns': ['FemurL_x', 'FemurL_y', 'FemurL_z'],
            'axes': ['X (Flexion/Extension)', 'Y (Abduction/Adduction)', 'Z (Internal/External Rotation)']
        },
        {
            'name': 'Left_Knee',
            'patterns': ['TibiaL'],
            'axes': ['Flexion/Extension']
        },
        {
            'name': 'Left_Ankle',
            'patterns': ['TalusL_x', 'TalusL_y', 'TalusL_z'],
            'axes': ['X (Dorsiflexion/Plantarflexion)', 'Y (Inversion/Eversion)', 'Z (Rotation)']
        },
    ]
    
    for comp in comparisons:
        print(f"\nProcessing: {comp['name']}")
        save_path = output_dir / f"{comp['name']}.png"
        plot_three_way_comparison(
            datasets,
            labels,
            comp['name'],
            comp['patterns'],
            comp['axes'],
            save_path
        )
    
    # 定义要对比的肌肉激活
    muscle_comparisons = [
        {
            'name': 'Left_Tibialis_Anterior',
            'patterns': ['L_Tibialis_Anterior'],
            'axes': ['Activation']
        },
        {
            'name': 'Left_Gastrocnemius',
            'patterns': ['L_Gastrocnemius_Lateral_Head', 'L_Gastrocnemius_Medial_Head'],
            'axes': ['Lateral Head', 'Medial Head']
        },
        {
            'name': 'Left_Soleus',
            'patterns': ['L_Soleus', 'L_Soleus1'],
            'axes': ['Soleus', 'Soleus1']
        }
    ]
    
    for muscle_comp in muscle_comparisons:
        print(f"\nProcessing muscle: {muscle_comp['name']}")
        save_path = output_dir / f"{muscle_comp['name']}_Activation.png"
        plot_muscle_activation_comparison(
            datasets,
            labels,
            muscle_comp['name'],
            muscle_comp['patterns'],
            muscle_comp['axes'],
            save_path
        )
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\n=== Summary ===")
    print(f"Compared {len(datasets)} datasets:")
    for i, (file, label) in enumerate(zip(files, labels)):
        print(f"  {label}: {file}")
    print(f"\nGenerated {len(comparisons)} joint comparison plots")
    print(f"Generated {len(muscle_comparisons)} muscle activation plots")
    print("Done!")

if __name__ == '__main__':
    main()
