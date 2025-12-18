#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析单个仿真TXT数据文件并可视化
支持选择文件、查看关节角度、肌肉激活等
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import os

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
    print(f"\nLoaded: {filepath}")
    print(f"  Shape: {data.shape}")
    print(f"  DOFs: {num_dofs}")
    print(f"  Muscles: {num_muscles}")
    print(f"  Frames: {len(data)}")
    
    # 数据格式: frame_index time pos[0..n] vel[0..n] com_x com_y com_z com_vel_x com_vel_y com_vel_z muscle_act[0..m]
    result = {
        'frame': data[:, 0],
        'time': data[:, 1],
        'positions': data[:, 2:2+num_dofs],
        'velocities': data[:, 2+num_dofs:2+2*num_dofs],
        'com': data[:, 2+2*num_dofs:2+2*num_dofs+6],
        'muscle_activations': data[:, 2+2*num_dofs+6:] if num_muscles > 0 else None,
        'dof_names': dof_names,
        'muscle_names': muscle_names,
        'num_dofs': num_dofs,
        'num_muscles': num_muscles
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

def plot_joint_angles(data, joint_patterns, title_suffix=''):
    """绘制关节角度（XYZ三轴）"""
    fig, axes = plt.subplots(3, 2, figsize=(32, 12))
    fig.suptitle(f'Joint Angles Analysis (XYZ) {title_suffix}', fontsize=16, fontweight='bold')
    
    # 关节配置：(名称, 基础名称, 行, 列)
    joint_configs = [
        ('Left Hip', 'FemurL', 0, 0),
        ('Left Knee', 'TibiaL', 1, 0),
        ('Left Ankle', 'TalusL', 2, 0),
        ('Right Hip', 'FemurR', 0, 1),
        ('Right Knee', 'TibiaR', 1, 1),
        ('Right Ankle', 'TalusR', 2, 1),
    ]
    
    frames = data['frame']
    colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
    
    for joint_name, base_pattern, row, col in joint_configs:
        ax = axes[row, col]
        
        # 尝试查找 X, Y, Z 三个轴
        found_any = False
        for axis in ['x', 'y', 'z']:
            # 对于 TibiaL/TibiaR，只有一个自由度（没有_x后缀）
            if 'Tibia' in base_pattern:
                if axis == 'x':
                    idx = find_dof_index(data['dof_names'], base_pattern)
                else:
                    idx = None
            else:
                idx = find_dof_index(data['dof_names'], f'{base_pattern}_{axis}')
            
            if idx is not None:
                angles = np.degrees(data['positions'][:, idx])
                ax.plot(frames, angles, linewidth=1.5, color=colors[axis], 
                       label=f'{axis.upper()}-axis', alpha=0.8)
                found_any = True
        
        if found_any:
            ax.set_xlabel('Frame', fontsize=10)
            ax.set_ylabel('Angle (degrees)', fontsize=10)
            ax.set_title(f'{joint_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{joint_name}\nNot Found', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{joint_name}', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_muscle_activations(data, muscle_groups, title_suffix=''):
    """绘制肌肉激活"""
    # 定义肌肉组
    muscle_group_patterns = {
        'Hip Flexors': ['psoas', 'iliacus', 'rectus_fem'],
        'Hip Extensors': ['glut_max', 'hamstrings', 'bifem'],
        'Knee Extensors': ['vas_', 'rectus_fem'],
        'Knee Flexors': ['bifem', 'semimem', 'semiten'],
        'Ankle Plantarflexors': ['gastroc', 'soleus'],
        'Ankle Dorsiflexors': ['tib_ant'],
    }
    
    n_groups = len(muscle_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(16, 4*n_groups))
    if n_groups == 1:
        axes = [axes]
    
    fig.suptitle(f'Muscle Activation Analysis {title_suffix}', fontsize=16, fontweight='bold')
    
    frames = data['frame']
    
    for i, group_name in enumerate(muscle_groups):
        ax = axes[i]
        patterns = muscle_group_patterns.get(group_name, [])
        
        found_muscles = []
        for pattern in patterns:
            for j, muscle_name in enumerate(data['muscle_names']):
                if pattern.lower() in muscle_name.lower():
                    found_muscles.append((muscle_name, j))
        
        if found_muscles:
            for muscle_name, idx in found_muscles:
                activations = data['muscle_activations'][:, idx]
                ax.plot(frames, activations, label=muscle_name, linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel('Frame', fontsize=10)
            ax.set_ylabel('Activation', fontsize=10)
            ax.set_title(f'{group_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, f'{group_name}\nNo muscles found', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{group_name}', fontsize=12)
    
    plt.tight_layout()
    return fig

def print_data_summary(data):
    """打印数据摘要"""
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    print(f"\nBasic Info:")
    print(f"  Total frames: {len(data['frame'])}")
    print(f"  Time range: {data['time'][0]:.3f}s - {data['time'][-1]:.3f}s")
    print(f"  Duration: {data['time'][-1] - data['time'][0]:.3f}s")
    
    print(f"\nDOFs ({data['num_dofs']} total):")
    for i, name in enumerate(data['dof_names'][:10]):  # 只显示前10个
        print(f"  [{i}] {name}")
    if len(data['dof_names']) > 10:
        print(f"  ... and {len(data['dof_names']) - 10} more")
    
    if data['num_muscles'] > 0:
        print(f"\nMuscles ({data['num_muscles']} total):")
        for i, name in enumerate(data['muscle_names'][:10]):  # 只显示前10个
            print(f"  [{i}] {name}")
        if len(data['muscle_names']) > 10:
            print(f"  ... and {len(data['muscle_names']) - 10} more")
    
    print("\nCenter of Mass:")
    print(f"  X range: [{data['com'][:, 0].min():.3f}, {data['com'][:, 0].max():.3f}]")
    print(f"  Y range: [{data['com'][:, 1].min():.3f}, {data['com'][:, 1].max():.3f}]")
    print(f"  Z range: [{data['com'][:, 2].min():.3f}, {data['com'][:, 2].max():.3f}]")
    
    print("="*70 + "\n")

def select_txt_file():
    """选择TXT文件"""
    # 查找所有txt文件
    txt_files = []
    
    # 在当前目录和常见目录查找
    search_dirs = [
        '.',
        'kinematics_data',
        'data',
        'output',
    ]
    
    for dir_path in search_dirs:
        if os.path.exists(dir_path):
            for file in Path(dir_path).glob('*.txt'):
                txt_files.append(file)
    
    if not txt_files:
        print("Error: No TXT files found!")
        return None
    
    # 显示文件列表
    print("\n" + "="*70)
    print("Available TXT files:")
    print("="*70)
    for i, file in enumerate(txt_files, 1):
        print(f"  [{i}] {file}")
    print("="*70)
    
    # 让用户选择
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(txt_files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(txt_files):
                return txt_files[idx]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(txt_files)}")
        except (ValueError, KeyboardInterrupt):
            print("\nOperation cancelled.")
            return None

def main():
    """主函数"""
    print("\n" + "="*70)
    print("Single TXT Data Analysis Tool")
    print("="*70)
    
    # 选择文件
    txt_file = select_txt_file()
    if txt_file is None:
        return
    
    # 加载数据
    print(f"\nLoading file: {txt_file}")
    data = load_sim_data(str(txt_file))
    
    # 打印摘要
    print_data_summary(data)
    
    # 生成输出文件名
    output_dir = Path('analysis_output')
    output_dir.mkdir(exist_ok=True)
    base_name = txt_file.stem
    
    # 绘制关节角度
    print("Generating joint angle plots...")
    fig1 = plot_joint_angles(data, [], title_suffix=f'({base_name})')
    output_file1 = output_dir / f'{base_name}_joint_angles.png'
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file1}")
    plt.close(fig1)
    
    # 绘制肌肉激活
    if data['num_muscles'] > 0:
        print("Generating muscle activation plots...")
        muscle_groups = ['Hip Flexors', 'Hip Extensors', 'Knee Extensors', 
                        'Knee Flexors', 'Ankle Plantarflexors', 'Ankle Dorsiflexors']
        fig2 = plot_muscle_activations(data, muscle_groups, title_suffix=f'({base_name})')
        output_file2 = output_dir / f'{base_name}_muscle_activations.png'
        fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file2}")
        plt.close(fig2)
    else:
        print("No muscle data available.")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
