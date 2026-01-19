#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取右侧目标肌肉激活数据并可视化
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

# 目标肌肉定义（右侧）
TARGET_MUSCLES = {
    'Gastrocnemius Medialis': ['R_Gastrocnemius_Medial_Head', 'gastroc_r', 'gas_med_r'],
    'Tibialis Anterior': ['R_Tibialis_Anterior', 'tib_ant_r', 'tibialis_ant_r'],
    'Soleus': ['R_Soleus', 'soleus_r', 'sol_r'],
    'Vastus Medialis': ['R_Vastus_Medialis', 'vas_med_r', 'vmed_r'],
    'Vastus Lateralis': ['R_Vastus_Lateralis', 'vas_lat_r', 'vlat_r'],
    'Rectus Femoris': ['R_Rectus_Femoris', 'rectus_fem_r', 'rect_fem_r'],
    'Biceps Femoris': ['R_Bicep_Femoris', 'bifem_r', 'biceps_fem_r'],
    'Semitendinosus': ['R_Semitendinosus', 'semiten_r', 'semi_r'],
    'Gracilis': ['R_Gracilis', 'gracilis_r', 'grac_r'],
    'Gluteus Medius': ['R_Gluteus_Medius', 'glut_med_r', 'gluteus_med_r'],
    'External Oblique': ['R_Internal_Oblique', 'obl_ext_r', 'external_oblique_r']
}

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
    
    # 数据格式: frame_index time pos[0..n] vel[0..n] com_x com_y com_z com_vel_x com_vel_y com_vel_z muscle_act[0..m] phase
    muscle_end_col = 2 + 2*num_dofs + 6 + num_muscles
    
    # 检查是否有phase列
    has_phase = data.shape[1] > muscle_end_col
    phase_data = data[:, -1] if has_phase else None
    
    result = {
        'frame': data[:, 0],
        'time': data[:, 1],
        'positions': data[:, 2:2+num_dofs],
        'velocities': data[:, 2+num_dofs:2+2*num_dofs],
        'com': data[:, 2+2*num_dofs:2+2*num_dofs+6],
        'muscle_activations': data[:, 2+2*num_dofs+6:muscle_end_col] if num_muscles > 0 else None,
        'phase': phase_data,
        'dof_names': dof_names,
        'muscle_names': muscle_names,
        'num_dofs': num_dofs,
        'num_muscles': num_muscles
    }
    
    if has_phase:
        print(f"  Phase data: Found (range: {np.min(phase_data):.3f} - {np.max(phase_data):.3f})")
    else:
        print(f"  Phase data: Not found")
    
    return result

def find_target_muscles(muscle_names):
    """查找目标肌肉在数据中的索引"""
    found_muscles = {}
    not_found = []
    
    for target_name, patterns in TARGET_MUSCLES.items():
        found = False
        for pattern in patterns:
            for idx, muscle_name in enumerate(muscle_names):
                if pattern.lower() in muscle_name.lower():
                    found_muscles[target_name] = {
                        'index': idx,
                        'full_name': muscle_name
                    }
                    found = True
                    break
            if found:
                break
        
        if not found:
            not_found.append(target_name)
    
    return found_muscles, not_found

def plot_target_muscles(data, found_muscles, title_suffix=''):
    """绘制目标肌肉激活图像"""
    n_muscles = len(found_muscles)
    
    if n_muscles == 0:
        print("Warning: No target muscles found in data!")
        return None
    
    # 只使用前100帧
    max_frames = min(100, len(data['frame']))
    
    # 计算子图布局
    n_cols = 2
    n_rows = (n_muscles + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    fig.suptitle(f'Target Right Side Muscles Activation (First 100 Frames) {title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 如果只有一行，确保axes是2D数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    frames = data['frame'][:max_frames]
    
    # 按照目标肌肉列表的顺序绘制
    muscle_list = list(TARGET_MUSCLES.keys())
    plot_idx = 0
    
    for muscle_name in muscle_list:
        if muscle_name not in found_muscles:
            continue
            
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        muscle_info = found_muscles[muscle_name]
        idx = muscle_info['index']
        full_name = muscle_info['full_name']
        
        activations = data['muscle_activations'][:max_frames, idx]
        
        # 如果有phase数据,在背景显示相位
        if data['phase'] is not None:
            phase_values = data['phase'][:max_frames]
            # 使用phase作为次坐标轴
            ax2 = ax.twinx()
            ax2.plot(frames, phase_values, linewidth=1, color='red', 
                    alpha=0.4, linestyle='--', label='Gait Phase')
            ax2.set_ylabel('Gait Phase', fontsize=9, color='red')
            ax2.set_ylim([0, max(1.0, np.max(phase_values))])
            ax2.tick_params(axis='y', labelcolor='red', labelsize=8)
            ax2.spines['right'].set_color('red')
            ax2.spines['right'].set_linewidth(1.5)
            
            # 添加相位的背景色块 (假设0-0.6为支撑相, 0.6-1为摆动相)
            for i in range(len(frames)-1):
                if phase_values[i] < 0.6:
                    ax.axvspan(frames[i], frames[i+1], alpha=0.05, color='yellow')
                else:
                    ax.axvspan(frames[i], frames[i+1], alpha=0.05, color='cyan')
        
        # 绘制激活曲线
        ax.plot(frames, activations, linewidth=2, color='royalblue', alpha=0.8, label='Activation')
        ax.fill_between(frames, 0, activations, alpha=0.3, color='lightblue')
        
        # 计算统计信息
        mean_act = np.mean(activations)
        max_act = np.max(activations)
        
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Activation', fontsize=10)
        ax.set_title(f'{muscle_name}\n({full_name})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_ylim([0, 1])
        
        # 如果有phase数据,添加图例说明
        if data['phase'] is not None:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.7)
        
        # 添加统计信息
        textstr = f'Mean: {mean_act:.3f}\nMax: {max_act:.3f}'
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
        
        plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def plot_muscle_comparison(data, found_muscles, title_suffix=''):
    """在同一图中比较所有目标肌肉"""
    if len(found_muscles) == 0:
        return None
    
    # 只使用前100帧
    max_frames = min(100, len(data['frame']))
    
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(f'Target Right Side Muscles - Comparison (First 100 Frames) {title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    frames = data['frame'][:max_frames]
    colors = plt.cm.tab20(np.linspace(0, 1, len(found_muscles)))
    
    # 如果有phase数据,在背景显示相位
    if data['phase'] is not None:
        phase_values = data['phase'][:max_frames]
        ax2 = ax.twinx()
        ax2.plot(frames, phase_values, linewidth=2, color='red', 
                alpha=0.5, linestyle='--', label='Gait Phase', zorder=1)
        ax2.set_ylabel('Gait Phase', fontsize=12, color='red')
        ax2.set_ylim([0, max(1.0, np.max(phase_values))])
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.spines['right'].set_color('red')
        ax2.spines['right'].set_linewidth(2)
        
        # 添加相位背景色
        for i in range(len(frames)-1):
            if phase_values[i] < 0.6:
                ax.axvspan(frames[i], frames[i+1], alpha=0.05, color='yellow', zorder=0)
            else:
                ax.axvspan(frames[i], frames[i+1], alpha=0.05, color='cyan', zorder=0)
    
    muscle_list = list(TARGET_MUSCLES.keys())
    plot_idx = 0
    
    for muscle_name in muscle_list:
        if muscle_name not in found_muscles:
            continue
            
        muscle_info = found_muscles[muscle_name]
        idx = muscle_info['index']
        
        activations = data['muscle_activations'][:max_frames, idx]
        ax.plot(frames, activations, linewidth=1.5, 
               label=muscle_name, color=colors[plot_idx], alpha=0.7, zorder=2)
        plot_idx += 1
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Activation', fontsize=12)
    ax.set_title('All Target Muscles', fontsize=14, fontweight='bold')
    
    # 调整图例位置,为phase数据腾出空间
    if data['phase'] is not None:
        ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.8)
    else:
        ax.legend(loc='upper right', fontsize=9, ncol=2)
    
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_ylim([0, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def export_muscle_data(data, found_muscles, output_file):
    """导出肌肉激活数据到CSV文件"""
    if len(found_muscles) == 0:
        print("No muscles to export!")
        return
    
    # 准备数据
    export_data = []
    headers = ['frame', 'time']
    
    muscle_list = list(TARGET_MUSCLES.keys())
    for muscle_name in muscle_list:
        if muscle_name in found_muscles:
            headers.append(muscle_name)
    
    # 组织数据
    for i in range(len(data['frame'])):
        row = [data['frame'][i], data['time'][i]]
        for muscle_name in muscle_list:
            if muscle_name in found_muscles:
                idx = found_muscles[muscle_name]['index']
                row.append(data['muscle_activations'][i, idx])
        export_data.append(row)
    
    # 写入CSV
    with open(output_file, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in export_data:
            f.write(','.join([str(v) for v in row]) + '\n')
    
    print(f"  Exported muscle data: {output_file}")

def print_muscle_summary(found_muscles, not_found):
    """打印肌肉查找摘要"""
    print("\n" + "="*70)
    print("TARGET MUSCLES SUMMARY")
    print("="*70)
    
    print(f"\nFound Muscles ({len(found_muscles)}/{len(TARGET_MUSCLES)}):")
    for muscle_name, info in found_muscles.items():
        print(f"  ✓ {muscle_name}: {info['full_name']} [index: {info['index']}]")
    
    if not_found:
        print(f"\nNot Found Muscles ({len(not_found)}):")
        for muscle_name in not_found:
            print(f"  ✗ {muscle_name}")
            print(f"      Searched patterns: {', '.join(TARGET_MUSCLES[muscle_name])}")
    
    print("="*70 + "\n")

def select_txt_file():
    """选择TXT文件"""
    txt_files = []
    
    search_dirs = [
        '.',
        'kinematics_data',
        'data',
        'output',
        'patient_data',
    ]
    
    for dir_path in search_dirs:
        if os.path.exists(dir_path):
            for file in Path(dir_path).glob('*.txt'):
                txt_files.append(file)
    
    if not txt_files:
        print("Error: No TXT files found!")
        return None
    
    print("\n" + "="*70)
    print("Available TXT files:")
    print("="*70)
    for i, file in enumerate(txt_files, 1):
        print(f"  [{i}] {file}")
    print("="*70)
    
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
    print("Target Right Side Muscles Extraction Tool")
    print("="*70)
    print("\nTarget Muscles:")
    for i, muscle_name in enumerate(TARGET_MUSCLES.keys(), 1):
        print(f"  {i:2d}. {muscle_name}")
    print("="*70)
    
    # 选择文件
    txt_file = select_txt_file()
    if txt_file is None:
        return
    
    # 加载数据
    print(f"\nLoading file: {txt_file}")
    data = load_sim_data(str(txt_file))
    
    if data['num_muscles'] == 0:
        print("\nError: No muscle data found in file!")
        return
    
    # 查找目标肌肉
    print("\nSearching for target muscles...")
    found_muscles, not_found = find_target_muscles(data['muscle_names'])
    print_muscle_summary(found_muscles, not_found)
    
    if len(found_muscles) == 0:
        print("Error: None of the target muscles were found!")
        return
    
    # 生成输出文件名
    output_dir = Path('target_muscle_output')
    output_dir.mkdir(exist_ok=True)
    base_name = txt_file.stem
    
    # 绘制独立肌肉图
    print("Generating individual muscle plots...")
    fig1 = plot_target_muscles(data, found_muscles, title_suffix=f'({base_name})')
    if fig1:
        output_file1 = output_dir / f'{base_name}_target_muscles_individual.png'
        fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file1}")
        plt.close(fig1)
    
    # 绘制对比图
    print("Generating comparison plot...")
    fig2 = plot_muscle_comparison(data, found_muscles, title_suffix=f'({base_name})')
    if fig2:
        output_file2 = output_dir / f'{base_name}_target_muscles_comparison.png'
        fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file2}")
        plt.close(fig2)
    
    # 导出CSV数据
    print("Exporting muscle data to CSV...")
    csv_file = output_dir / f'{base_name}_target_muscles_data.csv'
    export_muscle_data(data, found_muscles, csv_file)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"  - Individual muscle plots")
    print(f"  - Comparison plot")
    print(f"  - CSV data export")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
