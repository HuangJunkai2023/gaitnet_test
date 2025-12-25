#!/usr/bin/env python3
"""
Visualize muscle EMG activation patterns from NPZ files.

Generates comprehensive EMG visualizations including:
1. Individual plots for each muscle showing all gait cycles
2. Heatmap showing mean signal across all muscles
3. Statistical summary plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from pathlib import Path
import os


# Set up matplotlib for better output
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = '#f5f5f5'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9


def plot_all_cycles_comprehensive(emg_data, muscle_names, output_dir, file_prefix=''):
    """
    为每个肌肉绘制所有步态周期的曲线。
    
    Parameters:
    -----------
    emg_data : array, shape (n_cycles, n_frames, n_muscles)
        肌肉激活数据
    muscle_names : array of str
        肌肉名称列表
    output_dir : str
        输出目录
    file_prefix : str
        输出文件名前缀
    
    Returns:
    --------
    fig : matplotlib figure
    """
    n_cycles, n_frames, n_muscles = emg_data.shape
    time_axis = np.arange(n_frames)
    
    # 创建12个子图 (4x3) 或根据肌肉数调整
    n_cols = 3
    n_rows = (n_muscles + n_cols - 1) // n_cols
    
    figsize = (15, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'All Gait Cycles - EMG Activation Patterns (N={n_cycles} cycles)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    # 颜色映射：根据周期索引从浅到深
    colors = plt.cm.viridis(np.linspace(0, 1, n_cycles))
    
    for muscle_idx in range(n_muscles):
        ax = axes[muscle_idx]
        
        # 获取该肌肉的所有周期数据
        muscle_data = emg_data[:, :, muscle_idx]  # shape: (n_cycles, n_frames)
        
        # 绘制所有周期 (使用不同颜色，半透明)
        for cycle_idx in range(n_cycles):
            ax.plot(time_axis, muscle_data[cycle_idx], 
                   color=colors[cycle_idx], alpha=0.5, linewidth=0.8)
        
        # 绘制平均值 (黑色粗线)
        mean_signal = np.mean(muscle_data, axis=0)
        std_signal = np.std(muscle_data, axis=0)
        
        ax.plot(time_axis, mean_signal, color='black', linewidth=2.5, 
               label='Mean', zorder=100)
        
        # 绘制±1 std的阴影区域
        ax.fill_between(time_axis, 
                        mean_signal - std_signal, 
                        mean_signal + std_signal,
                        color='black', alpha=0.1, label='±1 Std')
        
        # 美化
        muscle_name = muscle_names[muscle_idx] if isinstance(muscle_names[muscle_idx], str) else str(muscle_names[muscle_idx])
        # 简化肌肉名称（移除前缀）
        if muscle_name.startswith('R_'):
            muscle_name = muscle_name[2:]
        
        ax.set_title(f'{muscle_name}\n(N={n_cycles})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Normalized Gait Cycle (%)', fontsize=10)
        ax.set_ylabel('Activation', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(0, n_frames - 1)
        ax.set_ylim(-0.05, max(mean_signal + std_signal) * 1.1)
        
        if muscle_idx == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # 设置刻度标签
        tick_positions = np.linspace(0, n_frames-1, 5)
        tick_labels = [f'{int(p/n_frames*100)}%' for p in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    
    # 隐藏多余的子图
    for idx in range(n_muscles, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_summary_heatmap(emg_data, muscle_names, output_dir, file_prefix=''):
    """
    生成热力图：显示所有肌肉的平均肌电信号。
    
    X轴: 时间步（归一化步态周期）
    Y轴: 肌肉
    颜色: 激活强度
    
    Parameters:
    -----------
    emg_data : array, shape (n_cycles, n_frames, n_muscles)
    muscle_names : array of str
    output_dir : str
    file_prefix : str
    
    Returns:
    --------
    fig : matplotlib figure
    """
    n_cycles, n_frames, n_muscles = emg_data.shape
    
    # 计算平均信号 (对所有周期平均)
    mean_data = np.mean(emg_data, axis=0)  # shape: (n_frames, n_muscles)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制热力图
    im = ax.imshow(mean_data.T, aspect='auto', cmap='YlOrRd', 
                   extent=[0, 100, 0, n_muscles-1], 
                   origin='lower', interpolation='bilinear')
    
    ax.set_xlabel('Normalized Gait Cycle (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Muscle', fontsize=12, fontweight='bold')
    ax.set_title(f'Mean EMG Signal Across All Gait Cycles (N={n_cycles} cycles)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Y轴标签 (肌肉名称)
    muscle_labels = []
    for name in muscle_names:
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        if name.startswith('R_'):
            name = name[2:]
        muscle_labels.append(name)
    
    ax.set_yticks(np.arange(n_muscles))
    ax.set_yticklabels(muscle_labels, fontsize=10)
    
    # X轴标签
    ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_xticklabels([f'{int(i)}%' for i in np.linspace(0, 100, 11)], fontsize=10)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, label='Mean Activation', pad=0.02)
    cbar.set_label('Mean Activation', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_activation_statistics(emg_data, muscle_names, output_dir, file_prefix=''):
    """
    绘制肌肉激活统计信息。
    
    显示每个肌肉的：
    - 平均激活水平
    - 激活范围
    - 峰值激活
    
    Parameters:
    -----------
    emg_data : array, shape (n_cycles, n_frames, n_muscles)
    muscle_names : array of str
    output_dir : str
    file_prefix : str
    
    Returns:
    --------
    fig : matplotlib figure
    """
    n_cycles, n_frames, n_muscles = emg_data.shape
    
    # 计算统计信息
    mean_activation = np.mean(emg_data, axis=(0, 1))  # (n_muscles,)
    max_activation = np.max(emg_data, axis=(0, 1))    # (n_muscles,)
    min_activation = np.min(emg_data, axis=(0, 1))    # (n_muscles,)
    std_activation = np.std(emg_data, axis=(0, 1))    # (n_muscles,)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'EMG Activation Statistics (N={n_cycles} cycles)', 
                fontsize=14, fontweight='bold')
    
    muscle_labels = []
    for name in muscle_names:
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        if name.startswith('R_'):
            name = name[2:]
        muscle_labels.append(name)
    
    x_pos = np.arange(n_muscles)
    
    # 1. 平均激活水平
    ax = axes[0, 0]
    bars = ax.bar(x_pos, mean_activation, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)
    ax.set_ylabel('Mean Activation', fontsize=11, fontweight='bold')
    ax.set_title('Mean Activation Level', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(muscle_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(mean_activation) * 1.15)
    
    # 2. 激活范围（最大-最小）
    ax = axes[0, 1]
    activation_range = max_activation - min_activation
    bars = ax.bar(x_pos, activation_range, color='coral', alpha=0.7, edgecolor='darkred', linewidth=1.5)
    ax.set_ylabel('Activation Range', fontsize=11, fontweight='bold')
    ax.set_title('Peak-to-Baseline Activation Range', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(muscle_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(activation_range) * 1.15)
    
    # 3. 标准差 (变异性)
    ax = axes[1, 0]
    bars = ax.bar(x_pos, std_activation, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
    ax.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    ax.set_title('Activation Variability (Std Dev)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(muscle_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(std_activation) * 1.15)
    
    # 4. 峰值激活
    ax = axes[1, 1]
    bars = ax.bar(x_pos, max_activation, color='mediumpurple', alpha=0.7, edgecolor='purple', linewidth=1.5)
    ax.set_ylabel('Peak Activation', fontsize=11, fontweight='bold')
    ax.set_title('Peak Activation Level', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(muscle_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max_activation) * 1.15)
    
    plt.tight_layout()
    return fig


def plot_individual_muscles(emg_data, muscle_names, output_dir, file_prefix=''):
    """
    为每个肌肉创建单独的高质量图。
    
    Parameters:
    -----------
    emg_data : array, shape (n_cycles, n_frames, n_muscles)
    muscle_names : array of str
    output_dir : str
    file_prefix : str
    """
    n_cycles, n_frames, n_muscles = emg_data.shape
    time_axis = np.arange(n_frames)
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, n_cycles))
    
    for muscle_idx in range(n_muscles):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        muscle_data = emg_data[:, :, muscle_idx]
        
        # 绘制所有周期
        for cycle_idx in range(n_cycles):
            ax.plot(time_axis, muscle_data[cycle_idx], 
                   color=colors[cycle_idx], alpha=0.4, linewidth=1)
        
        # 平均和标准差
        mean_signal = np.mean(muscle_data, axis=0)
        std_signal = np.std(muscle_data, axis=0)
        
        ax.plot(time_axis, mean_signal, color='black', linewidth=2.5, label='Mean')
        ax.fill_between(time_axis, 
                        mean_signal - std_signal,
                        mean_signal + std_signal,
                        color='black', alpha=0.15, label='±1 Std')
        
        # 标签
        muscle_name = muscle_names[muscle_idx]
        if isinstance(muscle_name, bytes):
            muscle_name = muscle_name.decode('utf-8')
        if muscle_name.startswith('R_'):
            muscle_name = muscle_name[2:]
        
        ax.set_title(f'{muscle_name}\nAll {n_cycles} Gait Cycles', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Normalized Gait Cycle (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Activation', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
        
        # 设置X轴为百分比
        tick_positions = np.linspace(0, n_frames-1, 5)
        tick_labels = [f'{int(p/n_frames*100)}%' for p in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        ax.set_ylim(-0.05, max(mean_signal + std_signal) * 1.1)
        
        plt.tight_layout()
        
        # 保存
        safe_name = muscle_name.replace('/', '_').replace(' ', '_')
        output_file = os.path.join(output_dir, f'{file_prefix}muscle_{safe_name}.png')
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ {output_file}")
        
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize EMG data from NPZ files')
    parser.add_argument('input_file', help='Input NPZ file')
    parser.add_argument('--output-dir', default='emg_plots',
                       help='Output directory for plots (default: emg_plots)')
    parser.add_argument('--prefix', default='',
                       help='Prefix for output filenames')
    parser.add_argument('--individual', action='store_true',
                       help='Create individual plots for each muscle')
    parser.add_argument('--no-heatmap', action='store_true',
                       help='Skip heatmap generation')
    parser.add_argument('--no-stats', action='store_true',
                       help='Skip statistics plots')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for saved figures (default: 150)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载NPZ文件
    print(f"\nLoading data from: {args.input_file}")
    data = np.load(args.input_file)
    
    emg_data = data['emg_data']
    muscle_names = data['muscle_names']
    
    print(f"Data shape: {emg_data.shape}")
    print(f"  Gait cycles: {emg_data.shape[0]}")
    print(f"  Frames per cycle: {emg_data.shape[1]}")
    print(f"  Muscles: {emg_data.shape[2]}")
    print(f"\nMuscles:")
    for i, name in enumerate(muscle_names):
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        print(f"  {i}: {name}")
    
    # 生成综合图
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}")
    
    # 1. 所有周期综合图
    print("\n1. Creating comprehensive all-cycles plot...")
    fig1 = plot_all_cycles_comprehensive(emg_data, muscle_names, str(output_dir), args.prefix)
    output_file = os.path.join(output_dir, f'{args.prefix}all_muscles_all_cycles.png')
    fig1.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print(f"  ✓ {output_file}")
    plt.close(fig1)
    
    # 2. 热力图
    if not args.no_heatmap:
        print("\n2. Creating heatmap...")
        fig2 = plot_summary_heatmap(emg_data, muscle_names, str(output_dir), args.prefix)
        output_file = os.path.join(output_dir, f'{args.prefix}mean_heatmap.png')
        fig2.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        print(f"  ✓ {output_file}")
        plt.close(fig2)
    
    # 3. 统计图
    if not args.no_stats:
        print("\n3. Creating statistics plot...")
        fig3 = plot_activation_statistics(emg_data, muscle_names, str(output_dir), args.prefix)
        output_file = os.path.join(output_dir, f'{args.prefix}statistics.png')
        fig3.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        print(f"  ✓ {output_file}")
        plt.close(fig3)
    
    # 4. 单个肌肉图
    if args.individual:
        print("\n4. Creating individual muscle plots...")
        plot_individual_muscles(emg_data, muscle_names, str(output_dir), args.prefix)
    
    print(f"\n✓ Visualization complete!")
    print(f"All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
