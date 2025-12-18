"""
将患者下肢数据与健康人上半身数据融合
用于生成 Backward GaitNet 的输入数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate
import sys
import os

# 添加Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'build/python'))

def load_patient_csv_data(left_csv, right_csv, target_frames=60):
    """
    从原始CSV文件加载患者数据
    
    Args:
        left_csv: 左侧运动学数据CSV文件路径
        right_csv: 右侧运动学数据CSV文件路径
        target_frames: 目标帧数（默认60，对应2个步态周期）
    
    Returns:
        patient_motions: (60, 101) 患者运动数据（只填充有数据的部分）
        joint_angles: dict 包含原始关节角度数据
    """
    print(f"\n  从CSV加载患者数据...")
    print(f"    左侧: {left_csv}")
    print(f"    右侧: {right_csv}")
    
    # 读取CSV文件，跳过前两行表头
    left_df = pd.read_csv(left_csv, skiprows=2)
    right_df = pd.read_csv(right_csv, skiprows=2)
    
    print(f"    左侧数据: {len(left_df)} 行")
    print(f"    右侧数据: {len(right_df)} 行")
    
    # 提取关节角度数据（度数）
    joint_angles = {
        'right_hip': right_df[['X', 'Y', 'Z']].values,  # 列1-3
        'right_knee': right_df[['X.1', 'Y.1', 'Z.1']].values,  # 列4-6
        'right_ankle': right_df[['X.2', 'Y.2', 'Z.2']].values,  # 列7-9
        'right_pelvis': right_df[['X.3', 'Y.3', 'Z.3']].values,  # 列10-12
        'right_foot': right_df['X.4'].values,  # 列13
        'left_hip': left_df[['X', 'Y', 'Z']].values,
        'left_knee': left_df[['X.1', 'Y.1', 'Z.1']].values,
        'left_ankle': left_df[['X.2', 'Y.2', 'Z.2']].values,
        'left_pelvis': left_df[['X.3', 'Y.3', 'Z.3']].values,
        'left_foot': left_df['X.4'].values,
    }
    
    # 插值到目标帧数（左右侧分别处理，因为长度可能不同）
    for key in joint_angles:
        original_frames = len(joint_angles[key])
        if original_frames != target_frames:
            x_old = np.linspace(0, 1, original_frames)
            x_new = np.linspace(0, 1, target_frames)
            
            if joint_angles[key].ndim == 2:
                # 3维角度
                interpolated = np.zeros((target_frames, 3))
                for i in range(3):
                    f = interp1d(x_old, joint_angles[key][:, i], kind='cubic')
                    interpolated[:, i] = f(x_new)
                joint_angles[key] = interpolated
            else:
                # 1维角度
                f = interp1d(x_old, joint_angles[key], kind='cubic')
                joint_angles[key] = f(x_new)
    
    print(f"    插值完成 -> {target_frames} 帧")
    
    # 创建101维的运动数据数组，初始化为0
    patient_motions = np.zeros((target_frames, 101))
    
    # 将角度转换为弧度并填充到对应位置
    # 骨架特征索引：
    # - Pelvis (Free joint): 0-8 (rotation 6 + translation 3)
    # - Right Hip (Ball joint): 9-14 (rotation matrix first 2 rows, 6 features)
    # - Right Knee (Revolute): 15 (1 feature, X-axis rotation)
    # - Right Ankle (Ball joint): 16-21 (6 features)
    # - Right Toes (Revolute): 22-23
    # - Left Hip (Ball joint): 24-29
    # - Left Knee (Revolute): 30
    # - Left Ankle (Ball joint): 31-36
    # - Left Toes (Revolute): 37-38
    
    # 将度数转换为弧度
    deg_to_rad = np.pi / 180.0
    
    def euler_to_rotation_matrix_first_two_rows(roll, pitch, yaw):
        """
        将XYZ欧拉角转换为旋转矩阵的前两行（6个值）
        
        Args:
            roll, pitch, yaw: 欧拉角（弧度）
        
        Returns:
            前两行的6个值 [r00, r01, r02, r10, r11, r12]
        """
        # 旋转矩阵 = Rz(yaw) * Ry(pitch) * Rx(roll)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # 第一行
        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr
        
        # 第二行
        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr
        
        return np.array([r00, r01, r02, r10, r11, r12])
    
    # 填充右侧关节
    # 右髋 (Ball joint, 3 DOF -> 6 features in rotation matrix)
    for frame in range(target_frames):
        angles_rad = joint_angles['right_hip'][frame] * deg_to_rad
        patient_motions[frame, 9:15] = euler_to_rotation_matrix_first_two_rows(
            angles_rad[0], angles_rad[1], angles_rad[2]
        )
    
    # 右膝 (Revolute, 1 DOF)
    patient_motions[:, 15] = joint_angles['right_knee'][:, 0] * deg_to_rad  # X轴
    
    # 右踝 (Ball joint, 3 DOF -> 6 features)
    for frame in range(target_frames):
        angles_rad = joint_angles['right_ankle'][frame] * deg_to_rad
        patient_motions[frame, 16:22] = euler_to_rotation_matrix_first_two_rows(
            angles_rad[0], angles_rad[1], angles_rad[2]
        )
    
    # 填充左侧关节
    # 左髋 (Ball joint, 3 DOF -> 6 features)
    for frame in range(target_frames):
        angles_rad = joint_angles['left_hip'][frame] * deg_to_rad
        patient_motions[frame, 24:30] = euler_to_rotation_matrix_first_two_rows(
            angles_rad[0], angles_rad[1], angles_rad[2]
        )
    
    # 左膝 (Revolute, 1 DOF)
    patient_motions[:, 30] = joint_angles['left_knee'][:, 0] * deg_to_rad
    
    # 左踝 (Ball joint, 3 DOF -> 6 features)
    for frame in range(target_frames):
        angles_rad = joint_angles['left_ankle'][frame] * deg_to_rad
        patient_motions[frame, 31:37] = euler_to_rotation_matrix_first_two_rows(
            angles_rad[0], angles_rad[1], angles_rad[2]
        )
    
    # 填充骨盆角度（Pelvis作为躯干基础）
    # Pelvis是Free joint，前6个是rotation（rotation matrix前2行）
    for frame in range(target_frames):
        angles_rad = joint_angles['right_pelvis'][frame] * deg_to_rad
        patient_motions[frame, 0:6] = euler_to_rotation_matrix_first_two_rows(
            angles_rad[0], angles_rad[1], angles_rad[2]
        )
    
    # 统计
    filled_features = []
    for i in range(101):
        if patient_motions[:, i].std() > 0.001:
            filled_features.append(i)
    
    print(f"    ✓ 患者数据加载完成: {patient_motions.shape}")
    print(f"    已填充 {len(filled_features)} 个特征: {filled_features}")
    print(f"    右膝角度: [{joint_angles['right_knee'][:, 0].min():.1f}°, {joint_angles['right_knee'][:, 0].max():.1f}°]")
    print(f"    左膝角度: [{joint_angles['left_knee'][:, 0].min():.1f}°, {joint_angles['left_knee'][:, 0].max():.1f}°]")
    
    return patient_motions, joint_angles

def load_motion_data(file_path):
    """加载运动数据"""
    data = np.load(file_path)
    motions = data['motions'].reshape(60, 101)
    params = data['params']
    return motions, params

def phase_align(patient_data, healthy_data, joint_index):
    """
    通过相位对齐匹配患者和健康人的步态周期
    
    Args:
        patient_data: 患者数据 (60, 101)
        healthy_data: 健康人数据 (60, 101)
        joint_index: 用于计算相位的关节索引（单个特征）
    
    Returns:
        aligned_patient_data: 对齐后的患者数据
    """
    # 使用指定关节特征计算相位
    patient_knee = patient_data[:, joint_index]
    healthy_knee = healthy_data[:, joint_index]
    
    # 使用互相关找到最佳对齐位置
    correlation = correlate(healthy_knee, patient_knee, mode='same')
    shift = np.argmax(correlation) - len(patient_knee) // 2
    
    print(f"  相位偏移: {shift} 帧")
    
    # 对患者数据进行循环移位以对齐相位
    aligned_patient_data = np.roll(patient_data, shift, axis=0)
    
    return aligned_patient_data

def merge_patient_healthy_data(patient_left_csv, patient_right_csv, healthy_file, output_file, use_all_valid):
    """
    将患者有数据的特征覆盖到健康人数据上
    
    Args:
        patient_left_csv: 患者左侧运动学数据CSV文件
        patient_right_csv: 患者右侧运动学数据CSV文件
        healthy_file: 健康人数据npz文件
        output_file: 输出文件路径
        use_all_valid: 如果为True，替换所有患者有数据的特征；否则只替换下肢
    """
    print("=" * 70)
    print("融合患者数据与健康人数据 (从CSV源数据)")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    patient_motions, joint_angles = load_patient_csv_data(patient_left_csv, patient_right_csv)
    healthy_motions, healthy_params = load_motion_data(healthy_file)
    
    print(f"  患者数据形状: {patient_motions.shape}")
    print(f"  健康人数据形状: {healthy_motions.shape}")
    
    # 2. 自动检测患者数据中的有效特征
    print("\n[2] 检测患者数据中的有效特征...")
    valid_indices = []
    threshold = 0.001
    
    for i in range(101):
        if patient_motions[:, i].std() > threshold:
            valid_indices.append(i)
    
    print(f"  检测到 {len(valid_indices)} 个有效特征 (std > {threshold})")
    print(f"  特征索引: {valid_indices[:20]}{'...' if len(valid_indices) > 20 else ''}")
    
    if len(valid_indices) == 0:
        print("\n  ⚠️  警告: 患者数据中未检测到有效特征！")
        return healthy_motions, healthy_params
    
    # 3. 定义要替换的特征
    if use_all_valid:
        replace_indices = valid_indices
        print(f"\n[3] 替换策略: 覆盖所有患者有数据的特征 ({len(replace_indices)} 个)")
    else:
        # 只替换下肢特征
        lower_body_range = list(range(9, 39))  # 髋、膝、踝关节范围
        replace_indices = [i for i in valid_indices if i in lower_body_range]
        print(f"\n[3] 替换策略: 仅下肢特征 ({len(replace_indices)} 个)")
    
    if len(replace_indices) == 0:
        print("\n  ⚠️  警告: 没有要替换的特征！")
        return healthy_motions, healthy_params
    
    print(f"  替换范围: 特征 {min(replace_indices)} - {max(replace_indices)}")
    
    # 4. 数据偏移
    print("\n[4] 患者数据偏移...")
    # 直接左移12帧（相当于右移-12帧，向前循环）
    shift = -12
    print(f"  固定偏移: {shift} 帧（左移12帧）")
    aligned_patient = np.roll(patient_motions, shift, axis=0)
    
    # 5. 创建融合数据
    print("\n[5] 融合数据...")
    merged_motions = np.copy(healthy_motions)
    
    # 用患者的数据替换
    merged_motions[:, replace_indices] = aligned_patient[:, replace_indices]
    
    print(f"  融合后数据形状: {merged_motions.shape}")
    print(f"  已替换 {len(replace_indices)} 个特征")
    print(f"  保留 {101 - len(replace_indices)} 个健康人特征")
    
    # 6. 使用健康人的参数（因为患者CSV没有参数数据）
    # 可以根据患者数据估算一些参数，这里暂时使用健康人参数
    merged_params = np.copy(healthy_params)
    print(f"\n  注意: 使用健康人参数作为初始值（患者CSV未包含参数）")
    
    # 7. 可视化对比
    print("\n[6] 生成对比图...")
    
    # 绘制所有被替换的特征
    frames = np.arange(60)
    
    # 为所有被替换的特征创建图表
    n_replaced = len(replace_indices)
    if n_replaced > 0:
        # 计算合适的布局
        n_cols = 4
        n_rows = (n_replaced + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))
        fig.suptitle(f'Patient Data Fusion: All {n_replaced} Replaced Features', 
                     fontsize=16, fontweight='bold')
        
        # 确保axes是二维数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 定义特征名称映射
        feature_names = {
            0: 'Pelvis_R00', 1: 'Pelvis_R01', 2: 'Pelvis_R02',
            3: 'Pelvis_R10', 4: 'Pelvis_R11', 5: 'Pelvis_R12',
            9: 'RHip_R00', 10: 'RHip_R01', 11: 'RHip_R02',
            12: 'RHip_R10', 13: 'RHip_R11', 14: 'RHip_R12',
            15: 'RKnee',
            16: 'RAnkle_R00', 17: 'RAnkle_R01', 18: 'RAnkle_R02',
            19: 'RAnkle_R10', 20: 'RAnkle_R11', 21: 'RAnkle_R12',
            24: 'LHip_R00', 25: 'LHip_R01', 26: 'LHip_R02',
            27: 'LHip_R10', 28: 'LHip_R11', 29: 'LHip_R12',
            30: 'LKnee',
            31: 'LAnkle_R00', 32: 'LAnkle_R01', 33: 'LAnkle_R02',
            34: 'LAnkle_R10', 35: 'LAnkle_R11', 36: 'LAnkle_R12',
        }
        
        for plot_idx, feature_idx in enumerate(replace_indices):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            # 绘制三条曲线
            ax.plot(frames, patient_motions[:, feature_idx], 'r-', 
                    linewidth=1.5, label='Patient', alpha=0.7)
            ax.plot(frames, healthy_motions[:, feature_idx], 'b--', 
                    linewidth=1, label='Healthy', alpha=0.5)
            ax.plot(frames, merged_motions[:, feature_idx], 'g-', 
                    linewidth=2, label='Merged', alpha=0.8)
            
            ax.axvline(x=30, color='gray', linestyle=':', alpha=0.3)
            ax.set_xlabel('Frame', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            
            feature_name = feature_names.get(feature_idx, f'Feature{feature_idx}')
            ax.set_title(f'{feature_name} [{feature_idx}]', fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(labelsize=7)
        
        # 隐藏多余的子图
        for plot_idx in range(n_replaced, n_rows * n_cols):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        comparison_file = output_file.replace('.npz', '_comparison_all.png')
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"  完整对比图保存到: {comparison_file}")
        plt.close()
    
    # 额外生成关键关节的简化图
    key_joints = [
        ('Right Hip', 9), ('Right Knee', 15), ('Right Ankle', 16),
        ('Left Hip', 24), ('Left Knee', 30), ('Left Ankle', 31),
    ]
    
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12))
    fig2.suptitle('Patient Data Fusion: Key Joints Summary', fontsize=14, fontweight='bold')
    
    for idx, (joint_name, joint_idx) in enumerate(key_joints):
        ax = axes2[idx // 2, idx % 2]
        is_replaced = joint_idx in replace_indices
        
        if is_replaced:
            ax.plot(frames, patient_motions[:, joint_idx], 'r-', 
                    linewidth=2, label='Patient', alpha=0.7)
            ax.plot(frames, healthy_motions[:, joint_idx], 'b--', 
                    linewidth=1.5, label='Healthy', alpha=0.5)
            ax.plot(frames, merged_motions[:, joint_idx], 'g-', 
                    linewidth=2.5, label='Merged', alpha=0.9)
        else:
            ax.plot(frames, healthy_motions[:, joint_idx], 'b-', 
                    linewidth=2, label='Healthy (kept)', alpha=0.7)
        
        ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        
        status = "✓ Replaced" if is_replaced else "✗ Not replaced"
        ax.set_title(f'{joint_name} [{joint_idx}] - {status}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    comparison_file2 = output_file.replace('.npz', '_comparison.png')
    plt.savefig(comparison_file2, dpi=150, bbox_inches='tight')
    print(f"  关键关节对比图保存到: {comparison_file2}")
    plt.close()
    
    # 8. 保存融合数据
    print("\n[7] 保存融合数据...")
    np.savez_compressed(output_file, motions=merged_motions.reshape(1, 6060), params=merged_params)
    print(f"  ✓ 融合数据保存到: {output_file}")
    
    # 9. 统计信息
    print("\n[8] 融合数据统计:")
    print(f"  形状: motions={merged_motions.reshape(1, 6060).shape}, params={merged_params.shape}")
    print(f"  数据范围: [{merged_motions.min():.3f}, {merged_motions.max():.3f}]")
    print(f"  Stride: {merged_params[0, 0]:.3f}")
    print(f"  Cadence: {merged_params[0, 1]:.3f}")
    
    # 10. 分析融合效果
    print("\n[9] 特征变化分析:")
    print(f"  {'特征名称':<35} {'健康std':>10} {'患者std':>10} {'融合std':>10} {'变化%':>10}")
    print("  " + "-" * 80)
    
    for joint_name, joint_idx, is_replaced in joints:
        healthy_std = healthy_motions[:, joint_idx].std()
        merged_std = merged_motions[:, joint_idx].std()
        patient_std = patient_motions[:, joint_idx].std()
        
        if is_replaced:
            change_pct = ((merged_std - healthy_std) / healthy_std * 100) if healthy_std > 0.001 else 0
            print(f"  {joint_name:<35} {healthy_std:>10.4f} {patient_std:>10.4f} {merged_std:>10.4f} {change_pct:>9.1f}%")
        else:
            print(f"  {joint_name:<35} {healthy_std:>10.4f} {'N/A':>10} {merged_std:>10.4f} {'0.0%':>10}")
    
    print(f"\n  替换特征总数: {len(replace_indices)}/{101}")
    print(f"  数据覆盖率: {len(replace_indices)/101*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ 融合完成！")
    print("=" * 70)
    
    return merged_motions, merged_params

def analyze_phase_correlation(patient_file, healthy_file):
    """分析患者和健康人的相位相关性"""
    print("\n" + "=" * 70)
    print("步态相位相关性分析")
    print("=" * 70)
    
    patient_motions, _ = load_motion_data(patient_file)
    healthy_motions, _ = load_motion_data(healthy_file)
    
    # 分析左右膝关节
    joints = [('Right Knee', 15), ('Left Knee', 30)]
    
    for joint_name, joint_idx in joints:
        patient_data = patient_motions[:, joint_idx]
        healthy_data = healthy_motions[:, joint_idx]
        
        # 互相关
        correlation = correlate(healthy_data, patient_data, mode='same')
        best_shift = np.argmax(correlation) - len(patient_data) // 2
        max_corr = correlation[np.argmax(correlation)] / (np.std(patient_data) * np.std(healthy_data) * len(patient_data))
        
        print(f"\n{joint_name}:")
        print(f"  最佳偏移: {best_shift} 帧")
        print(f"  相关系数: {max_corr:.3f}")
        print(f"  患者数据范围: [{patient_data.min():.3f}, {patient_data.max():.3f}]")
        print(f"  健康人数据范围: [{healthy_data.min():.3f}, {healthy_data.max():.3f}]")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='融合患者下肢数据与健康人上半身数据（从CSV源数据）')
    parser.add_argument('--patient-left', type=str, 
                        default='patient_data/left运动学数据_插值.csv',
                        help='患者左侧运动学数据CSV文件')
    parser.add_argument('--patient-right', type=str,
                        default='patient_data/right运动学数据_插值.csv',
                        help='患者右侧运动学数据CSV文件')
    parser.add_argument('--healthy', type=str, default='motions/Sim_Healthy.npz',
                        help='健康人数据文件路径')
    parser.add_argument('--output', type=str, default='motions/Patient03_Merged_CSV.npz',
                        help='输出文件路径')
    parser.add_argument('--only-lower-body', action='store_true',
                        help='仅替换下肢特征（默认替换所有有数据的特征）')
    
    args = parser.parse_args()
    
    # 执行融合
    merged_motions, merged_params = merge_patient_healthy_data(
        args.patient_left, args.patient_right, args.healthy, 
        args.output, use_all_valid=not args.only_lower_body
    )
    
    print(f"\n💡 下一步: 使用融合数据 {args.output} 作为 Backward GaitNet 的输入")
    print(f"   这个数据包含: 患者CSV关节角度 + 健康人其余特征")
    print(f"\n   推荐命令:")
    print(f"   python infer_bgn_merged.py --input {args.output}")
