import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# 选择要分析的 npz 文件
print("=" * 70)
print("Available NPZ files in motions/ directory:")
print("=" * 70)

motion_dir = 'motions'
npz_files = [f for f in os.listdir(motion_dir) if f.endswith('.npz')]
npz_files.sort()

if not npz_files:
    print("No .npz files found in motions/ directory!")
    sys.exit(1)

for idx, filename in enumerate(npz_files, 1):
    filepath = os.path.join(motion_dir, filename)
    file_size = os.path.getsize(filepath) / 1024  # KB
    print(f"  {idx}. {filename:<30} ({file_size:.1f} KB)")

print("\nEnter the number of the file you want to analyze (or press Enter for default):")
print(f"Default: 1 (Sim_Healthy.npz)")

try:
    choice = input("Your choice: ").strip()
    if choice == "":
        file_idx = 0
    else:
        file_idx = int(choice) - 1
        if file_idx < 0 or file_idx >= len(npz_files):
            print(f"Invalid choice! Using default: Sim_Healthy.npz")
            file_idx = 0
except (ValueError, KeyboardInterrupt, EOFError):
    print("\nUsing default: Sim_Healthy.npz")
    file_idx = 0

selected_file = npz_files[file_idx]
filepath = os.path.join(motion_dir, selected_file)

print(f"\n✓ Selected: {selected_file}")
print("=" * 70)

# 加载数据
data = np.load(filepath)
motions = data['motions'].reshape(60, 101)

print("\n" + "=" * 70)
print(f"Joint XYZ Analysis - {selected_file}")
print("=" * 70)

# 根据 skeleton_gaitnet_narrow_model.xml 和 posToSixDof 转换规则：
# - Pelvis (Free Joint): 0-8 (6 rotation + 3 translation = 9 features)
# - Ball Joint (3 DOF) → 6 features (rotation matrix前两行)
# - Revolute Joint (1 DOF) → 1 feature

# 正确的关节索引（基于实际骨架定义）：
joint_groups = {
    'Right Hip (FemurR)': (9, 10, 11, 12, 13, 14),      # Ball Joint, 6 features
    'Right Knee (TibiaR)': (15,),                        # Revolute Joint, 1 feature
    'Right Ankle (TalusR)': (16, 17, 18, 19, 20, 21),   # Ball Joint, 6 features
    'Left Hip (FemurL)': (24, 25, 26, 27, 28, 29),      # Ball Joint, 6 features
    'Left Knee (TibiaL)': (30,),                         # Revolute Joint, 1 feature
    'Left Ankle (TalusL)': (31, 32, 33, 34, 35, 36),    # Ball Joint, 6 features
}

# 创建图像
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Joint Angles - All Features (60 frames = 2 gait cycles)', 
             fontsize=14, fontweight='bold')

axes = axes.flatten()
frames = np.arange(60)

for idx, (joint_name, indices) in enumerate(joint_groups.items()):
    ax = axes[idx]
    
    # 根据关节类型绘制不同数量的特征
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    print(f"\n{joint_name}:")
    for feat_idx, col_idx in enumerate(indices):
        data = motions[:, col_idx]
        ax.plot(frames, data, color=colors[feat_idx % len(colors)], 
                linewidth=2, label=f'Feat{feat_idx} (col {col_idx})', alpha=0.7)
        
        print(f"  Feature {feat_idx} (col {col_idx}): range=[{data.min():.3f}, {data.max():.3f}], std={data.std():.3f}")
        
    # 标记周期分界线
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title(f'{joint_name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best', ncol=2)

plt.tight_layout()
# 使用文件名生成输出图像名
base_name = os.path.splitext(selected_file)[0]
output_path = f'joint_xyz_{base_name}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"✓ XYZ analysis image saved to: {output_path}")
print(f"{'='*70}")

# 生成原始欧拉角图像（从6维旋转矩阵反算回3维欧拉角）
print("\n" + "=" * 70)
print("Generating Original Euler Angles (3D) Visualization...")
print("=" * 70)

def rotation_matrix_to_euler(r00, r01, r02, r10, r11, r12):
    """
    从旋转矩阵的前两行反算XYZ欧拉角
    返回 (roll, pitch, yaw) 单位：弧度
    """
    # 计算第三行
    r20 = r01 * r12 - r02 * r11
    r21 = r02 * r10 - r00 * r12
    r22 = r00 * r11 - r01 * r10
    
    # 提取欧拉角
    pitch = np.arcsin(-r20)
    
    # 处理万向锁情况
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(r21, r22)
        yaw = np.arctan2(r10, r00)
    else:
        roll = 0
        yaw = np.arctan2(-r01, r11)
    
    return roll, pitch, yaw

# 定义Ball Joint关节（需要从6D转回3D）
ball_joints = {
    'Right Hip (FemurR)': (9, 10, 11, 12, 13, 14),
    'Right Ankle (TalusR)': (16, 17, 18, 19, 20, 21),
    'Left Hip (FemurL)': (24, 25, 26, 27, 28, 29),
    'Left Ankle (TalusL)': (31, 32, 33, 34, 35, 36),
}

# 定义Revolute Joint关节（已经是1D角度）
revolute_joints = {
    'Right Knee (TibiaR)': 15,
    'Left Knee (TibiaL)': 30,
}

# 创建欧拉角图像
fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12))
fig2.suptitle('Joint Angles - Original Euler Angles (Roll/Pitch/Yaw)', 
              fontsize=14, fontweight='bold')

axes2 = axes2.flatten()
plot_idx = 0

# 绘制Ball Joints的欧拉角
for joint_name, indices in ball_joints.items():
    ax = axes2[plot_idx]
    
    # 提取旋转矩阵前两行
    r00 = motions[:, indices[0]]
    r01 = motions[:, indices[1]]
    r02 = motions[:, indices[2]]
    r10 = motions[:, indices[3]]
    r11 = motions[:, indices[4]]
    r12 = motions[:, indices[5]]
    
    # 反算欧拉角
    roll = np.zeros(60)
    pitch = np.zeros(60)
    yaw = np.zeros(60)
    
    for frame in range(60):
        roll[frame], pitch[frame], yaw[frame] = rotation_matrix_to_euler(
            r00[frame], r01[frame], r02[frame],
            r10[frame], r11[frame], r12[frame]
        )
    
    # 转换为角度
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    # 绘制三个轴
    ax.plot(frames, roll_deg, 'r-', linewidth=2, label='Roll (X)', alpha=0.7)
    ax.plot(frames, pitch_deg, 'g-', linewidth=2, label='Pitch (Y)', alpha=0.7)
    ax.plot(frames, yaw_deg, 'b-', linewidth=2, label='Yaw (Z)', alpha=0.7)
    
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Angle (degrees)', fontsize=10)
    ax.set_title(f'{joint_name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    print(f"\n{joint_name} (Euler Angles):")
    print(f"  Roll (X): range=[{roll_deg.min():.1f}°, {roll_deg.max():.1f}°], std={roll_deg.std():.1f}°")
    print(f"  Pitch (Y): range=[{pitch_deg.min():.1f}°, {pitch_deg.max():.1f}°], std={pitch_deg.std():.1f}°")
    print(f"  Yaw (Z): range=[{yaw_deg.min():.1f}°, {yaw_deg.max():.1f}°], std={yaw_deg.std():.1f}°")
    
    plot_idx += 1

# 绘制Revolute Joints（单轴角度）
for joint_name, col_idx in revolute_joints.items():
    ax = axes2[plot_idx]
    
    angle_rad = motions[:, col_idx]
    angle_deg = np.degrees(angle_rad)
    
    ax.plot(frames, angle_deg, 'r-', linewidth=2, label='Angle (X)', alpha=0.7)
    
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Angle (degrees)', fontsize=10)
    ax.set_title(f'{joint_name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    print(f"\n{joint_name} (Single Axis):")
    print(f"  Angle: range=[{angle_deg.min():.1f}°, {angle_deg.max():.1f}°], std={angle_deg.std():.1f}°")
    
    plot_idx += 1

plt.tight_layout()
output_path_euler = f'joint_euler_{base_name}.png'
plt.savefig(output_path_euler, dpi=150, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"✓ Euler angles image saved to: {output_path_euler}")
print(f"{'='*70}")

# 额外分析：哪些特征最活跃
print("\n" + "=" * 70)
print("Active Feature Analysis:")
print("=" * 70)

for joint_name, indices in joint_groups.items():
    stds = [motions[:, col_idx].std() for col_idx in indices]
    max_std = max(stds)
    max_idx = stds.index(max_std)
    
    print(f"\n{joint_name}:")
    print(f"  Most active feature: Feature {max_idx} (col {indices[max_idx]}, std={max_std:.3f})")
    for feat_idx, (col_idx, std_val) in enumerate(zip(indices, stds)):
        bar = '█' * int(std_val * 50)
        print(f"  Feature {feat_idx} (col {col_idx}): {bar} {std_val:.3f}")
