"""
使用融合数据进行 Backward GaitNet 推理
输入: 患者下肢 + 健康人躯干的融合运动数据
输出: 推理得到的步态参数
"""

import numpy as np
import torch
import sys
import os

# 添加Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'build/python'))

from advanced_vae import AdvancedVAE
import RayEnvManager as rem

def load_bgn_model(model_path, num_paramstate):
    """加载 Backward GaitNet 模型"""
    print(f"\n[1] 加载 BGN 模型: {model_path}")
    
    # 创建模型
    motion_dim = 6060  # 60 frames × 101 features
    num_known_param = 13
    
    model = AdvancedVAE(
        motion_dim=motion_dim,
        num_known_param=num_known_param,
        num_paramstate=num_paramstate,
        latent_dim=32
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ 模型加载成功")
    print(f"  - 输入维度: {motion_dim + num_known_param} (motion={motion_dim} + known_params={num_known_param})")
    print(f"  - 输出维度: {num_paramstate} (known={num_known_param} + predicted={num_paramstate - num_known_param})")
    
    return model

def prepare_bgn_input(motion_data, known_params):
    """
    准备 BGN 输入数据
    
    Args:
        motion_data: (60, 101) 或 (6060,) 融合运动数据
        known_params: (13,) 已知参数 或 完整参数数组
    
    Returns:
        input_tensor: (1, 6073) PyTorch tensor
    """
    # 1. 展平运动数据
    if motion_data.shape == (60, 101):
        motion_flat = motion_data.flatten()
    elif motion_data.shape == (6060,):
        motion_flat = motion_data
    else:
        motion_flat = motion_data.reshape(-1)[:6060]
    
    # 2. 提取已知参数
    if len(known_params) == 13:
        known_params_normalized = known_params
    else:
        # 从完整参数中提取前13个（已标准化）
        known_params_normalized = rem.getNormalizedParamStateFromParam(known_params)[:13]
    
    # 3. 拼接
    bgn_input = np.concatenate([motion_flat, known_params_normalized])
    
    # 4. 转换为 tensor
    input_tensor = torch.FloatTensor(bgn_input).unsqueeze(0)
    
    return input_tensor

def run_bgn_inference(model, input_tensor, num_known_param=13):
    """
    运行 BGN 推理
    
    Args:
        model: AdvancedVAE 模型
        input_tensor: (1, 6073) 输入数据
        num_known_param: 已知参数数量
    
    Returns:
        predicted_params: (279,) 完整参数（包含已知+预测）
    """
    with torch.no_grad():
        # 1. Encode
        mu, logvar = model.encode(input_tensor)
        
        # 2. Reparameterize
        z = model.reparameterize(mu, logvar)
        
        # 3. Pre-decode (预测未知参数)
        known_params = input_tensor[:, -num_known_param:]
        z_with_known = torch.cat([z, known_params], dim=1)
        predicted_unknown = model.pre_decoder(z_with_known)
        
        # 4. 组合完整参数
        full_params = torch.cat([known_params, predicted_unknown], dim=1)
        
        return full_params.squeeze(0).numpy()

def analyze_predicted_params(predicted_params):
    """分析预测的参数"""
    print("\n[4] 预测参数分析:")
    
    # 反标准化
    denormalized_params = rem.getParamFromNormalizedParamState(predicted_params)
    
    print(f"\n  已知参数 (前13个):")
    known_names = ["stride", "cadence", "step_width", "step_ratio_R", "step_ratio_L",
                   "stride_vel", "com_vel_x", "com_vel_y", "com_vel_z",
                   "step_height_R", "step_height_L", "foot_angle_R", "foot_angle_L"]
    for i, name in enumerate(known_names):
        print(f"    {name:20} = {denormalized_params[i]:.4f} (normalized: {predicted_params[i]:.4f})")
    
    print(f"\n  肌肉参数统计 (第14-279个):")
    muscle_params = predicted_params[13:]
    print(f"    数量: {len(muscle_params)}")
    print(f"    范围: [{muscle_params.min():.3f}, {muscle_params.max():.3f}]")
    print(f"    均值: {muscle_params.mean():.3f}")
    print(f"    标准差: {muscle_params.std():.3f}")
    
    # 找出异常值
    outliers = np.where(np.abs(muscle_params) > 3.0)[0]
    if len(outliers) > 0:
        print(f"    ⚠️  发现 {len(outliers)} 个异常值 (|value| > 3.0)")
        for idx in outliers[:5]:  # 只显示前5个
            print(f"       参数 {idx+13}: {muscle_params[idx]:.3f}")
    
    return denormalized_params

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用融合数据进行 BGN 推理')
    parser.add_argument('--input', type=str, default='motions/Patient03_Merged_CSV.npz',
                        help='输入融合数据文件')
    parser.add_argument('--model', type=str, default='bgn/bgn_narrow_model_entire_01',
                        help='BGN 模型路径')
    parser.add_argument('--output', type=str, default='motions/Patient03_Predicted.npz',
                        help='输出预测参数文件')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Backward GaitNet 推理 - 融合数据版本")
    print("=" * 70)
    
    # 1. 初始化环境
    print("\n[0] 初始化环境...")
    rem.initialize("", False, 0)
    num_param_state = rem.getNumParamState()
    print(f"  ✓ RayEnvManager 初始化完成")
    print(f"  - 参数状态维度: {num_param_state}")
    
    # 2. 加载模型
    model = load_bgn_model(args.model, num_param_state)
    
    # 3. 加载融合数据
    print(f"\n[2] 加载融合数据: {args.input}")
    data = np.load(args.input)
    motions = data['motions'].reshape(60, 101)
    params = data['params'].squeeze()
    
    print(f"  ✓ 数据加载成功")
    print(f"  - Motion shape: {motions.shape}")
    print(f"  - Params shape: {params.shape}")
    print(f"  - 数据来源: 患者右下肢 + 健康人躯干和左下肢")
    
    # 4. 准备输入
    print(f"\n[3] 准备 BGN 输入...")
    input_tensor = prepare_bgn_input(motions, params)
    print(f"  ✓ 输入准备完成: {input_tensor.shape}")
    
    # 5. 运行推理
    print(f"\n[3] 运行推理...")
    predicted_params = run_bgn_inference(model, input_tensor)
    print(f"  ✓ 推理完成: {predicted_params.shape}")
    
    # 6. 分析结果
    denormalized_params = analyze_predicted_params(predicted_params)
    
    # 7. 保存结果
    print(f"\n[5] 保存结果...")
    np.savez_compressed(
        args.output,
        params=predicted_params.reshape(1, -1),
        params_denormalized=denormalized_params.reshape(1, -1),
        motions=data['motions']  # 保留原始运动数据
    )
    print(f"  ✓ 结果保存到: {args.output}")
    
    # 8. 对比分析
    print(f"\n[6] 输入输出对比:")
    print(f"  输入参数 (已知13个):")
    for i in range(13):
        print(f"    [{i}] 输入={params[i]:.4f}, 输出={predicted_params[i]:.4f}, 差异={abs(params[i]-predicted_params[i]):.6f}")
    
    print("\n" + "=" * 70)
    print("✓ BGN 推理完成！")
    print("=" * 70)
    print(f"\n下一步: 使用 convert_csv_to_viewer.py 可视化推理结果")
    print(f"或者: 使用 Forward GaitNet 生成运动并对比")

if __name__ == "__main__":
    main()
