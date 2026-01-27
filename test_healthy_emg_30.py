#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
健康肌电数据异常检测测试
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK KR']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


class EMGAnomalyDetector:
    """EMG异常检测推理器"""
    
    def __init__(self, model_path):
        print(f"加载模型: {model_path}")
        self.model = load_model(model_path, compile=False)
        print(f"模型输入形状: {self.model.input_shape}")
    
    def resample_signal(self, signal, target_length=33):
        """将信号重采样到33点（完整步态周期）"""
        if signal.shape[0] == target_length:
            return signal
        
        n_samples, n_features = signal.shape
        resampled = np.zeros((target_length, n_features), dtype=np.float32)
        
        for feat_idx in range(n_features):
            x_old = np.linspace(0, 1, n_samples)
            x_new = np.linspace(0, 1, target_length)
            f = interp1d(x_old, signal[:, feat_idx], kind='linear', fill_value='extrapolate')
            resampled[:, feat_idx] = f(x_new)
        
        return resampled
    
    def normalize_signal(self, signal):
        """Min-Max 归一化 [0, 1]"""
        signal_normalized = signal.copy().astype(np.float32)
        for ch in range(signal.shape[1]):
            min_val = signal[:, ch].min()
            max_val = signal[:, ch].max()
            if max_val > min_val:
                signal_normalized[:, ch] = (signal[:, ch] - min_val) / (max_val - min_val)
        return signal_normalized
    
    def predict_reconstruction_error(self, signal, normalize=True, resample=True):
        """计算单个信号的重构误差"""
        # 如果不是33帧，根据resample参数决定是否重采样
        if signal.shape[0] != 33:
            if resample:
                signal = self.resample_signal(signal, target_length=33)
        # 如果已经是33帧，直接跳过重采样
        
        # 确保形状 (33, 10)
        if signal.shape[0] != 33 or signal.shape[1] != 10:
            raise ValueError(f"信号形状应该是 (33, 10)，得到 {signal.shape}")
        
        # 归一化
        if normalize:
            signal = self.normalize_signal(signal)
        
        # 转换为 float32 并添加批次维度
        signal = signal.astype(np.float32)
        signal_batch = np.expand_dims(signal, axis=0)
        
        # 推理
        reconstruction = self.model.predict(signal_batch, verbose=0)[0]
        
        # 计算误差
        mse = np.mean((signal - reconstruction) ** 2)
        mae = np.mean(np.abs(signal - reconstruction))
        
        return mse, mae, reconstruction


def main():
    print("="*70)
    print("健康肌电数据异常检测测试")
    print("="*70 + "\n")
    
    # 加载推理器
    detector = EMGAnomalyDetector(
        "lstm_vae_reward/emg_lstm_vae_20260127_150617/best_model.h5"
    )
    
    # 加载健康肌电数据
    print("加载健康肌电数据...")
    health_data = np.load("kinematics_20260127_144844_0000_muscle_emg.npz")
    X_healthy = health_data['emg_data']  # 将自动重采样到33个时间步
    muscle_names = health_data['muscle_names']
    n_healthy_samples = len(X_healthy)
    
    print(f"✓ 健康数据形状: {X_healthy.shape}")
    print(f"✓ 样本数: {n_healthy_samples}")
    print(f"✓ 肌肉通道数: {len(muscle_names)}")
    
    # 加载评估结果（用于获取阈值）
    print("\n加载训练集评估结果...")
    eval_data = np.load("lstm_vae_reward/emg_lstm_vae_20260127_150617/evaluation_results.npz")
    test_mse = eval_data['mse']  # 测试集MSE
    test_mae = eval_data['mae']  # 测试集MAE
    
    print(f"✓ 测试集MSE统计:")
    print(f"  均值: {test_mse.mean():.6f}")
    print(f"  95%分位: {np.percentile(test_mse, 95):.6f}")
    print(f"  99%分位: {np.percentile(test_mse, 99):.6f}")
    
    # 异常检测阈值
    threshold_95 = np.percentile(test_mse, 95)  # 0.00475
    threshold_99 = np.percentile(test_mse, 99)  # 0.00762
    
    print("\n" + "="*70)
    print("对健康肌电数据进行异常检测")
    print("="*70 + "\n")
    
    healthy_mse = []
    healthy_mae = []
    results = []
    
    for i, signal in enumerate(X_healthy):
        print(f"处理样本 {i+1:2d}/{n_healthy_samples} ...", end=" ")
        
        mse, mae, recon = detector.predict_reconstruction_error(signal, normalize=True, resample=True)
        healthy_mse.append(mse)
        healthy_mae.append(mae)
        
        # 判断异常 (使用95%分位阈值)
        is_anomaly = mse > threshold_95
        status = "⚠️  异常" if is_anomaly else "✓ 正常"
        
        results.append({
            'index': i,
            'mse': mse,
            'mae': mae,
            'is_anomaly': is_anomaly,
            'signal': signal,
            'reconstruction': recon
        })
        
        print(f"MSE={mse:.6f} - {status}")
    
    healthy_mse = np.array(healthy_mse)
    healthy_mae = np.array(healthy_mae)
    
    # 统计结果
    print("\n" + "="*70)
    print("检测结果统计")
    print("="*70 + "\n")
    
    print(f"阈值设置:")
    print(f"  95% 分位 (敏感): {threshold_95:.6f}")
    print(f"  99% 分位 (保守): {threshold_99:.6f}")
    
    print(f"\n健康数据MSE统计:")
    print(f"  均值: {healthy_mse.mean():.6f}")
    print(f"  标准差: {healthy_mse.std():.6f}")
    print(f"  最小值: {healthy_mse.min():.6f}")
    print(f"  最大值: {healthy_mse.max():.6f}")
    print(f"  中位数: {np.median(healthy_mse):.6f}")
    
    print(f"\n健康数据MAE统计:")
    print(f"  均值: {healthy_mae.mean():.6f}")
    print(f"  标准差: {healthy_mae.std():.6f}")
    print(f"  最小值: {healthy_mae.min():.6f}")
    print(f"  最大值: {healthy_mae.max():.6f}")
    
    # 异常检测统计
    anomaly_count_95 = sum(1 for r in results if r['mse'] > threshold_95)
    anomaly_count_99 = sum(1 for r in results if r['mse'] > threshold_99)
    
    print(f"\n异常检测结果 (95% 分位阈值={threshold_95:.6f}):")
    print(f"  异常样本数: {anomaly_count_95}/{n_healthy_samples} ({100*anomaly_count_95/n_healthy_samples:.1f}%)")
    print(f"  正常样本数: {n_healthy_samples-anomaly_count_95}/{n_healthy_samples} ({100*(n_healthy_samples-anomaly_count_95)/n_healthy_samples:.1f}%)")
    
    print(f"\n异常检测结果 (99% 分位阈值={threshold_99:.6f}):")
    print(f"  异常样本数: {anomaly_count_99}/{n_healthy_samples} ({100*anomaly_count_99/n_healthy_samples:.1f}%)")
    print(f"  正常样本数: {n_healthy_samples-anomaly_count_99}/{n_healthy_samples} ({100*(n_healthy_samples-anomaly_count_99)/n_healthy_samples:.1f}%)")
    
    # 绘制对比图
    print("\n" + "="*70)
    print("生成对比图表")
    print("="*70 + "\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE分布对比
    ax = axes[0]
    x_test_mse = test_mse
    ax.hist(x_test_mse, bins=30, alpha=0.6, label='测试集 (训练数据)', color='blue', edgecolor='black')
    ax.hist(healthy_mse, bins=15, alpha=0.6, label='健康数据', color='green', edgecolor='black')
    ax.axvline(threshold_95, color='red', linestyle='--', linewidth=2, label=f'95%分位 ({threshold_95:.6f})')
    ax.axvline(threshold_99, color='orange', linestyle='--', linewidth=2, label=f'99%分位 ({threshold_99:.6f})')
    ax.set_xlabel('MSE (Mean Squared Error)')
    ax.set_ylabel('样本数')
    ax.set_title('MSE分布对比: 训练测试集 vs 健康数据')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # MAE分布对比
    ax = axes[1]
    test_mae = eval_data['mae']
    ax.hist(test_mae, bins=30, alpha=0.6, label='测试集 (训练数据)', color='blue', edgecolor='black')
    ax.hist(healthy_mae, bins=15, alpha=0.6, label='健康数据', color='green', edgecolor='black')
    ax.set_xlabel('MAE (Mean Absolute Error)')
    ax.set_ylabel('样本数')
    ax.set_title('MAE分布对比: 训练测试集 vs 健康数据')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = "healthy_emg_anomaly_detection.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {save_path}")
    plt.close()
    
    # # 异常样本可视化 (如果有的话)
    # if anomaly_count_95 > 0:
    #     print("\n生成异常样本重构对比图...")
    #     anomaly_results = [r for r in results if r['mse'] > threshold_95]
        
    #     for idx, anom_result in enumerate(anomaly_results[:3]):  # 最多显示3个异常样本
    #         fig, axes = plt.subplots(10, 1, figsize=(12, 13))
    #         fig.suptitle(f"异常样本 #{anom_result['index']+1} 重构对比\nMSE={anom_result['mse']:.6f}", 
    #                      fontsize=14, fontweight='bold')
            
    #         signal = anom_result['signal']
    #         recon = anom_result['reconstruction']
            
    #         for ch in range(10):
    #             ax = axes[ch]
    #             t = np.arange(200)
    #             ax.plot(t, signal[:, ch], 'b-', alpha=0.7, label='原始信号', linewidth=1.5)
    #             ax.plot(t, recon[:, ch], 'r--', alpha=0.7, label='重构信号', linewidth=1.5)
    #             ax.set_ylabel(f'肌肉{ch+1}')
    #             ax.grid(alpha=0.3)
    #             if ch == 0:
    #                 ax.legend(loc='upper right')
            
    #         axes[-1].set_xlabel('时间步')
    #         plt.tight_layout()
    #         save_path = f"healthy_emg_anomaly_sample{idx+1}.png"
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         print(f"✓ 异常样本 {idx+1} 已保存: {save_path}")
    #         plt.close()
    
    # 保存结果
    # print("\n" + "="*70)
    # print("保存检测结果")
    # print("="*70 + "\n")
    
    # np.savez('healthy_emg_anomaly_results.npz',
    #          mse=healthy_mse,
    #          mae=healthy_mae,
    #          threshold_95=threshold_95,
    #          threshold_99=threshold_99,
    #          anomaly_flags_95=np.array([r['is_anomaly'] for r in results]))
    
    # print("✓ 结果已保存到: healthy_emg_anomaly_results.npz")
    
    print("\n" + "="*70)
    print("检测完成!")
    print("="*70)


if __name__ == "__main__":
    main()
