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

class EMGAnomalyDetector:
    """EMG异常检测推理器"""
    
    def __init__(self, model_path):
        print(f"加载模型: {model_path}")
        self.model = load_model(model_path, compile=False)
        print(f"模型输入形状: {self.model.input_shape}")
    
    def resample_signal(self, signal, target_length=30):
        """将信号从200点降采样到30点"""
        # 确保信号是 numpy 数组
        signal = np.asarray(signal, dtype=np.float32)
        
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
        signal = np.asarray(signal, dtype=np.float32)
        signal_normalized = signal.copy()
        
        for ch in range(signal.shape[1]):
            min_val = signal[:, ch].min()
            max_val = signal[:, ch].max()
            if max_val > min_val:
                signal_normalized[:, ch] = (signal[:, ch] - min_val) / (max_val - min_val)
            else:
                # 如果所有值相同，设置为0.5
                signal_normalized[:, ch] = 0.5
        
        return signal_normalized
    
    def predict_reconstruction_error(self, signal, normalize=True, resample=True):
        """计算单个信号的重构误差"""
        # 转换为 numpy 数组
        signal = np.asarray(signal, dtype=np.float32)
        
        # 检查输入形状
        if len(signal.shape) == 1:
            # 如果是1D数组，假设是单通道信号
            signal = signal.reshape(-1, 1)
        
        # 降采样到30个
        if resample and signal.shape[0] != 30:
            signal = self.resample_signal(signal, target_length=30)
        
        # 归一化
        if normalize:
            signal = self.normalize_signal(signal)
        
        # 添加批次维度
        signal_batch = np.expand_dims(signal, axis=0)
        
        # 推理
        reconstruction = self.model.predict(signal_batch, verbose=0)[0]
        
        # 计算误差
        mse = np.mean((signal - reconstruction) ** 2)
        mae = np.mean(np.abs(signal - reconstruction))

        print(f"  MSE (均方误差): {mse:.6f}  MAE (平均绝对误差): {mae:.6f}")
        
        return mse, mae, reconstruction, signal
    
