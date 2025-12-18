# Backward GaitNet 推理数据格式说明

## 📊 输入数据格式

### 1. 总体结构
```
输入维度: 6073 = 6060 (运动学数据) + 13 (已知参数)
```

### 2. 运动学数据部分 (6060维)

**原始形状**: `(60 frames, 101 features)` → **展平为**: `(6060,)`

#### 特征结构 (101维/帧):

```
索引 0-8:   Pelvis (Free Joint, 6 DOF)
            [0-5] rotation matrix (前两行): r00, r01, r02, r10, r11, r12
            [6-8] translation: tx, ty, tz

索引 9-14:  右髋 FemurR (Ball Joint, 3 DOF → 6 features)
            rotation matrix (前两行): r00, r01, r02, r10, r11, r12

索引 15:    右膝 TibiaR (Revolute Joint, 1 DOF → 1 feature)
            单轴旋转角度

索引 16-21: 右踝 TalusR (Ball Joint, 3 DOF → 6 features)
            rotation matrix (前两行)

索引 22:    右脚小趾 FootPinkyR (Revolute Joint, 1 DOF)
索引 23:    右脚大趾 FootThumbR (Revolute Joint, 1 DOF)

索引 24-29: 左髋 FemurL (Ball Joint, 3 DOF → 6 features)
            rotation matrix (前两行)

索引 30:    左膝 TibiaL (Revolute Joint, 1 DOF → 1 feature)
            单轴旋转角度

索引 31-36: 左踝 TalusL (Ball Joint, 3 DOF → 6 features)
            rotation matrix (前两行)

索引 37:    左脚小趾 FootPinkyL (Revolute Joint, 1 DOF)
索引 38:    左脚大趾 FootThumbL (Revolute Joint, 1 DOF)

索引 39-44: 脊柱 Spine (Ball Joint, 3 DOF → 6 features)

索引 45+:   上肢、头部等其他关节
```

#### 数据组织方式:
```python
# 60帧数据按时间顺序排列
frame_0[101 features] + frame_1[101 features] + ... + frame_59[101 features]
= 6060 个连续的特征值
```

### 3. 已知参数部分 (13维)

```
索引 0: stride (步长)
索引 1: cadence (步频)
索引 2-12: 其他已知参数 (11个)
         - 可能包括：身高、体重、质量分布等
```

### 4. 完整输入示例

```python
bgn_input = np.array([
    # 运动学数据 (6060维)
    motion_frame0_feature0,    # Pelvis r00
    motion_frame0_feature1,    # Pelvis r01
    ...
    motion_frame0_feature100,  # 第0帧最后一个特征
    motion_frame1_feature0,    # 第1帧第一个特征
    ...
    motion_frame59_feature100, # 第59帧最后一个特征
    
    # 已知参数 (13维)
    stride,                    # 6060: 步长
    cadence,                   # 6061: 步频
    known_param_2,             # 6062
    ...
    known_param_12             # 6072
])
```

---

## 📤 输出数据格式

### 1. 总体结构
```
输出维度: 279 = 13 (已知参数) + 266 (推理参数)
```

### 2. 参数组成

```python
output_params = [
    # 已知参数 (13维) - 直接传递
    stride,           # 0
    cadence,          # 1
    known_param_2,    # 2
    ...
    known_param_12,   # 12
    
    # 推理参数 (266维) - 由 BGN 推理得到
    predicted_param_0,   # 13
    predicted_param_1,   # 14
    ...
    predicted_param_265  # 278
]
```

### 3. 推理参数内容 (266维)

这些参数包含：
- **肌肉激活状态** (~200+维)
  - 每块肌肉的激活强度
  - 左右对称的肌肉组
  
- **骨骼参数** (~60+维)
  - 关节刚度 (kp, kv)
  - 质量分布
  - 几何参数

---

## 🔄 数据流程图

```
原始CSV数据 (286/287 frames)
    ↓ [resample_data]
60帧关节角度 (60, 13 columns)
    ↓ [map_csv_to_56_dofs]
56 DOF数据 (60, 56)
    ↓ [expand_to_viewer_format]
101特征数据 (60, 101)
    ↓ [flatten + add known params]
6073维输入向量
    ↓ [BGN inference]
279维参数向量
```

---

## 💻 代码示例

### 完整推理流程

```python
import numpy as np
import torch
from advanced_vae import AdvancedVAE

# 1. 准备输入数据
motion_data = np.load('motion.npz')['motions']  # (60, 101)
motion_flat = motion_data.flatten()              # (6060,)

known_params = np.array([
    0.8,   # stride
    1.2,   # cadence
    # ... 其他11个参数
])

bgn_input = np.concatenate([motion_flat, known_params])  # (6073,)

# 2. 加载模型
model = AdvancedVAE(
    pose_dof=101,
    frame_num=60,
    num_known_param=13,
    num_paramstate=279
)
model.load_state_dict(trained_weights)
model.eval()

# 3. 推理
with torch.no_grad():
    input_tensor = torch.from_numpy(bgn_input).float().unsqueeze(0)
    
    # 编码
    mu, log_var = model.encode(input_tensor)
    z = model.reparameterize(mu, log_var)
    
    # 解码到参数空间
    known_param_tensor = input_tensor[:, -13:]
    z_with_known = torch.cat((z, known_param_tensor), dim=1)
    predicted_params = model.pre_decoder(z_with_known)
    
    # 组合完整输出
    full_params = torch.cat((known_param_tensor, predicted_params), dim=1)
    output = full_params.cpu().numpy()[0]  # (279,)

print(f"输出参数形状: {output.shape}")
print(f"Stride: {output[0]:.3f}")
print(f"Cadence: {output[1]:.3f}")
print(f"推理的肌肉参数范围: [{output[13:].min():.3f}, {output[13:].max():.3f}]")
```

---

## 🎯 关键要点

### 1. **数据归一化**
- 所有运动学数据已经过 `posToSixDof` 转换
- 旋转用 rotation matrix 表示，不是欧拉角
- 数值范围通常在 [-1, 1] 之间

### 2. **时序信息**
- 60帧 = 2个完整步态周期
- 每个周期约30帧
- 帧之间的时序关系通过 flatten 后的顺序保留

### 3. **关节自由度**
- Ball Joint (3 DOF) → **6个特征** (rotation matrix)
- Revolute Joint (1 DOF) → **1个特征** (角度)
- Free Joint (6 DOF) → **9个特征** (6 rotation + 3 translation)

### 4. **模型架构**
```
Encoder: (6073,) → [256, 256, 256] → 32 (latent)
PreDecoder: (32+13,) → [256, 256, 256] → 266 (predicted params)
Output: 13 (known) + 266 (predicted) = 279 total params
```

---

## 📝 注意事项

1. **输入维度必须严格匹配**: 6073 = 6060 + 13
2. **帧数固定**: 必须是60帧，不能多也不能少
3. **特征顺序**: 101个特征的顺序必须与骨架定义一致
4. **参数归一化**: 已知参数需要按照训练时的归一化方式处理
5. **设备兼容**: 注意 CPU/CUDA 的数据转换

---

## 🔗 相关文件

- `skeleton_gaitnet_narrow_model.xml` - 骨架结构定义
- `advanced_vae.py` - BGN 模型定义
- `train_backward_gaitnet.py` - 训练代码(展示数据处理)
- `convert_csv_to_viewer.py` - 您的推理实现示例
- `Character.cpp` - posToSixDof 转换实现
