# 真实病人数据格式说明 - Backward GaitNet

## 概述
Backward GaitNet 可以从病人的步态数据反推出肌肉力量和骨骼参数。本文档详细说明了如何准备真实病人的数据。

---

## 1. 数据要求

### 1.1 输入数据维度
```python
input_dimension = pose_dof * frame_num + num_known_param
                = 56 * 60 + 2
                = 3362
```

### 1.2 数据组成
- **motion_sequence**: (56 DOFs × 60 frames) = 3360 维
- **known_parameters**: 2 维 (stride, cadence)

---

## 2. Motion Sequence (3360维)

### 2.1 关节自由度 (56 DOFs)

按以下顺序排列的关节角度（**单位：弧度**）：

```
1-6.   Pelvis (骨盆)
       - Pelvis_rot_x, Pelvis_rot_y, Pelvis_rot_z     (旋转)
       - Pelvis_pos_x, Pelvis_pos_y, Pelvis_pos_z     (位置)

7-15.  Right Leg (右腿)
       - FemurR_x, FemurR_y, FemurR_z                 (髋关节)
       - TibiaR                                       (膝关节)
       - TalusR_x, TalusR_y, TalusR_z                 (踝关节)
       - FootPinkyR, FootThumbR                       (脚趾)

16-24. Left Leg (左腿)
       - FemurL_x, FemurL_y, FemurL_z                 (髋关节)
       - TibiaL                                       (膝关节)
       - TalusL_x, TalusL_y, TalusL_z                 (踝关节)
       - FootPinkyL, FootThumbL                       (脚趾)

25-36. Spine & Head (脊柱和头部)
       - Spine_x, Spine_y, Spine_z                    (脊柱)
       - Torso_x, Torso_y, Torso_z                    (躯干)
       - Neck_x, Neck_y, Neck_z                       (颈部)
       - Head_x, Head_y, Head_z                       (头部)

37-46. Right Arm (右臂)
       - ShoulderR_x, ShoulderR_y, ShoulderR_z        (肩关节)
       - ArmR_x, ArmR_y, ArmR_z                       (上臂)
       - ForeArmR                                     (前臂)
       - HandR_x, HandR_y, HandR_z                    (手)

47-56. Left Arm (左臂)
       - ShoulderL_x, ShoulderL_y, ShoulderL_z        (肩关节)
       - ArmL_x, ArmL_y, ArmL_z                       (上臂)
       - ForeArmL                                     (前臂)
       - HandL_x, HandL_y, HandL_z                    (手)
```

### 2.2 数据格式
```python
# motion_sequence 为 (56, 60) 的矩阵
# 每列是一帧的56个关节角度
# 共60帧，代表1个完整步态周期

motion_flattened = motion_sequence.flatten()  # 展平为 3360 维向量
# 顺序: [frame0_dof0, frame0_dof1, ..., frame0_dof55,
#        frame1_dof0, frame1_dof1, ..., frame1_dof55,
#        ...
#        frame59_dof0, ..., frame59_dof55]
```

### 2.3 采样要求
- **帧数**: 60帧（固定）
- **时间跨度**: 1个完整步态周期（通常1-2秒）
- **采样方式**: 
  - 如果原始数据帧数不是60，需要插值到60帧
  - 建议使用三次样条插值保持运动平滑性

---

## 3. Known Parameters (2维)

### 3.1 Stride (步幅)
```python
stride_normalized = stride_actual / reference_stride
```
- **stride_actual**: 实际步幅（米）
- **reference_stride**: 参考步幅 = 1.0 米
- **正常范围**: 0.75 - 1.25
- **归一化方法**: 
  ```python
  stride_param = (stride_normalized - 0.75) / (1.25 - 0.75)  # 归一化到 [0, 1]
  ```

### 3.2 Cadence (步频)
```python
cadence_normalized = cadence_actual / reference_cadence
```
- **cadence_actual**: 实际步频（步/秒）
- **reference_cadence**: 参考步频 = 1.0 步/秒
- **正常范围**: 0.75 - 1.25
- **归一化方法**:
  ```python
  cadence_param = (cadence_normalized - 0.75) / (1.25 - 0.75)  # 归一化到 [0, 1]
  ```

### 3.3 参数测量方法
```python
# 从motion数据计算
def calculate_stride(motion_data):
    """
    计算步幅
    motion_data: (56, 60) 矩阵
    """
    # Pelvis_pos_z 是第6个DOF（索引5）
    pelvis_z = motion_data[5, :]
    stride = pelvis_z[-1] - pelvis_z[0]  # 一个周期内前进距离
    return stride

def calculate_cadence(motion_data, time_duration):
    """
    计算步频
    time_duration: 60帧的总时间（秒）
    """
    cadence = 1.0 / time_duration  # 步/秒
    return cadence
```

---

## 4. 完整数据准备流程

### 4.1 Python代码示例

```python
import numpy as np

def prepare_patient_data_for_bgn(joint_angles, stride, cadence):
    """
    准备病人数据用于Backward GaitNet推理
    
    Parameters:
    -----------
    joint_angles : np.ndarray, shape (n_frames, 56)
        原始关节角度数据（弧度），每行是一帧的56个DOF
    stride : float
        实际步幅（米）
    cadence : float  
        实际步频（步/秒）
        
    Returns:
    --------
    input_data : np.ndarray, shape (3362,)
        准备好的输入数据
    """
    
    # 1. 确保是60帧
    n_frames = joint_angles.shape[0]
    if n_frames != 60:
        # 使用插值调整到60帧
        from scipy.interpolate import interp1d
        old_indices = np.linspace(0, 1, n_frames)
        new_indices = np.linspace(0, 1, 60)
        
        interpolated = np.zeros((60, 56))
        for i in range(56):
            f = interp1d(old_indices, joint_angles[:, i], kind='cubic')
            interpolated[:, i] = f(new_indices)
        
        joint_angles = interpolated
    
    # 2. 展平motion数据 (转置使其按帧展平)
    motion_flattened = joint_angles.T.flatten()  # shape: (3360,)
    
    # 3. 归一化stride和cadence
    stride_normalized = (stride / 1.0 - 0.75) / 0.5  # [0.75-1.25] -> [0-1]
    cadence_normalized = (cadence / 1.0 - 0.75) / 0.5  # [0.75-1.25] -> [0-1]
    
    # 确保在[0, 1]范围内
    stride_normalized = np.clip(stride_normalized, 0, 1)
    cadence_normalized = np.clip(cadence_normalized, 0, 1)
    
    # 4. 组合最终输入
    known_params = np.array([stride_normalized, cadence_normalized])
    input_data = np.concatenate([motion_flattened, known_params])
    
    return input_data

# 使用示例
# 假设你有病人的关节角度数据
joint_angles = np.load('patient_joint_angles.npy')  # shape: (n_frames, 56)
stride_measured = 1.1  # 米
cadence_measured = 0.9  # 步/秒

# 准备输入数据
bgn_input = prepare_patient_data_for_bgn(joint_angles, stride_measured, cadence_measured)

print(f"Input shape: {bgn_input.shape}")  # 应该输出 (3362,)
```

### 4.2 使用Backward GaitNet推理

```python
import torch
from advanced_vae import load_gaitvae

# 1. 加载模型
bgn_model = load_gaitvae(
    checkpoint_path='bgn/bgn_narrow_model_entire_01',
    pose_dof=56,
    frame_num=60,
    num_knownparam=2,
    num_paramstate=304  # 2个已知参数 + 302个未知参数（肌肉+骨骼）
)

# 2. 推理
predicted_motion, predicted_params = bgn_model.render_forward(bgn_input)

# 3. 解析结果
print(f"Predicted motion shape: {predicted_motion.shape}")  # (3360,) - 重建的运动
print(f"Predicted params shape: {predicted_params.shape}")  # (304,) - 预测的身体参数

# predicted_params 包含:
# [0-1]:   stride, cadence (已知，用于条件化)
# [2-9]:   骨骼参数 (全局缩放、股骨、胫骨长度等)
# [10-303]: 肌肉力量参数 (304个肌肉单元的最大力量)
```

---

## 5. 数据采集建议

### 5.1 推荐的采集系统
- **运动捕捉系统**: Vicon, OptiTrack, Xsens等
- **采样率**: ≥100 Hz（后处理到30Hz或60帧）
- **标记点**: 至少覆盖所有主要关节

### 5.2 数据预处理
1. **滤波**: 
   - 低通滤波器（6-10 Hz截止频率）
   - 消除高频噪声

2. **步态周期分割**:
   - 从左脚着地到下一次左脚着地
   - 或从右脚着地到下一次右脚着地

3. **归一化时间**:
   - 将一个步态周期重采样为60帧
   - 使用三次样条插值

### 5.3 质量检查
```python
def validate_joint_angles(angles):
    """验证关节角度数据的合理性"""
    
    # 检查形状
    assert angles.shape[1] == 56, "Should have 56 DOFs"
    
    # 检查范围（大多数关节应在合理范围内）
    for i in range(56):
        if i < 6:  # Pelvis位置和旋转
            continue
        # 关节角度一般在 [-π, π] 范围内
        if np.abs(angles[:, i]).max() > np.pi * 2:
            print(f"Warning: DOF {i} has unusual range")
    
    # 检查连续性（相邻帧变化不应过大）
    diff = np.diff(angles, axis=0)
    if np.abs(diff).max() > 0.5:  # 0.5弧度 ≈ 30度
        print("Warning: Large frame-to-frame changes detected")
    
    return True
```

---

## 6. 常见问题

### Q1: 如果没有完整的56个DOF怎么办？
**A**: 如果只有下肢数据（如临床步态分析），可以：
1. 将上肢和躯干角度设为默认值（0或小角度）
2. 重点关注预测的下肢肌肉参数
3. 注意：骨盆运动数据是必需的

### Q2: 如何验证输入数据的质量？
**A**: 
```python
# 使用Forward GaitNet验证
# 如果输入数据合理，FGN应该能重建出相似的运动
fgn_output = forward_gaitnet_model(predicted_params)
reconstruction_error = np.mean((fgn_output - motion_flattened)**2)
print(f"Reconstruction MSE: {reconstruction_error}")
# 应该 < 0.01 弧度²
```

### Q3: Stride和Cadence测量不准确怎么办？
**A**: 
- Stride可以从运动捕捉数据直接计算（骨盆前进距离）
- Cadence可以从步态周期时间计算（1/周期时间）
- 如果测量困难，可以使用正常值：stride=1.0, cadence=1.0

### Q4: 输出的304个参数都是什么？
**A**:
```
参数 0-1:   stride, cadence (输入的已知参数)
参数 2:     global skeleton scale (全局骨骼缩放)
参数 3-9:   FemurL, FemurR, TibiaL, TibiaR, ArmL, ArmR, ForeArmL, ForeArmR
参数 10-11: FemurL_torsion, FemurR_torsion (股骨扭转)
参数 12-303: 304个肌肉单元的最大力量比例 (相对于标准值)
```

---

## 7. 输出解释

### 7.1 肌肉参数
预测的肌肉参数代表**相对于标准模型的力量比例**：
- 值 = 0.0: 该肌肉力量为标准模型的最小值（通常是标准值的5%）
- 值 = 0.5: 该肌肉力量为标准值
- 值 = 1.0: 该肌肉力量为标准模型的最大值（通常是标准值的100%）

### 7.2 关键肌肉索引
```python
# 从输出的304个参数中定位特定肌肉
# 肌肉名称在训练时已固定顺序

key_muscles = {
    'L_Tibialis_Anterior': 268,      # 左胫骨前肌
    'L_Gastrocnemius_Lateral': 140,  # 左腓肠肌外侧头
    'L_Gastrocnemius_Medial': 141,   # 左腓肠肌内侧头
    'L_Soleus': 275,                  # 左比目鱼肌
    # ... 其他肌肉索引需要查看具体模型
}
```

---

## 8. 完整工作流程示例

```bash
# 1. 准备数据
python prepare_patient_data.py --input patient_mocap.c3d --output patient_input.npz

# 2. 运行推理
python run_backward_gaitnet.py --input patient_input.npz --model bgn/bgn_narrow_model_entire_01 --output patient_results.json

# 3. 分析结果
python analyze_muscle_params.py --results patient_results.json --visualize
```

---

## 附录A: 数据文件格式

### 推荐的NPZ格式
```python
import numpy as np

# 保存
np.savez('patient_data.npz',
         joint_angles=joint_angles,  # (n_frames, 56)
         stride=stride,
         cadence=cadence,
         patient_id='P001',
         condition='Left_Foot_Drop')

# 加载
data = np.load('patient_data.npz')
joint_angles = data['joint_angles']
stride = data['stride']
cadence = data['cadence']
```

---

## 联系方式
如有技术问题，请联系原作者或参考论文：
- Paper: https://arxiv.org/pdf/2306.04161.pdf
- Contact: jungnam04@imo.snu.ac.kr
