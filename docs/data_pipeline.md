# FlightMoE 数据预处理流程

## 输入数据

原始数据来自 RflyMAD 数据集，包含：
- 41 维传感器时序数据（IMU、PWM、电池、GPS、姿态等）
- 飞行阶段标签（由 CaseID 提取）
- 故障标签（正常/异常）

## 处理流程

### 1. 数据解析
- 从原始日志解析为结构化 CSV
- 提取 41 个关键传感器字段
- 按飞行阶段分组（hover/waypoint/velocity/circling/acce/dece）

### 2. Z-Score 标准化
- 计算训练集每个传感器的 mean 和 std
- 对所有数据进行标准化：`(x - mean) / std`
- **注意**：std=0 的字段（如 battery_temperature）会产生 NaN，后续需填充为 0

### 3. 滑动窗口切分
- 窗口大小：128 时间步
- 步长：根据数据密度确定
- 输出：`temporal` 数组 `[N, 128, 41]`

### 4. STFT 时频图像生成
- 对每个窗口做短时傅里叶变换
- 输出：`spectral` 数组 `[N, 4, 17, 9, 3]`
  - 4: 4 张频谱图（对应不同传感器组或 STFT 参数）
  - 17: 频率 bin 数
  - 9: 时间窗口数
  - 3: RGB 通道
- 同时生成可视化 PNG 图像

### 5. 数据集划分
- **训练集**：正常样本（各种飞行阶段混合）
- **验证集**：正常+异常混合，用于调参
- **闭集测试集**：与训练集同分布的异常样本
- **开集测试集**：真实飞行数据（分布偏移）
- **划分策略**：按 CaseID 分层，确保各集合包含所有故障类型

### 6. 物理一致性邻接矩阵
- 按 6 个飞行阶段分别计算
- 对 mean 和 diff 分别做 L/H 离散化（10%/90% 分位数）
- NumPy 向量化 2-项集计数，计算对称置信度
- 隔离伪恒定字段（与所有其他字段关联清零）

## 输出文件

| 文件 | 形状 | 说明 |
|------|------|------|
| `train.npz` | temporal: [5300,128,41] | 训练集 |
| | spectral: [5300,4,17,9,3] | STFT 频域数据 |
| | anomaly_labels: [5300] | 0=正常 |
| | phase_labels: [5300] | 0-5 飞行阶段 |
| `val.npz` | [17263,...] | 验证集 |
| `test_closed.npz` | [69134,...] | 闭集测试 |
| `test_open.npz` | [27946,...] | 开集测试 |
| `zscore_params.npz` | mean: [41], std: [41] | 标准化参数 |
| `adjacency_{phase}.npy` | [41,41] | 物理一致性矩阵 |

## 使用方式

```python
import numpy as np

# 加载时序数据
data = np.load('data/preprocessed/train.npz')
temporal = data['temporal']      # [N, 128, 41]
spectral = data['spectral']      # [N, 4, 17, 9, 3]
labels = data['anomaly_labels']  # [N]
phases = data['phase_labels']    # [N]

# 加载标准化参数
zscore = np.load('data/preprocessed/zscore_params.npz')
mean, std = zscore['mean'], zscore['std']

# 加载邻接矩阵
adj = np.load('data/adjacency/adjacency_hover.npy')  # [41, 41]
```
