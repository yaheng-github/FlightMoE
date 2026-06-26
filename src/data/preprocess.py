"""
数据预处理流水线：CSV -> 滑动窗口 -> Z-Score -> .npz

流程：
1. 读取 split_metadata.json 获取划分结果
2. 对每个集合遍历所有 CaseID
3. 提取 41 维特征 + 故障标签 + 飞行阶段标签
4. 滑动窗口切分（128帧窗口，64帧步长）
5. 纯净窗口筛选（异常帧比例 > 0 且 < 1 的窗口丢弃）
6. Z-Score 标准化（μ/σ 仅在训练集计算）
7. 保存为 .npz
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


# 41 维核心特征字段（与 Dataset_Fields_Reference.md 一致）
FEATURE_COLS = [
    '_actuator_controls_0_0_control[0]', '_actuator_controls_0_0_control[1]',
    '_actuator_controls_0_0_control[2]', '_actuator_controls_0_0_control[3]',
    '_actuator_outputs_0_output[0]', '_actuator_outputs_0_output[1]',
    '_actuator_outputs_0_output[2]', '_actuator_outputs_0_output[3]',
    '_sensor_combined_0_gyro_rad[0]', '_sensor_combined_0_gyro_rad[1]',
    '_sensor_combined_0_gyro_rad[2]',
    '_sensor_combined_0_accelerometer_m_s2[0]', '_sensor_combined_0_accelerometer_m_s2[1]',
    '_sensor_combined_0_accelerometer_m_s2[2]',
    '_vehicle_air_data_0_baro_alt_meter', '_vehicle_air_data_0_baro_pressure_pa',
    '_vehicle_air_data_0_baro_temp_celcius',
    '_vehicle_attitude_0_q[0]', '_vehicle_attitude_0_q[1]',
    '_vehicle_attitude_0_q[2]', '_vehicle_attitude_0_q[3]',
    '_vehicle_local_position_0_x', '_vehicle_local_position_0_y',
    '_vehicle_local_position_0_z', '_vehicle_local_position_0_vx',
    '_vehicle_local_position_0_vy', '_vehicle_local_position_0_vz',
    '_vehicle_magnetometer_0_magnetometer_ga[0]',
    '_vehicle_magnetometer_0_magnetometer_ga[1]',
    '_vehicle_magnetometer_0_magnetometer_ga[2]',
    '_battery_status_0_voltage_v', '_battery_status_0_current_a',
    '_battery_status_0_remaining', '_battery_status_0_temperature',
    '_vehicle_gps_position_0_lat', '_vehicle_gps_position_0_lon',
    '_vehicle_gps_position_0_alt', '_vehicle_gps_position_0_satellites_used',
    '_vehicle_gps_position_0_eph', '_vehicle_gps_position_0_epv',
    '_vehicle_gps_position_0_fix_type',
]

# 故障标签列（优先级：UAVState > rfly_ctrl）
FAULT_LABEL_COLS = ['UAVState_data_fault_state', '_rfly_ctrl_lxl_0_id']

# 飞行阶段标签列
PHASE_LABEL_COLS = ['_rfly_ctrl_lxl_0_mode', 'UAVState_data_cmd']


def parse_case_id(case_id: int) -> Tuple[int, int, int, int]:
    """解析 CaseID"""
    data_type = case_id // 1000000000
    flight_mode = (case_id % 1000000000) // 100000000
    fault_type = (case_id % 100000000) // 1000000
    case_num = case_id % 1000000
    return data_type, flight_mode, fault_type, case_num


def extract_fault_labels(df: pd.DataFrame) -> np.ndarray:
    """
    提取逐帧故障标签
    优先级：UAVState_data_fault_state > _rfly_ctrl_lxl_0_id
    返回：0=正常, 1=异常
    """
    n_frames = len(df)

    # 优先使用 UAVState_data_fault_state（地面真值，反映实际系统状态）
    if 'UAVState_data_fault_state' in df.columns:
        labels = df['UAVState_data_fault_state'].fillna(0).values
        return (labels > 0).astype(np.int32)

    # 次选 _rfly_ctrl_lxl_0_id（故障注入指令，提前约 0.55s）
    if '_rfly_ctrl_lxl_0_id' in df.columns:
        ids = df['_rfly_ctrl_lxl_0_id'].fillna(0).values
        # 0 或 1500 表示正常，123450~123549 表示故障
        labels = np.zeros(n_frames, dtype=np.int32)
        labels[(ids >= 123450) & (ids <= 123549)] = 1
        return labels

    # 都没有，默认全正常（Real 数据的 no_fault）
    return np.zeros(n_frames, dtype=np.int32)


def extract_phase_labels(case_id: int, n_frames: int) -> np.ndarray:
    """
    从 CaseID 解析飞行阶段标签
    CaseID = DataType * 1e9 + FlightMode * 1e8 + FaultType * 1e6 + CaseNum
    返回整数编码：0=hover, 1=waypoint, 2=velocity, 3=circling, 4=acce, 5=dece
    """
    flight_mode = (case_id % 1000000000) // 100000000
    return np.full(n_frames, flight_mode, dtype=np.int32)


def create_windows(
    features: np.ndarray,
    labels: np.ndarray,
    phase_labels: np.ndarray,
    window_size: int = 128,
    stride: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    滑动窗口切分

    Args:
        features: [N, 41] 特征矩阵
        labels: [N] 故障标签（0/1）
        phase_labels: [N] 飞行阶段标签
        window_size: 窗口长度（帧数）
        stride: 步长

    Returns:
        windows: [M, window_size, 41]
        window_labels: [M] 窗口级标签（0=正常, 1=异常）
        window_phases: [M] 窗口级阶段标签
    """
    n_frames, n_features = features.shape
    windows = []
    window_labels = []
    window_phases = []

    for start in range(0, n_frames - window_size + 1, stride):
        end = start + window_size
        win_features = features[start:end]
        win_labels = labels[start:end]
        win_phases = phase_labels[start:end]

        # 纯净窗口筛选
        anomaly_ratio = win_labels.mean()
        if anomaly_ratio == 0.0:
            # 纯净正常窗口
            windows.append(win_features)
            window_labels.append(0)
            phase_val = int(np.bincount(win_phases.astype(int)).argmax()) if len(win_phases) > 0 else 0
            window_phases.append(phase_val)
        elif anomaly_ratio == 1.0:
            # 纯净异常窗口
            windows.append(win_features)
            window_labels.append(1)
            phase_val = int(np.bincount(win_phases.astype(int)).argmax()) if len(win_phases) > 0 else 0
            window_phases.append(phase_val)
        else:
            # 混合窗口，丢弃
            pass

    if len(windows) == 0:
        return np.empty((0, window_size, n_features)), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    return np.array(windows), np.array(window_labels, dtype=np.int32), np.array(window_phases, dtype=np.int32)


def load_and_process_case(
    case_id: int,
    csv_dir: str,
    window_size: int,
    stride: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    加载单个案例 CSV 并切分为窗口

    Returns:
        (windows [M, W, 41], labels [M], phases [M]) 或 None（如果无有效窗口）
    """
    case_file = os.path.join(csv_dir, f'Case_{case_id}.csv')
    if not os.path.exists(case_file):
        return None

    try:
        df = pd.read_csv(case_file)
    except Exception as e:
        print(f"[WARN] 读取失败 Case_{case_id}: {e}")
        return None

    if len(df) < window_size:
        return None

    # 提取特征
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(available_cols) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_cols)
        print(f"[WARN] Case_{case_id} 缺少列: {missing}")

    features = df[available_cols].values.astype(np.float32)

    # 缺失列用 0 填充
    if len(available_cols) < len(FEATURE_COLS):
        full_features = np.zeros((len(df), len(FEATURE_COLS)), dtype=np.float32)
        for i, col in enumerate(available_cols):
            full_features[:, FEATURE_COLS.index(col)] = features[:, i]
        features = full_features

    # 提取标签
    data_type, flight_mode, _, _ = parse_case_id(case_id)
    labels = extract_fault_labels(df)
    phase_labels = extract_phase_labels(case_id, len(df))

    # 创建窗口
    windows, window_labels, window_phases = create_windows(
        features, labels, phase_labels, window_size, stride
    )

    if len(windows) == 0:
        return None

    return windows, window_labels, window_phases


def compute_zscore_params(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Z-Score 参数（按特征维度）

    Args:
        windows: [N, W, 41]

    Returns:
        mean: [41], std: [41]
    """
    # 展平所有窗口的所有时间步
    flattened = windows.reshape(-1, windows.shape[-1])  # [N*W, 41]

    # 使用 nanmean/nanstd 处理缺失值
    mean = np.nanmean(flattened, axis=0)
    std = np.nanstd(flattened, axis=0)

    # 处理全 NaN 列
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)

    # 避免除零
    std[std == 0] = 1.0

    return mean, std


def apply_zscore(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """应用 Z-Score 标准化"""
    return (windows - mean) / std


def preprocess_split(
    split_name: str,
    case_ids: List[int],
    csv_dir: str,
    output_dir: str,
    window_size: int = 128,
    stride: int = 64,
    zscore_mean: Optional[np.ndarray] = None,
    zscore_std: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    处理单个集合

    Returns:
        如果 split_name == 'train'，返回 (mean, std)
        否则返回 None
    """
    print(f"\n[INFO] 处理 {split_name}: {len(case_ids)} 个案例")

    all_windows = []
    all_labels = []
    all_phases = []
    all_case_ids = []

    for case_id in tqdm(case_ids, desc=f"Processing {split_name}"):
        result = load_and_process_case(case_id, csv_dir, window_size, stride)
        if result is None:
            continue

        windows, labels, phases = result
        all_windows.append(windows)
        all_labels.append(labels)
        all_phases.append(phases)
        all_case_ids.extend([case_id] * len(windows))

    if len(all_windows) == 0:
        print(f"[WARN] {split_name} 无有效窗口")
        return None

    # 合并
    windows = np.concatenate(all_windows, axis=0)  # [N, W, 41]
    labels = np.concatenate(all_labels, axis=0)    # [N]
    phases = np.concatenate(all_phases, axis=0)    # [N]
    case_ids_arr = np.array(all_case_ids, dtype=np.int64)

    print(f"[INFO] {split_name}: {len(windows)} 个窗口")
    print(f"  正常窗口: {(labels == 0).sum()}")
    print(f"  异常窗口: {(labels == 1).sum()}")

    # Z-Score
    if split_name == 'train':
        mean, std = compute_zscore_params(windows)
        # 保存 Z-Score 参数
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, 'zscore_params.npz'), mean=mean, std=std)
        print(f"[INFO] Z-Score 参数已保存: mean shape={mean.shape}, std shape={std.shape}")
    else:
        assert zscore_mean is not None and zscore_std is not None, "非训练集需要提供 Z-Score 参数"
        mean, std = zscore_mean, zscore_std

    windows_norm = apply_zscore(windows, mean, std)

    # 保存为 .npz
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{split_name}.npz')
    np.savez(
        output_path,
        temporal=windows_norm.astype(np.float32),
        anomaly_labels=labels.astype(np.int32),
        phase_labels=phases.astype(np.int32),
        case_ids=case_ids_arr,
    )
    print(f"[INFO] 已保存: {output_path}")

    if split_name == 'train':
        return mean, std
    return None


def main(args):
    """主流程"""
    # 加载划分元数据
    with open(args.split_meta, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    splits = meta['case_ids']

    # 处理训练集（计算 Z-Score 参数）
    zscore_result = preprocess_split(
        'train', splits['train'], args.csv_dir, args.output_dir,
        args.window_size, args.stride
    )

    if zscore_result is None:
        print("[ERROR] 训练集处理失败")
        return

    mean, std = zscore_result

    # 处理其他集合
    for split_name in ['val', 'test_closed', 'test_open']:
        preprocess_split(
            split_name, splits[split_name], args.csv_dir, args.output_dir,
            args.window_size, args.stride, mean, std
        )

    print("\n[INFO] 全部处理完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument('--csv_dir', type=str, default='./Data_processing_tools/ProcessData',
                        help='CSV 文件目录')
    parser.add_argument('--split_meta', type=str, default='./data/preprocessed/split_metadata.json',
                        help='划分元数据 JSON')
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed',
                        help='输出目录')
    parser.add_argument('--window_size', type=int, default=128,
                        help='窗口长度（帧数）')
    parser.add_argument('--stride', type=int, default=64,
                        help='窗口步长')
    args = parser.parse_args()

    main(args)
