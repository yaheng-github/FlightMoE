"""
STFT 时频图像生成模块

将 4 组传感域的时序数据转换为 RGB 时频图像：
- Control: control[0] (R), control[1] (G), control[2] (B)
- Gyro: gyro_rad[0] (R), gyro_rad[1] (G), gyro_rad[2] (B)
- Accel: accelerometer_m_s2[0] (R), accelerometer_m_s2[1] (G), accelerometer_m_s2[2] (B)
- Magnetometer: magnetometer_ga[0] (R), magnetometer_ga[1] (G), magnetometer_ga[2] (B)

每张图像：[H, W, 3]，其中 H=频率轴, W=时间轴
"""

import os
import argparse
from typing import Dict, List, Tuple
import numpy as np
from scipy import signal
from tqdm import tqdm


# 4 组传感域的通道映射
SPECTRAL_GROUPS = {
    'control': {
        'channels': [
            '_actuator_controls_0_0_control[0]',
            '_actuator_controls_0_0_control[1]',
            '_actuator_controls_0_0_control[2]',
        ],
        'rgb_names': ['Roll', 'Pitch', 'Yaw'],
    },
    'gyro': {
        'channels': [
            '_sensor_combined_0_gyro_rad[0]',
            '_sensor_combined_0_gyro_rad[1]',
            '_sensor_combined_0_gyro_rad[2]',
        ],
        'rgb_names': ['X', 'Y', 'Z'],
    },
    'accel': {
        'channels': [
            '_sensor_combined_0_accelerometer_m_s2[0]',
            '_sensor_combined_0_accelerometer_m_s2[1]',
            '_sensor_combined_0_accelerometer_m_s2[2]',
        ],
        'rgb_names': ['X', 'Y', 'Z'],
    },
    'magnetometer': {
        'channels': [
            '_vehicle_magnetometer_0_magnetometer_ga[0]',
            '_vehicle_magnetometer_0_magnetometer_ga[1]',
            '_vehicle_magnetometer_0_magnetometer_ga[2]',
        ],
        'rgb_names': ['X', 'Y', 'Z'],
    },
}

# 41 维特征字段的完整列表（与 preprocess.py 一致）
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


def compute_stft(
    time_series: np.ndarray,
    fs: float = 20.0,
    nperseg: int = 32,
    noverlap: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算单通道 STFT

    Args:
        time_series: [window_length] 单通道时序
        fs: 采样频率 (Hz)
        nperseg: 每段 FFT 长度
        noverlap: 段间重叠长度

    Returns:
        f: 频率轴
        t: 时间轴
        Zxx: STFT 复数结果 [n_freq, n_time]
    """
    f, t, Zxx = signal.stft(
        time_series,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary='zeros',
    )
    return f, t, Zxx


def stft_to_spectrogram(Zxx: np.ndarray) -> np.ndarray:
    """STFT 复数结果 -> 幅值谱，取对数压缩动态范围"""
    magnitude = np.abs(Zxx)
    # 加小量避免 log(0)
    log_magnitude = np.log(magnitude + 1e-8)
    return log_magnitude


def normalize_to_uint8(spectrogram: np.ndarray) -> np.ndarray:
    """将谱图归一化到 [0, 255] 范围"""
    min_val = spectrogram.min()
    max_val = spectrogram.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(spectrogram, dtype=np.uint8)
    normalized = (spectrogram - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def generate_spectral_image(
    window_data: np.ndarray,
    group_name: str,
    fs: float = 20.0,
    nperseg: int = 32,
    noverlap: int = 16,
) -> np.ndarray:
    """
    为单个窗口生成一张 RGB 时频图像

    Args:
        window_data: [128, 41] 单个窗口的 41 维时序
        group_name: 'control' / 'gyro' / 'accel' / 'magnetometer'
        fs: 采样频率
        nperseg: FFT 段长
        noverlap: 重叠长度

    Returns:
        rgb_image: [H, W, 3] uint8
    """
    group_config = SPECTRAL_GROUPS[group_name]
    channel_names = group_config['channels']

    # 获取通道索引
    indices = [FEATURE_COLS.index(name) for name in channel_names]

    rgb_channels = []
    for idx in indices:
        channel_data = window_data[:, idx]
        _, _, Zxx = compute_stft(channel_data, fs, nperseg, noverlap)
        spectrogram = stft_to_spectrogram(Zxx)
        uint8_spec = normalize_to_uint8(spectrogram)
        rgb_channels.append(uint8_spec)

    # 堆叠为 RGB: [H, W, 3]
    rgb_image = np.stack(rgb_channels, axis=-1)
    return rgb_image


def generate_all_spectral_images(
    temporal_data: np.ndarray,
    fs: float = 20.0,
    nperseg: int = 32,
    noverlap: int = 16,
) -> np.ndarray:
    """
    为一批窗口生成 4 张 RGB 时频图像

    Args:
        temporal_data: [N, 128, 41] 时序窗口
        fs, nperseg, noverlap: STFT 参数

    Returns:
        spectral_data: [N, 4, H, W, 3] uint8
            4 = control, gyro, accel, magnetometer
            H = n_freq, W = n_time
    """
    N, window_len, n_features = temporal_data.shape
    group_names = list(SPECTRAL_GROUPS.keys())
    n_groups = len(group_names)

    # 先用一个样本确定输出尺寸
    sample_img = generate_spectral_image(
        temporal_data[0], group_names[0], fs, nperseg, noverlap
    )
    H, W, _ = sample_img.shape

    spectral_data = np.zeros((N, n_groups, H, W, 3), dtype=np.uint8)

    for i in tqdm(range(N), desc='Generating STFT images'):
        for g, group_name in enumerate(group_names):
            spectral_data[i, g] = generate_spectral_image(
                temporal_data[i], group_name, fs, nperseg, noverlap
            )

    return spectral_data


def process_split(
    split_name: str,
    input_dir: str,
    output_dir: str,
    fs: float = 20.0,
    nperseg: int = 32,
    noverlap: int = 16,
):
    """处理单个集合，生成 spectral 数据并保存"""
    input_path = os.path.join(input_dir, f'{split_name}.npz')
    if not os.path.exists(input_path):
        print(f"[WARN] {input_path} 不存在，跳过")
        return

    print(f"\n[INFO] 处理 {split_name} ...")
    data = np.load(input_path)
    temporal = data['temporal']
    anomaly_labels = data['anomaly_labels']
    phase_labels = data['phase_labels']
    case_ids = data['case_ids']

    print(f"  输入 temporal shape: {temporal.shape}")

    # 生成 STFT 图像
    spectral = generate_all_spectral_images(
        temporal, fs=fs, nperseg=nperseg, noverlap=noverlap
    )

    print(f"  输出 spectral shape: {spectral.shape}")

    # 保存（覆盖原文件，加入 spectral）
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{split_name}.npz')
    np.savez(
        output_path,
        temporal=temporal,
        spectral=spectral,
        anomaly_labels=anomaly_labels,
        phase_labels=phase_labels,
        case_ids=case_ids,
    )
    print(f"[INFO] 已保存: {output_path}")


def main(args):
    """主流程"""
    for split_name in ['train', 'val', 'test_closed', 'test_open']:
        process_split(
            split_name,
            args.input_dir,
            args.output_dir,
            fs=args.fs,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )

    print("\n[INFO] STFT 图像生成全部完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STFT 时频图像生成')
    parser.add_argument('--input_dir', type=str, default='./data/preprocessed',
                        help='输入 .npz 目录')
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed',
                        help='输出目录')
    parser.add_argument('--fs', type=float, default=20.0,
                        help='采样频率 (Hz)')
    parser.add_argument('--nperseg', type=int, default=32,
                        help='FFT 段长')
    parser.add_argument('--noverlap', type=int, default=16,
                        help='段间重叠')
    args = parser.parse_args()

    main(args)
