"""
STFT 时频图像可视化导出

从 .npz 中读取 spectral 数据，导出为可查看的 PNG 图片
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt


# 4 组传感域的名称和 RGB 通道含义
SPECTRAL_GROUPS = {
    'control': {'title': 'Control (R=Roll, G=Pitch, B=Yaw)'},
    'gyro': {'title': 'Gyro (R=X, G=Y, B=Z)'},
    'accel': {'title': 'Accel (R=X, G=Y, B=Z)'},
    'magnetometer': {'title': 'Magnetometer (R=X, G=Y, B=Z)'},
}
GROUP_NAMES = list(SPECTRAL_GROUPS.keys())


def visualize_single_window(
    spectral_window: np.ndarray,
    anomaly_label: int,
    phase_label: int,
    case_id: int,
    win_idx: int,
    output_dir: str,
):
    """
    可视化单个窗口的 4 张时频图

    Args:
        spectral_window: [4, H, W, 3] uint8
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for g, (group_name, ax) in enumerate(zip(GROUP_NAMES, axes)):
        img = spectral_window[g]  # [H, W, 3]
        ax.imshow(img, aspect='auto', interpolation='nearest')
        ax.set_title(SPECTRAL_GROUPS[group_name]['title'], fontsize=10)
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Freq bin')
        ax.tick_params(labelsize=8)

    # 解析 CaseID
    data_type = case_id // 1000000000
    flight_mode = (case_id % 1000000000) // 100000000
    fault_type = (case_id % 100000000) // 1000000
    case_num = case_id % 1000000

    status = 'Anomaly' if anomaly_label == 1 else 'Normal'
    fig.suptitle(
        f'CaseID={case_id} | {status} | Phase={phase_label} | '
        f'DataType={data_type}, FlightMode={flight_mode}, FaultType={fault_type}, CaseNum={case_num}',
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(output_dir, f'win_{win_idx:06d}_case_{case_id}_label_{anomaly_label}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def main(args):
    input_path = os.path.join(args.input_dir, f'{args.split}.npz')
    if not os.path.exists(input_path):
        print(f'[ERROR] 文件不存在: {input_path}')
        return

    print(f'[INFO] 加载 {input_path} ...')
    data = np.load(input_path)
    spectral = data['spectral']
    anomaly_labels = data['anomaly_labels']
    phase_labels = data['phase_labels']
    case_ids = data['case_ids']

    N = len(spectral)
    print(f'[INFO] 总样本数: {N}')

    # 决定采样哪些窗口
    if args.indices:
        indices = args.indices
    elif args.random_n > 0:
        indices = np.random.choice(N, min(args.random_n, N), replace=False)
        indices = sorted(indices.tolist())
    else:
        print('[ERROR] 请指定 --indices 或 --random_n')
        return

    output_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_dir, exist_ok=True)

    print(f'[INFO] 将导出 {len(indices)} 个样本到 {output_dir}')

    for i, idx in enumerate(indices):
        path = visualize_single_window(
            spectral[idx],
            anomaly_labels[idx],
            phase_labels[idx],
            case_ids[idx],
            idx,
            output_dir,
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{len(indices)}] {path}')

    print(f'[INFO] 完成! 共导出 {len(indices)} 张图片到 {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STFT 时频图像可视化导出')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test_closed', 'test_open'],
                        help='数据集划分')
    parser.add_argument('--input_dir', type=str, default='./data/preprocessed',
                        help='输入 .npz 目录')
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed/spectral_images',
                        help='输出图片目录')
    parser.add_argument('--random_n', type=int, default=20,
                        help='随机采样 N 个样本可视化')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='指定要可视化的样本索引，如 0 10 20')
    args = parser.parse_args()

    main(args)
