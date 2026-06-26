"""
FP-Growth 关联规则挖掘 + 分阶段物理一致性邻接矩阵生成（方案 B）

严格遵循论文思想 + 用户改进：
- 分阶段：6 种飞行阶段分别挖掘（论文思想）
- 绝对值离散化：mean -> L/N/H（25%/75% 分位数，论文思想）
- 差值离散化：diff_mean -> L/N/H（用户改进）
- 算法：FP-Growth（pyfpgrowth，C 实现）
- 置信度：0.99（论文的 99%）
- 支持度：按绝对计数（论文风格）

输出：
- data/adjacency/adjacency_{phase}.npy  (6 个矩阵)
- data/adjacency/rules_{phase}.json      (6 个规则列表)
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pyfpgrowth


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

PHASE_NAMES = {0: 'hover', 1: 'waypoint', 2: 'velocity', 3: 'circling', 4: 'acce', 5: 'dece'}


def inverse_zscore(temporal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return temporal * std + mean


def extract_features(window_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """提取 mean 和 diff_mean"""
    mean_vals = window_data.mean(axis=0)
    diffs = np.diff(window_data, axis=0)
    diff_mean_vals = diffs.mean(axis=0)
    return mean_vals, diff_mean_vals


def compute_phase_quantiles(
    all_mean: np.ndarray,
    all_diff_mean: np.ndarray,
    phase_labels: np.ndarray,
) -> Dict[int, Dict]:
    """按飞行阶段计算 25% 和 75% 分位数"""
    quantiles = {}
    for phase in range(6):
        mask = phase_labels == phase
        if mask.sum() == 0:
            mask = np.ones(len(phase_labels), dtype=bool)
        quantiles[phase] = {
            'mean_q25': np.percentile(all_mean[mask], 25, axis=0),
            'mean_q75': np.percentile(all_mean[mask], 75, axis=0),
            'diff_q25': np.percentile(all_diff_mean[mask], 25, axis=0),
            'diff_q75': np.percentile(all_diff_mean[mask], 75, axis=0),
        }
    return quantiles


def discretize_window(
    mean_vals: np.ndarray,
    diff_mean_vals: np.ndarray,
    q: Dict,
) -> List[str]:
    """L/N/H 三元离散化"""
    items = []
    for i in range(41):
        if mean_vals[i] < q['mean_q25'][i]:
            items.append(f'sensor_{i:02d}_mean_L')
        elif mean_vals[i] > q['mean_q75'][i]:
            items.append(f'sensor_{i:02d}_mean_H')
        else:
            items.append(f'sensor_{i:02d}_mean_N')

        if diff_mean_vals[i] < q['diff_q25'][i]:
            items.append(f'sensor_{i:02d}_diff_L')
        elif diff_mean_vals[i] > q['diff_q75'][i]:
            items.append(f'sensor_{i:02d}_diff_H')
        else:
            items.append(f'sensor_{i:02d}_diff_N')
    return items


def parse_sensor_from_item(item: str) -> int:
    """从项名解析传感器索引"""
    return int(item.split('_')[1])


def build_transactions(
    phase_original: np.ndarray,
    q: Dict,
) -> Tuple[List[List[str]], List[List[str]]]:
    """分别构建 mean 和 diff 的事务集"""
    N = len(phase_original)
    mean_transactions = []
    diff_transactions = []
    for w in range(N):
        mean_vals, diff_mean_vals = extract_features(phase_original[w])
        mean_items = []
        diff_items = []
        for i in range(41):
            if mean_vals[i] < q['mean_q25'][i]:
                mean_items.append(f'sensor_{i:02d}_L')
            elif mean_vals[i] > q['mean_q75'][i]:
                mean_items.append(f'sensor_{i:02d}_H')
            else:
                mean_items.append(f'sensor_{i:02d}_N')

            if diff_mean_vals[i] < q['diff_q25'][i]:
                diff_items.append(f'sensor_{i:02d}_L')
            elif diff_mean_vals[i] > q['diff_q75'][i]:
                diff_items.append(f'sensor_{i:02d}_H')
            else:
                diff_items.append(f'sensor_{i:02d}_N')
        mean_transactions.append(mean_items)
        diff_transactions.append(diff_items)
    return mean_transactions, diff_transactions


def mine_rules(transactions: List[List[str]], min_support_count: int, min_confidence: float) -> Tuple[Dict, Dict]:
    """FP-Growth 挖掘频繁项集和关联规则"""
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support_count)
    if len(patterns) == 0:
        return {}, {}
    rules = pyfpgrowth.generate_association_rules(patterns, min_confidence)
    return patterns, rules


def build_adjacency_for_phase(
    phase_original: np.ndarray,
    q: Dict,
    min_support_count: int,
    min_confidence: float,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    为单个飞行阶段：分别对 mean 和 diff 做 FP-Growth，合并结果
    """
    N = len(phase_original)
    mean_transactions, diff_transactions = build_transactions(phase_original, q)

    print(f"  事务数: {N}, mean 项/事务: ~{len(mean_transactions[0])}, diff 项/事务: ~{len(diff_transactions[0])}")

    # 分别挖掘 mean 和 diff
    print(f"  FP-Growth 挖掘 mean (min_support={min_support_count}) ...")
    mean_patterns, mean_rules = mine_rules(mean_transactions, min_support_count, min_confidence)
    print(f"  mean 频繁项集: {len(mean_patterns)}, 规则: {len(mean_rules)}")

    print(f"  FP-Growth 挖掘 diff (min_support={min_support_count}) ...")
    diff_patterns, diff_rules = mine_rules(diff_transactions, min_support_count, min_confidence)
    print(f"  diff 频繁项集: {len(diff_patterns)}, 规则: {len(diff_rules)}")

    if len(mean_rules) == 0 and len(diff_rules) == 0:
        print("  [WARN] 未找到任何规则")
        return np.eye(41, dtype=np.float32), []

    # 构建邻接矩阵
    adj = np.zeros((41, 41), dtype=np.float32)
    rules_list = []

    def process_rules(rules, stat_type):
        for antecedent, (consequent, confidence) in rules.items():
            ante_sensors = {parse_sensor_from_item(it) for it in antecedent if it.startswith('sensor_')}
            cons_sensors = {parse_sensor_from_item(it) for it in consequent if it.startswith('sensor_')}
            for s_a in ante_sensors:
                for s_c in cons_sensors:
                    if s_a != s_c:
                        adj[s_a, s_c] = max(adj[s_a, s_c], confidence)
                        adj[s_c, s_a] = max(adj[s_c, s_a], confidence)
            rules_list.append({
                'type': stat_type,
                'antecedents': sorted(list(antecedent)),
                'consequents': sorted(list(consequent)),
                'confidence': round(float(confidence), 4),
            })

    if len(mean_rules) > 0:
        process_rules(mean_rules, 'mean')
    if len(diff_rules) > 0:
        process_rules(diff_rules, 'diff')

    np.fill_diagonal(adj, 1.0)
    return adj, rules_list


def main(args):
    print("=" * 70)
    print("FP-Growth 关联规则挖掘 + 分阶段物理一致性邻接矩阵")
    print("=" * 70)

    # 加载数据
    print(f"\n[INFO] 加载训练集: {args.train_npz}")
    train_data = np.load(args.train_npz)
    temporal = train_data['temporal']
    phase_labels = train_data['phase_labels']

    print(f"[INFO] 加载 Z-Score 参数: {args.zscore_npz}")
    zscore = np.load(args.zscore_npz)
    mean = zscore['mean']
    std = zscore['std']

    N = len(temporal)
    print(f"[INFO] 训练窗口数: {N}")

    # Step 1: 反标准化 + 统计特征
    print("\n[Step 1/3] 反标准化 + 统计特征提取 ...")
    original = inverse_zscore(temporal, mean, std)

    all_mean = original.mean(axis=1)
    all_diff = np.diff(original, axis=1).mean(axis=1)

    # Step 2: 按阶段计算分位数
    print("[Step 2/3] 按飞行阶段计算分位数 ...")
    quantiles = compute_phase_quantiles(all_mean, all_diff, phase_labels)

    # Step 3: 逐阶段 FP-Growth + 邻接矩阵
    print("[Step 3/3] 逐阶段 FP-Growth 挖掘 ...")
    os.makedirs(args.output_dir, exist_ok=True)

    for phase in range(6):
        mask = phase_labels == phase
        n_phase = mask.sum()
        phase_name = PHASE_NAMES[phase]
        print(f"\n--- Phase {phase} ({phase_name}): {n_phase} 个窗口 ---")

        if n_phase == 0:
            print("  [WARN] 无样本，跳过")
            continue

        # 支持度：按论文风格，用绝对计数
        # 例如 min_support=10 表示某项集至少出现在 10 个事务中
        adj, rules = build_adjacency_for_phase(
            original[mask],
            quantiles[phase],
            min_support_count=args.min_support_count,
            min_confidence=args.min_confidence,
        )

        # 保存
        adj_path = os.path.join(args.output_dir, f'adjacency_{phase_name}.npy')
        np.save(adj_path, adj)
        n_edges = (adj > 0).sum() - 41
        print(f"  邻接矩阵: {adj_path} | 有效边={n_edges} | 规则={len(rules)}")

        rules_path = os.path.join(args.output_dir, f'rules_{phase_name}.json')
        with open(rules_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)

    # 元数据
    meta = {
        'method': 'FP-Growth (pyfpgrowth) with L/N/H binning + differential binning',
        'features': ['mean', 'diff_mean'],
        'bins': ['L', 'N', 'H'],
        'quantiles': '25% / 75% per phase',
        'min_support_count': args.min_support_count,
        'min_confidence': args.min_confidence,
        'phases': PHASE_NAMES,
    }
    with open(os.path.join(args.output_dir, 'adjacency_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[DONE] 全部完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FP-Growth 物理一致性邻接矩阵')
    parser.add_argument('--train_npz', type=str, default='./data/preprocessed/train.npz')
    parser.add_argument('--zscore_npz', type=str, default='./data/preprocessed/zscore_params.npz')
    parser.add_argument('--output_dir', type=str, default='./data/adjacency')
    parser.add_argument('--min_support_count', type=int, default=10,
                        help='FP-Growth 最小支持度（绝对计数）')
    parser.add_argument('--min_confidence', type=float, default=0.99,
                        help='关联规则最小置信度')
    args = parser.parse_args()
    main(args)
