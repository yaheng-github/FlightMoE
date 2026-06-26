"""
物理一致性邻接矩阵生成（NumPy 向量化实现，等价于 FP-Growth 2-项集）

严格遵循论文思想 + 用户改进：
- 分阶段：6 种飞行阶段分别计算（论文思想）
- 绝对值离散化：mean -> L/N/H（10%/90% 分位数，保留更极端状态）
- 差值离散化：diff_mean -> L/N/H（用户改进）
- 算法：NumPy 向量化 2-项集计数（数学上等价于 FP-Growth/Apriori 2-项集）
- 置信度：0.99（论文的 99%）

数学原理：
  X: [N, 41] bool 矩阵，X[i,j]=True 表示传感器 j 在事务 i 中处于某状态
  cooccur = X.T @ X      # [41, 41] 共现次数
  support = X.sum(axis=0) # [41] 支持度
  confidence = cooccur / support[:, None]  # [41, 41] 置信度矩阵

输出：
- data/adjacency/adjacency_{phase}.npy  (6 个矩阵)
- data/adjacency/rules_{phase}.json      (6 个规则列表)
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np


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

# 伪恒定字段索引：这些字段在数据集中变化极小，会产生虚假高置信关联
# 33=battery_temperature(全0), 37=GPS_satellites_used, 38=GPS_eph, 39=GPS_epv, 40=GPS_fix_type
PSEUDO_CONSTANT_INDICES = [33, 37, 38, 39, 40]


def inverse_zscore(temporal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return temporal * std + mean


def compute_phase_quantiles(
    all_mean: np.ndarray,
    all_diff_mean: np.ndarray,
    phase_labels: np.ndarray,
) -> Dict[int, Dict]:
    """按飞行阶段计算 10% 和 90% 分位数"""
    quantiles = {}
    for phase in range(6):
        mask = phase_labels == phase
        if mask.sum() == 0:
            mask = np.ones(len(phase_labels), dtype=bool)
        quantiles[phase] = {
            'mean_q10': np.percentile(all_mean[mask], 10, axis=0),
            'mean_q90': np.percentile(all_mean[mask], 90, axis=0),
            'diff_q10': np.percentile(all_diff_mean[mask], 10, axis=0),
            'diff_q90': np.percentile(all_diff_mean[mask], 90, axis=0),
        }
    return quantiles


def build_state_matrices(
    phase_original: np.ndarray,
    q: Dict,
) -> Dict[str, np.ndarray]:
    """
    为单个阶段构建 6 种状态矩阵

    Returns:
        {'mean_L': [N, 41], 'mean_H': [N, 41],
         'diff_L': [N, 41], 'diff_H': [N, 41]}
    """
    means = phase_original.mean(axis=1)           # [N, 41]
    diffs = np.diff(phase_original, axis=1).mean(axis=1)  # [N, 41]

    return {
        'mean_L': means < q['mean_q10'],
        'mean_H': means > q['mean_q90'],
        'diff_L': diffs < q['diff_q10'],
        'diff_H': diffs > q['diff_q90'],
    }


def mine_pairwise_rules(
    state_mat: np.ndarray,
    state_name: str,
    min_confidence: float,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    NumPy 向量化 2-项集挖掘

    Args:
        state_mat: [N, 41] bool 矩阵
        state_name: 如 'mean_H'
        min_confidence: 最小置信度阈值

    Returns:
        adj_contrib: [41, 41] 该状态对邻接矩阵的贡献
        rules: 规则列表
    """
    N, n_sensors = state_mat.shape

    # 核心：矩阵乘法统计共现
    cooccur = state_mat.astype(np.int32).T @ state_mat.astype(np.int32)  # [41, 41]
    support = state_mat.sum(axis=0)  # [41]

    # 避免除零
    support_safe = np.where(support == 0, 1, support)

    # 置信度矩阵：conf[i, j] = P(j | i) = cooccur[i,j] / support[i]
    conf_matrix = cooccur / support_safe[:, None]  # [41, 41]

    # 对称化：取 max(conf(i->j), conf(j->i))
    conf_sym = np.maximum(conf_matrix, conf_matrix.T)

    # 邻接矩阵：保留全部置信度（不过滤），对角线为 1.0
    adj_contrib = conf_sym.astype(np.float32)
    np.fill_diagonal(adj_contrib, 1.0)

    # 提取强规则（>= 阈值），用于论文展示和人工分析
    rules = []
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            if conf_sym[i, j] >= min_confidence:
                rules.append({
                    'type': state_name,
                    'sensor_i': FEATURE_COLS[i],
                    'sensor_j': FEATURE_COLS[j],
                    'confidence': round(float(conf_sym[i, j]), 4),
                    'cooccur': int(cooccur[i, j]),
                    'support_i': int(support[i]),
                    'support_j': int(support[j]),
                })

    return adj_contrib, rules


def build_adjacency_for_phase(
    phase_original: np.ndarray,
    q: Dict,
    min_confidence: float,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    为单个飞行阶段构建邻接矩阵
    对 6 种状态分别挖掘，然后合并取最大值
    """
    state_matrices = build_state_matrices(phase_original, q)

    adj = np.zeros((41, 41), dtype=np.float32)
    all_rules = []

    for state_name, state_mat in state_matrices.items():
        adj_contrib, rules = mine_pairwise_rules(state_mat, state_name, min_confidence)
        adj = np.maximum(adj, adj_contrib)
        all_rules.extend(rules)

    # 隔离伪恒定字段：只保留自耦合，与其他字段的关联强制清零
    for idx in PSEUDO_CONSTANT_INDICES:
        adj[idx, :] = 0.0
        adj[:, idx] = 0.0
        adj[idx, idx] = 1.0

    return adj, all_rules


def main(args):
    print("=" * 70)
    print("物理一致性邻接矩阵生成（NumPy 向量化 2-项集计数）")
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

    # Step 3: 逐阶段构建邻接矩阵
    print("[Step 3/3] 逐阶段 NumPy 向量化挖掘 ...")
    os.makedirs(args.output_dir, exist_ok=True)

    for phase in range(6):
        mask = phase_labels == phase
        n_phase = mask.sum()
        phase_name = PHASE_NAMES[phase]
        print(f"\n--- Phase {phase} ({phase_name}): {n_phase} 个窗口 ---")

        if n_phase == 0:
            print("  [WARN] 无样本，跳过")
            continue

        adj, rules = build_adjacency_for_phase(
            original[mask],
            quantiles[phase],
            min_confidence=args.min_confidence,
        )

        # 保存邻接矩阵
        adj_path = os.path.join(args.output_dir, f'adjacency_{phase_name}.npy')
        np.save(adj_path, adj)
        n_edges = (adj > 0).sum() - 41
        print(f"  邻接矩阵: {adj_path} | 有效边={n_edges} | 规则={len(rules)}")

        # 保存规则
        rules_path = os.path.join(args.output_dir, f'rules_{phase_name}.json')
        with open(rules_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)

    # 元数据
    meta = {
        'method': 'NumPy vectorized 2-itemset counting (equivalent to FP-Growth 2-itemset)',
        'features': ['mean', 'diff_mean'],
        'bins': ['L', 'H'],
        'quantiles': '10% / 90% per phase',
        'min_confidence': args.min_confidence,
        'phases': PHASE_NAMES,
    }
    with open(os.path.join(args.output_dir, 'adjacency_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[DONE] 全部完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='物理一致性邻接矩阵生成')
    parser.add_argument('--train_npz', type=str, default='./data/preprocessed/train.npz')
    parser.add_argument('--zscore_npz', type=str, default='./data/preprocessed/zscore_params.npz')
    parser.add_argument('--output_dir', type=str, default='./data/adjacency')
    parser.add_argument('--min_confidence', type=float, default=0.99,
                        help='关联规则最小置信度')
    args = parser.parse_args()
    main(args)
