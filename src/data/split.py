"""
数据集划分模块：按故障类型分层划分 train/val/test_closed/test_open

关键约束：
1. 每个具体案例 (CaseID) 只进入一个集合
2. 共享故障类型按 CaseID 拆分；同 (data_type, flight_mode, fault_type) 可能跨集合但具体案例不跨集合
3. no_fault 按比例分配到所有集合
4. 共享故障类型（如 motor 同时属于 val 和 test_closed）按案例数比例拆分
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np


# 故障类型 ID 映射（与 get_parse.py 一致，0-based）
FAULT_TYPE_ID_MAP = {
    'motor': 0, 'propeller': 1, 'low_voltage': 2, 'wind_affect': 3,
    'load_lose': 4, 'accelerometer': 5, 'gyroscope': 6,
    'magnetometer': 7, 'barometer': 8, 'GPS': 9, 'no_fault': 10
}

FAULT_ID_TYPE_MAP = {v: k for k, v in FAULT_TYPE_ID_MAP.items()}

# 飞行状态 ID 映射（0-based）
FLIGHT_MODE_ID_MAP = {
    'hover': 0, 'waypoint': 1, 'velocity': 2,
    'circling': 3, 'acce': 4, 'dece': 5
}

# 默认划分策略：每种故障类型归属哪些集合
DEFAULT_FAULT_SPLIT = {
    'train': [],
    'val': ['motor', 'propeller'],
    'test_closed': ['motor', 'propeller', 'low_voltage', 'wind_affect', 'load_lose'],
    'test_open': ['accelerometer', 'gyroscope', 'magnetometer', 'barometer', 'GPS'],
}

# no_fault 样本分配比例
NO_FAULT_SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.10,
    'test_closed': 0.10,
    'test_open': 0.10,
}

# 共享故障类型的案例拆分比例（val : test_closed）
SHARED_FAULT_RATIOS = {
    'val': 0.3,
    'test_closed': 0.7,
}


def parse_case_id(case_id: int) -> Tuple[int, int, int, int]:
    """从 CaseID 解析数据类型、飞行状态、故障类型、案例编号"""
    data_type = case_id // 1000000000
    flight_mode = (case_id % 1000000000) // 100000000
    fault_type = (case_id % 100000000) // 1000000
    case_num = case_id % 1000000
    return data_type, flight_mode, fault_type, case_num


def get_fault_type_name(fault_id: int) -> str:
    """根据故障 ID 获取故障类型名称"""
    return FAULT_ID_TYPE_MAP.get(fault_id, 'unknown')


def split_dataset(
    csv_dir: str,
    fault_split: Optional[Dict[str, List[str]]] = None,
    no_fault_ratios: Optional[Dict[str, float]] = None,
    shared_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    按故障类型分层划分数据集

    Args:
        csv_dir: ProcessData 目录路径
        fault_split: 异常类型划分策略
        no_fault_ratios: no_fault 分配比例
        shared_ratios: 共享故障类型拆分比例
        seed: 随机种子

    Returns:
        {'train': [case_id, ...], 'val': [...], 'test_closed': [...], 'test_open': [...]}
    """
    np.random.seed(seed)
    fault_split = fault_split or DEFAULT_FAULT_SPLIT
    no_fault_ratios = no_fault_ratios or NO_FAULT_SPLIT_RATIOS
    shared_ratios = shared_ratios or SHARED_FAULT_RATIOS

    # 收集所有 CSV 文件
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

    # 按 (data_type, flight_mode, fault_type) 分组
    groups = {}
    for f in csv_files:
        case_id = int(f.replace('Case_', '').replace('.csv', ''))
        data_type, flight_mode, fault_type, case_num = parse_case_id(case_id)
        fault_name = get_fault_type_name(fault_type)

        key = (data_type, flight_mode, fault_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(case_id)

    splits = {'train': [], 'val': [], 'test_closed': [], 'test_open': []}

    # Step 1: 分配 no_fault 样本（按飞行阶段分别分层）
    no_fault_keys = [k for k in groups.keys() if k[2] == 'no_fault']

    # 先按 flight_mode 分组
    no_fault_by_flight_mode = {}
    for key in no_fault_keys:
        data_type, flight_mode, _ = key
        if flight_mode not in no_fault_by_flight_mode:
            no_fault_by_flight_mode[flight_mode] = []
        no_fault_by_flight_mode[flight_mode].extend(groups[key])

    # 对每个飞行阶段分别按比例分配，保证各集合都覆盖所有飞行阶段
    for flight_mode, case_ids in no_fault_by_flight_mode.items():
        case_ids = np.array(sorted(case_ids))
        np.random.shuffle(case_ids)

        n_total = len(case_ids)
        n_train = int(n_total * no_fault_ratios['train'])
        n_val = int(n_total * no_fault_ratios['val'])
        n_test_closed = int(n_total * no_fault_ratios['test_closed'])

        idx = 0
        splits['train'].extend(case_ids[idx:idx + n_train].tolist())
        idx += n_train
        splits['val'].extend(case_ids[idx:idx + n_val].tolist())
        idx += n_val
        splits['test_closed'].extend(case_ids[idx:idx + n_test_closed].tolist())
        idx += n_test_closed
        splits['test_open'].extend(case_ids[idx:].tolist())

    # Step 2: 分配异常样本
    # 先确定哪些故障类型是共享的（出现在多个集合中）
    fault_type_to_splits = {}
    for split_name, fault_names in fault_split.items():
        for fault_name in fault_names:
            if fault_name not in fault_type_to_splits:
                fault_type_to_splits[fault_name] = []
            fault_type_to_splits[fault_name].append(split_name)

    # 按组处理异常样本
    for key, case_ids in groups.items():
        data_type, flight_mode, fault_name = key
        if fault_name == 'no_fault':
            continue  # 已处理

        case_ids = np.array(sorted(case_ids))
        np.random.shuffle(case_ids)

        target_splits = fault_type_to_splits.get(fault_name, [])
        if len(target_splits) == 0:
            print(f"[WARN] 故障类型 '{fault_name}' 未被任何集合包含，跳过")
            continue
        elif len(target_splits) == 1:
            # 只归属一个集合，全部分配
            splits[target_splits[0]].extend(case_ids.tolist())
        else:
            # 共享故障类型，按比例拆分
            n_cases = len(case_ids)
            # 按 shared_ratios 中 target_splits 的比例分配
            ratios = [shared_ratios.get(s, 0.0) for s in target_splits]
            total_ratio = sum(ratios)
            if total_ratio == 0:
                ratios = [1.0 / len(target_splits)] * len(target_splits)
            else:
                ratios = [r / total_ratio for r in ratios]

            idx = 0
            for split_name, ratio in zip(target_splits, ratios):
                n = int(n_cases * ratio)
                splits[split_name].extend(case_ids[idx:idx + n].tolist())
                idx += n
            # 剩余案例给最后一个集合
            if idx < n_cases:
                splits[target_splits[-1]].extend(case_ids[idx:].tolist())

    # 打印统计
    print("=" * 70)
    print("数据集划分结果")
    print("=" * 70)
    for split_name, case_ids in splits.items():
        print(f"{split_name:12s}: {len(case_ids):5d} 个案例")
    print("=" * 70)

    # 打印各类故障分布
    print("\n各集合故障类型分布:")
    print("-" * 70)
    for split_name, case_ids in splits.items():
        fault_counts = {}
        for cid in case_ids:
            _, _, ft, _ = parse_case_id(cid)
            name = get_fault_type_name(ft)
            fault_counts[name] = fault_counts.get(name, 0) + 1
        print(f"\n{split_name}:")
        for name, count in sorted(fault_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:15s}: {count:4d}")

    return splits


def save_split_metadata(
    splits: Dict[str, List[int]],
    output_dir: str,
    fault_split: Dict,
    no_fault_ratios: Dict,
    shared_ratios: Dict
):
    """保存划分元数据到 JSON"""
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        'fault_split': fault_split,
        'no_fault_ratios': no_fault_ratios,
        'shared_ratios': shared_ratios,
        'counts': {k: len(v) for k, v in splits.items()},
        'case_ids': {k: [int(cid) for cid in v] for k, v in splits.items()}
    }

    output_path = os.path.join(output_dir, 'split_metadata.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] 划分元数据已保存: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='划分数据集')
    parser.add_argument('--csv_dir', type=str, default='./Data_processing_tools/ProcessData',
                        help='CSV 文件目录')
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    splits = split_dataset(args.csv_dir, seed=args.seed)
    save_split_metadata(
        splits, args.output_dir,
        DEFAULT_FAULT_SPLIT, NO_FAULT_SPLIT_RATIOS, SHARED_FAULT_RATIOS
    )
