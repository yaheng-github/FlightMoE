"""Graph utilities for GDN graph-structure ablations."""
import numpy as np
import torch

PHASE_NAMES = ['hover', 'waypoint', 'velocity', 'circling', 'acce', 'dece']


def load_phase_edge_index(phase_name, threshold=0.0):
    """
    加载指定阶段的邻接矩阵并转为 edge_index (COO 格式)

    Args:
        phase_name: 阶段名，如 'hover'
        threshold: 阈值，低于此值的边被忽略（0=保留所有非零边）

    Returns:
        edge_index: LongTensor, shape [2, E]
    """
    adj = np.load(f'./data/adjacency/adjacency_{phase_name}.npy')
    rows, cols = np.where(adj > threshold)
    edge_index = np.stack([rows, cols], axis=0)
    return torch.tensor(edge_index, dtype=torch.long)


def adjacency_to_edge_index(adj, threshold=0.0):
    rows, cols = np.where(adj > threshold)
    edge_index = np.stack([rows, cols], axis=0)
    return torch.tensor(edge_index, dtype=torch.long)


def load_phase_adjacency(phase_name):
    return np.load(f'./data/adjacency/adjacency_{phase_name}.npy').astype(np.float32)


def build_global_adjacency():
    mats = [load_phase_adjacency(name) for name in PHASE_NAMES]
    adj = np.mean(np.stack(mats, axis=0), axis=0)
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)


def build_full_adjacency(node_num=41):
    return np.ones((node_num, node_num), dtype=np.float32)


def build_identity_adjacency(node_num=41):
    return np.eye(node_num, dtype=np.float32)


def build_random_adjacency(reference_adj, seed=0):
    rng = np.random.default_rng(seed)
    node_num = reference_adj.shape[0]
    off_mask = ~np.eye(node_num, dtype=bool)
    edge_count = int(((reference_adj > 0) & off_mask).sum())
    candidates = np.argwhere(off_mask)
    chosen = rng.choice(len(candidates), size=edge_count, replace=False)
    adj = np.eye(node_num, dtype=np.float32)
    rows, cols = candidates[chosen].T
    adj[rows, cols] = 1.0
    return adj


def get_graph_adjacency(graph_type, phase_name=None, threshold=0.0, seed=0):
    if graph_type == 'phase':
        if phase_name is None:
            raise ValueError("phase_name is required for phase graph")
        return load_phase_adjacency(phase_name)
    if graph_type == 'global':
        return build_global_adjacency()
    if graph_type == 'full':
        return build_full_adjacency()
    if graph_type == 'identity':
        return build_identity_adjacency()
    if graph_type == 'random':
        if phase_name is None:
            raise ValueError("phase_name is required for random graph density")
        return build_random_adjacency(load_phase_adjacency(phase_name), seed=seed)
    raise ValueError(f"Unknown graph_type: {graph_type}")


def get_graph_edge_index(graph_type, phase_name=None, threshold=0.0, seed=0):
    adj = get_graph_adjacency(graph_type, phase_name=phase_name, threshold=threshold, seed=seed)
    return adjacency_to_edge_index(adj, threshold=threshold)
