"""
MAD-GAN 数据加载适配：从 RflyMAD 预处理后的 npz 加载
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "third_party" / "madgan-pytorch"))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NPZDataset(Dataset):
    """从 npz 文件加载时序窗口数据

    Args:
        npz_path: 预处理后的数据路径
        normal_only: 是否只返回正常样本（训练 MAD-GAN 时必须为 True）
    """

    def __init__(self, npz_path: str, normal_only: bool = False):
        data = np.load(npz_path)
        self.temporal = data['temporal'].astype(np.float32)  # [N, T, F]

        if 'anomaly_labels' in data:
            self.labels = data['anomaly_labels']
        else:
            self.labels = np.zeros(len(self.temporal), dtype=np.int64)

        if normal_only:
            mask = self.labels == 0
            self.temporal = self.temporal[mask]
            self.labels = self.labels[mask]
            print(f"[Dataset] 过滤后正常样本: {len(self.temporal)}")
        else:
            print(f"[Dataset] 总样本: {len(self.temporal)}, 正常: {(self.labels==0).sum()}, 异常: {(self.labels==1).sum()}")

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.temporal[index])

    def __len__(self) -> int:
        return self.temporal.shape[0]


def get_dataloader(
    npz_path: str,
    batch_size: int,
    normal_only: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
):
    ds = NPZDataset(npz_path, normal_only=normal_only)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
