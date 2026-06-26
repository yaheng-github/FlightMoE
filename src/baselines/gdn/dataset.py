"""
GDN 数据集适配：直接从 npz 加载，窗口内滑动
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class GDNTimeDataset(Dataset):
    def __init__(self, temporal, window_labels, edge_index, mode='train', slide_win=32, slide_stride=5):
        """
        Args:
            temporal: [N, T, D] 窗口化时序数据，T=128, D=41
            window_labels: [N] 窗口级标签（0=正常，1=异常）
            edge_index: [2, E] 图边索引（兼容 GDN 原始格式）
            mode: 'train' / 'val' / 'test'
            slide_win: 滑动窗口长度
            slide_stride: 训练时滑动步长（val/test 时固定为只取最后一步）
        """
        self.temporal = temporal
        self.window_labels = window_labels
        self.edge_index = edge_index
        self.mode = mode
        self.slide_win = slide_win
        self.slide_stride = slide_stride

        self.x, self.y, self.labels = self.process()

    def process(self):
        x_arr, y_arr, label_arr = [], [], []
        N, T, D = self.temporal.shape

        for i in range(N):
            if self.mode == 'train':
                rang = range(self.slide_win, T, self.slide_stride)
            else:
                rang = [T - 1]

            for t in rang:
                ft = self.temporal[i, t - self.slide_win:t, :].T
                tar = self.temporal[i, t, :]
                x_arr.append(ft)
                y_arr.append(tar)
                label_arr.append(self.window_labels[i])

        x = torch.stack([torch.tensor(x, dtype=torch.float32) for x in x_arr])
        y = torch.stack([torch.tensor(y, dtype=torch.float32) for y in y_arr])
        labels = torch.tensor(label_arr, dtype=torch.float32)
        return x, y, labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.labels[idx], self.edge_index
