"""
GANomaly 数据加载适配：从 RflyMAD spectral 图像数据加载
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class SpectralDataset(Dataset):
    """从 npz 加载 spectral 图像数据 [N, 4, 17, 9, 3]

    把每张 [17, 9, 3] 小图 resize 到 [64, 64]，4 张拼成 1 个样本的 4 通道
    或者：4 张小图按 2x2 拼成 [64, 64, 3]
    """

    def __init__(self, npz_path: str, normal_only: bool = False, image_size: int = 64):
        data = np.load(npz_path)
        self.spectral = data['spectral'].astype(np.float32)  # [N, 4, 17, 9, 3]

        if 'anomaly_labels' in data:
            self.labels = data['anomaly_labels']
        else:
            self.labels = np.zeros(len(self.spectral), dtype=np.int64)

        if normal_only:
            mask = self.labels == 0
            self.spectral = self.spectral[mask]
            self.labels = self.labels[mask]

        self.image_size = image_size
        print(f"[Dataset] 样本: {len(self.spectral)}, 正常: {(self.labels==0).sum()}, 异常: {(self.labels==1).sum()}")

    def __getitem__(self, index: int):
        # [4, 17, 9, 3] -> 把 4 张按 2x2 拼成 [34, 18, 3]，再 resize 到 [64, 64, 3]
        imgs = self.spectral[index]  # [4, 17, 9, 3]

        # 每张 resize 到 [3, 32, 32]
        patches = []
        for i in range(4):
            patch = torch.from_numpy(imgs[i]).permute(2, 0, 1)  # [3, 17, 9]
            patch = F.interpolate(patch.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
            patches.append(patch.squeeze(0))  # [3, 32, 32]

        # 2x2 拼接: top-left, top-right, bottom-left, bottom-right
        top = torch.cat([patches[0], patches[1]], dim=2)      # [3, 32, 64]
        bottom = torch.cat([patches[2], patches[3]], dim=2)   # [3, 32, 64]
        img = torch.cat([top, bottom], dim=1)                  # [3, 64, 64]

        # 归一化到 [-1, 1]
        img = img / 255.0 * 2 - 1

        return img, self.labels[index]

    def __len__(self) -> int:
        return len(self.spectral)


def get_dataloader(
    npz_path: str,
    batch_size: int,
    normal_only: bool = False,
    shuffle: bool = True,
    image_size: int = 64,
    drop_last: bool = True,
):
    ds = SpectralDataset(npz_path, normal_only=normal_only, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
