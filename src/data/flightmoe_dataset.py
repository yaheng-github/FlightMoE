"""Unified dataset loader for FlightMoE v2.

Loads temporal windows, spectral images, phase labels, anomaly labels, and case IDs
from preprocessed .npz files. Supports counterfactual perturbations during training.
"""

import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from models.perturbation import composite_perturbation
from utils.config import load_config


class FlightMoEDataset(Dataset):
    """PyTorch Dataset for FlightMoE v2.

    Returns a dict with keys:
        - temporal: [T, F] torch.float32
        - spectral: [G, F, T, C] torch.float32 (if available)
        - phase: int64
        - label: int32 (0=normal, 1=anomaly)
        - case_id: int64
        - modality_mask: [F] torch.float32 (1.0 = available, 0.0 = masked)
        - perturbation_type: str (only set when perturbation is applied)
    """

    def __init__(
        self,
        npz_path: str,
        mode: str = "train",
        perturbation_cfg: Optional[Dict] = None,
        apply_perturb_prob: float = 0.0,
        seed: int = 42,
    ):
        """
        Args:
            npz_path: path to preprocessed .npz file.
            mode: 'train', 'val', or 'test'.
            perturbation_cfg: config dict for perturbations.
            apply_perturb_prob: probability of applying a perturbation to each sample.
            seed: random seed.
        """
        super().__init__()
        self.npz_path = npz_path
        self.mode = mode
        self.perturbation_cfg = perturbation_cfg or {}
        self.apply_perturb_prob = apply_perturb_prob if mode == "train" else 0.0
        self.seed = seed

        data = np.load(npz_path)
        self.temporal = data["temporal"].astype(np.float32)
        self.spectral = data["spectral"].astype(np.float32) if "spectral" in data else None
        self.labels = data["anomaly_labels"].astype(np.int32)
        self.phase_labels = data["phase_labels"].astype(np.int64)
        self.case_ids = data["case_ids"].astype(np.int64)

        assert self.temporal.ndim == 3, f"temporal must be [N, T, F], got {self.temporal.shape}"
        if self.spectral is not None:
            assert self.spectral.ndim == 5, f"spectral must be [N, G, F, T, C], got {self.spectral.shape}"
            assert len(self.spectral) == len(self.temporal)

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.temporal)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        temporal = torch.from_numpy(self.temporal[idx].copy())
        spectral = (
            torch.from_numpy(self.spectral[idx].copy())
            if self.spectral is not None
            else None
        )
        phase = int(self.phase_labels[idx])
        label = int(self.labels[idx])
        case_id = int(self.case_ids[idx])

        modality_mask = torch.ones(temporal.size(-1), dtype=torch.float32)
        perturbation_type = "none"

        if self.apply_perturb_prob > 0 and random.random() < self.apply_perturb_prob:
            # Add batch dim for perturbation functions, then remove
            temporal = temporal.unsqueeze(0)
            spectral = spectral.unsqueeze(0) if spectral is not None else None
            temporal, spectral, modality_mask, perturbation_type = composite_perturbation(
                temporal, spectral, self.perturbation_cfg
            )
            temporal = temporal.squeeze(0)
            spectral = spectral.squeeze(0) if spectral is not None else None
            modality_mask = modality_mask.squeeze(0)

        sample = {
            "temporal": temporal,
            "phase": torch.tensor(phase, dtype=torch.int64),
            "label": torch.tensor(label, dtype=torch.int32),
            "case_id": torch.tensor(case_id, dtype=torch.int64),
            "modality_mask": modality_mask,
            "perturbation_type": perturbation_type,
        }
        if spectral is not None:
            sample["spectral"] = spectral
        return sample


def build_dataloader(
    npz_path: str,
    mode: str = "train",
    batch_size: int = 128,
    num_workers: int = 4,
    perturbation_cfg: Optional[Dict] = None,
    apply_perturb_prob: float = 0.0,
    seed: int = 42,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Convenience builder for FlightMoE DataLoader."""
    dataset = FlightMoEDataset(
        npz_path=npz_path,
        mode=mode,
        perturbation_cfg=perturbation_cfg,
        apply_perturb_prob=apply_perturb_prob,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == "train"),
    )


if __name__ == "__main__":
    cfg = load_config()
    data_cfg = cfg["data"]
    perturb_cfg = cfg["perturbation"]

    ds = FlightMoEDataset(
        npz_path=data_cfg["train_npz"],
        mode="train",
        perturbation_cfg=perturb_cfg,
        apply_perturb_prob=perturb_cfg.get("apply_prob", 0.5),
        seed=42,
    )
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: {v}")
