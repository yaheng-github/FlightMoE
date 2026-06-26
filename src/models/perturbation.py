"""Counterfactual perturbations for FlightMoE v2.

Implements 5 perturbation types designed to improve robustness:
1. sensor missing (modality dropout)
2. temporal async (random time shift per sensor group)
3. local noise (Gaussian / impulse noise)
4. spectral artifact (patch corruption on spectrograms)
5. domain shift (multiplicative scaling simulating sim-to-real drift)
"""

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# Sensor groups used for group-wise perturbations
SENSOR_GROUPS = {
    "control": [0, 1, 2],        # roll/pitch/yaw control commands
    "gyro": [3, 4, 5],           # x/y/z gyroscope
    "accel": [6, 7, 8],          # x/y/z accelerometer
    "magnetometer": [9, 10, 11], # x/y/z magnetometer
    "motor": [12, 13, 14, 15],   # motor 1-4
    "battery": [16, 17, 18],     # voltage/current/temperature
    "position": [19, 20, 21],    # x/y/z position
    "velocity": [22, 23, 24],    # x/y/z velocity
    "attitude": [25, 26, 27, 28],# attitude quaternion
    "angular_rate": [29, 30, 31],# angular velocity
    "altitude": [32, 33],        # barometer / altitude
    "gps": [34, 35, 36, 37],     # gps related
    "misc": [38, 39, 40],        # remaining fields
}


def _validate_temporal(x: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError(f"temporal input must be [B, T, F], got {x.shape}")


def _validate_spectral(s: torch.Tensor) -> None:
    if s.ndim != 5:
        raise ValueError(f"spectral input must be [B, G, F, T, C], got {s.shape}")


def perturb_missing(
    temporal: torch.Tensor,
    spectral: Optional[torch.Tensor],
    mask_ratio: float = 0.1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Randomly mask entire sensor channels.

    Args:
        temporal: [B, T, F]
        spectral: [B, G, F, T, C] or None
        mask_ratio: fraction of sensors to mask

    Returns:
        temporal, spectral, modality_mask [B, F]
    """
    _validate_temporal(temporal)
    B, T, F = temporal.shape
    mask = torch.ones(B, F, device=temporal.device, dtype=torch.float32)
    num_mask = max(1, int(F * mask_ratio))
    for b in range(B):
        idx = torch.randperm(F, device=temporal.device)[:num_mask]
        mask[b, idx] = 0.0
        temporal[b, :, idx] = 0.0

    if spectral is not None:
        _validate_spectral(spectral)
        # Map sensor group to spectral group roughly:
        # 0:control, 1:gyro, 2:accel, 3:magnetometer
        for b in range(B):
            masked = (mask[b] == 0).nonzero(as_tuple=True)[0]
            group_mask = torch.zeros(F, dtype=torch.bool)
            group_mask[masked] = True
            if torch.any(group_mask[:3]):
                spectral[b, 0] = 0.0
            if torch.any(group_mask[3:6]):
                spectral[b, 1] = 0.0
            if torch.any(group_mask[6:9]):
                spectral[b, 2] = 0.0
            if torch.any(group_mask[9:12]):
                spectral[b, 3] = 0.0

    return temporal, spectral, mask


def perturb_async(
    temporal: torch.Tensor,
    max_shift: int = 5,
) -> torch.Tensor:
    """Apply random time shift to sensor groups to simulate bus delay.

    Args:
        temporal: [B, T, F]
        max_shift: max shift in time steps

    Returns:
        temporal: [B, T, F]
    """
    _validate_temporal(temporal)
    B, T, F = temporal.shape
    out = temporal.clone()
    for b in range(B):
        for group_name, idxs in SENSOR_GROUPS.items():
            if random.random() < 0.3:  # 30% groups shifted per sample
                shift = random.randint(-max_shift, max_shift)
                if shift == 0:
                    continue
                out[b, :, idxs] = torch.roll(out[b, :, idxs], shifts=shift, dims=0)
                # Zero-pad rolled edges instead of wrap-around
                if shift > 0:
                    out[b, :shift, idxs] = 0.0
                else:
                    out[b, shift:, idxs] = 0.0
    return out


def perturb_noise(
    temporal: torch.Tensor,
    std: float = 0.1,
) -> torch.Tensor:
    """Add Gaussian noise scaled by per-sensor standard deviation.

    Args:
        temporal: [B, T, F]
        std: noise scale relative to signal std

    Returns:
        temporal: [B, T, F]
    """
    _validate_temporal(temporal)
    signal_std = temporal.std(dim=1, keepdim=True).clamp(min=1e-6)
    noise = torch.randn_like(temporal) * std * signal_std
    return temporal + noise


def perturb_spectral_artifact(
    spectral: torch.Tensor,
    patch_ratio: float = 0.1,
    intensity: float = 0.5,
) -> torch.Tensor:
    """Corrupt random patches of spectrograms.

    Args:
        spectral: [B, G, F, T, C]
        patch_ratio: fraction of patches to corrupt
        intensity: magnitude of corruption

    Returns:
        spectral: [B, G, F, T, C]
    """
    _validate_spectral(spectral)
    B, G, F, T, C = spectral.shape
    out = spectral.clone()
    num_patches = max(1, int(G * F * T * patch_ratio))
    for b in range(B):
        for _ in range(num_patches):
            g = random.randint(0, G - 1)
            f0 = random.randint(0, F - 1)
            t0 = random.randint(0, T - 1)
            out[b, g, f0, t0, :] += intensity * torch.randn(C, device=spectral.device)
    return out


def perturb_domain_shift(
    temporal: torch.Tensor,
    scale_range: Tuple[float, float] = (0.9, 1.1),
) -> torch.Tensor:
    """Apply per-sensor-group multiplicative scaling to simulate sim-to-real drift.

    Args:
        temporal: [B, T, F]
        scale_range: (min, max) multiplicative scale

    Returns:
        temporal: [B, T, F]
    """
    _validate_temporal(temporal)
    B, T, F = temporal.shape
    out = temporal.clone()
    for b in range(B):
        for idxs in SENSOR_GROUPS.values():
            scale = random.uniform(*scale_range)
            out[b, :, idxs] *= scale
    return out


def composite_perturbation(
    temporal: torch.Tensor,
    spectral: Optional[torch.Tensor],
    cfg: Dict,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, str]:
    """Apply one random perturbation according to config.

    Args:
        temporal: [B, T, F]
        spectral: [B, G, F, T, C] or None
        cfg: perturbation config dict

    Returns:
        temporal, spectral, modality_mask [B, F], perturbation_type
    """
    ptype = random.choice(["missing", "async", "noise", "spectral_artifact", "domain_shift"])
    modality_mask = torch.ones(temporal.size(0), temporal.size(2), device=temporal.device)

    if ptype == "missing":
        temporal, spectral, modality_mask = perturb_missing(
            temporal, spectral, mask_ratio=cfg.get("missing", {}).get("mask_ratio", 0.1)
        )
    elif ptype == "async":
        temporal = perturb_async(temporal, max_shift=cfg.get("async", {}).get("max_shift", 5))
    elif ptype == "noise":
        temporal = perturb_noise(temporal, std=cfg.get("noise", {}).get("std", 0.1))
    elif ptype == "spectral_artifact":
        if spectral is not None:
            spectral = perturb_spectral_artifact(
                spectral,
                patch_ratio=cfg.get("spectral_artifact", {}).get("patch_ratio", 0.1),
                intensity=cfg.get("spectral_artifact", {}).get("intensity", 0.5),
            )
    elif ptype == "domain_shift":
        temporal = perturb_domain_shift(
            temporal, scale_range=tuple(cfg.get("domain_shift", {}).get("scale_range", [0.9, 1.1]))
        )

    return temporal, spectral, modality_mask, ptype


if __name__ == "__main__":
    B, T, F = 4, 128, 41
    temporal = torch.randn(B, T, F)
    spectral = torch.randn(B, 4, 17, 9, 3)
    cfg = {
        "missing": {"mask_ratio": 0.1},
        "async": {"max_shift": 5},
        "noise": {"std": 0.1},
        "spectral_artifact": {"patch_ratio": 0.1, "intensity": 0.5},
        "domain_shift": {"scale_range": [0.9, 1.1]},
    }
    t_aug, s_aug, mask, ptype = composite_perturbation(temporal, spectral, cfg)
    print(f"Perturbation type: {ptype}")
    print(f"temporal shape: {t_aug.shape}, spectral shape: {s_aug.shape}, mask shape: {mask.shape}")
    print(f"masked sensors per sample: {(mask == 0).sum(dim=1)}")
