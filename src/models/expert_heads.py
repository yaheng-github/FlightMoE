"""Expert anomaly score heads for FlightMoE v2.

Each expert head maps a feature representation to a scalar anomaly score.
"""

import torch
import torch.nn as nn


class ExpertHead(nn.Module):
    """Generic expert head: feature vector -> anomaly score."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            score: [B]
        """
        return self.net(x).squeeze(-1)


class BurstHead(ExpertHead):
    """Detects sudden bursts / transient instabilities."""
    pass


class DriftHead(ExpertHead):
    """Detects slow drifts / statistical distribution shifts."""
    pass


class SpectralHead(ExpertHead):
    """Detects spectral texture / vibration pattern anomalies."""
    pass


class ConsistencyHead(ExpertHead):
    """Detects cross-sensor consistency violations."""
    pass


class ExpertPool(nn.Module):
    """Container for all 4 expert heads.

    Input:
        burst_feat, drift_feat, spectral_feat, consistency_feat: each [B, input_dim]
    Output:
        scores: [B, 4]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.burst = BurstHead(input_dim, hidden_dim, dropout)
        self.drift = DriftHead(input_dim, hidden_dim, dropout)
        self.spectral = SpectralHead(input_dim, hidden_dim, dropout)
        self.consistency = ConsistencyHead(input_dim, hidden_dim, dropout)

    def forward(
        self,
        burst_feat: torch.Tensor,
        drift_feat: torch.Tensor,
        spectral_feat: torch.Tensor,
        consistency_feat: torch.Tensor,
    ) -> torch.Tensor:
        s1 = self.burst(burst_feat)
        s2 = self.drift(drift_feat)
        s3 = self.spectral(spectral_feat)
        s4 = self.consistency(consistency_feat)
        return torch.stack([s1, s2, s3, s4], dim=1)  # [B, 4]


if __name__ == "__main__":
    B, D = 4, 256
    x = torch.randn(B, D)
    pool = ExpertPool(D)
    scores = pool(x, x, x, x)
    print("Expert scores shape:", scores.shape)  # [B, 4]
    assert scores.shape == (B, 4)
    print("Expert heads OK")
