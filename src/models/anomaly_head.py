"""Final anomaly scoring head for FlightMoE v2.

Maps a fused feature representation to a scalar anomaly score.
"""

import torch
import torch.nn as nn


class AnomalyHead(nn.Module):
    """MLP head from fused feature to anomaly score.

    Input:  [B, input_dim]
    Output: [B]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


if __name__ == "__main__":
    B, D = 4, 256
    x = torch.randn(B, D)
    head = AnomalyHead(D)
    score = head(x)
    print("Anomaly score shape:", score.shape)
    assert score.shape == (B,)
    print("Anomaly head OK")
