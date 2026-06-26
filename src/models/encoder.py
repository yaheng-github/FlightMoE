"""Phase-aware multi-view encoder for FlightMoE v2.

Combines:
- 1D-CNN temporal branch over [B, T, F]
- 2D-CNN spectral branch over [B, G, F, T, C]
- Learnable phase embedding
- Cross-modal attention between temporal and spectral representations
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBranch(nn.Module):
    """1D-CNN encoder for temporal windows.

    Input:  [B, T, F]
    Output: [B, out_dim]
    """

    def __init__(self, input_dim: int = 41, hidden_dims=None, out_dim: int = 256,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, h, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        x = self.conv_layers(x)  # [B, hidden, T]
        x = self.global_pool(x).squeeze(-1)  # [B, hidden]
        x = self.project(x)  # [B, out_dim]
        return x


class SpectralBranch(nn.Module):
    """2D-CNN encoder for spectral images.

    Input:  [B, G, F, T, C]
    Output: [B, out_dim]
    """

    def __init__(self, num_groups: int = 4, input_channels: int = 3,
                 base_channels: int = 32, out_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.num_groups = num_groups
        self.input_channels = input_channels

        # Shared CNN per group
        self.group_cnn = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # Compute flattened size assuming input ~ [C, 17, 9]
        # After two MaxPool2d(2): 17->8->4, 9->4->2  => 4*2=8 spatial
        self.flat_size = (base_channels * 2) * 4 * 2

        self.group_fusion = nn.Sequential(
            nn.Linear(self.flat_size * num_groups, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, G, F, T, C]
        B, G, F, T, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, G, C, F, T]
        x = x.view(B * G, C, F, T)
        x = self.group_cnn(x)  # [B*G, C', F', T']
        x = x.view(B, G, -1)  # [B, G, flat_size]
        x = x.view(B, -1)  # [B, G * flat_size]
        x = self.group_fusion(x)  # [B, out_dim]
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention between temporal and spectral features.

    Uses temporal as query, spectral as key/value by default.
    Input:  query [B, Tq, Dq], kv [B, Tk, Dk]
    Output: [B, Tq, Dq]
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, kv, kv, need_weights=False)
        return self.norm(query + out)


class PhaseAwareMultiViewEncoder(nn.Module):
    """Main encoder combining temporal, spectral, and phase information.

    Input:
        temporal: [B, T, F]
        spectral: [B, G, F, T, C]
        phase:    [B]
    Output:
        [B, output_dim]
    """

    def __init__(
        self,
        temporal_dim: int = 41,
        temporal_embed_dim: int = 256,
        spectral_embed_dim: int = 256,
        phase_embed_dim: int = 64,
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,
        output_dim: int = 256,
        dropout: float = 0.1,
        use_phase: bool = True,
    ):
        super().__init__()
        self.use_phase = use_phase
        self.temporal_branch = TemporalBranch(
            input_dim=temporal_dim,
            out_dim=temporal_embed_dim,
            dropout=dropout,
        )
        self.spectral_branch = SpectralBranch(
            out_dim=spectral_embed_dim,
            dropout=dropout,
        )
        self.phase_embedding = nn.Embedding(6, phase_embed_dim)

        # Cross-modal attention: treat temporal as query, spectral as key/value
        # We project both to a common dim
        self.cross_attn = CrossModalAttention(
            d_model=temporal_embed_dim,
            num_heads=cross_attention_heads,
            dropout=cross_attention_dropout,
        )
        self.spectral_proj = nn.Linear(spectral_embed_dim, temporal_embed_dim)

        fusion_dim = temporal_embed_dim + (phase_embed_dim if use_phase else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )

    def forward(
        self,
        temporal: torch.Tensor,
        spectral: Optional[torch.Tensor],
        phase: torch.Tensor,
    ) -> torch.Tensor:
        # temporal: [B, T, F]
        # spectral: [B, G, F, T, C] or None
        # phase: [B]
        B = temporal.size(0)
        temp_feat = self.temporal_branch(temporal)  # [B, temporal_embed_dim]

        if spectral is not None:
            spec_feat = self.spectral_branch(spectral)  # [B, spectral_embed_dim]
            spec_feat_proj = self.spectral_proj(spec_feat).unsqueeze(1)  # [B, 1, temporal_embed_dim]
            temp_feat_seq = temp_feat.unsqueeze(1)  # [B, 1, temporal_embed_dim]
            fused = self.cross_attn(temp_feat_seq, spec_feat_proj)  # [B, 1, temporal_embed_dim]
            fused = fused.squeeze(1)  # [B, temporal_embed_dim]
        else:
            fused = temp_feat

        if self.use_phase:
            phase_feat = self.phase_embedding(phase)  # [B, phase_embed_dim]
            x = torch.cat([fused, phase_feat], dim=-1)  # [B, temporal_embed_dim + phase_embed_dim]
        else:
            x = fused

        x = self.fusion(x)  # [B, output_dim]
        return x


if __name__ == "__main__":
    B, T, F = 4, 128, 41
    temporal = torch.randn(B, T, F)
    spectral = torch.randn(B, 4, 17, 9, 3)
    phase = torch.randint(0, 6, (B,))

    encoder = PhaseAwareMultiViewEncoder()
    out = encoder(temporal, spectral, phase)
    print(f"Encoder output shape: {out.shape}")  # [B, 256]
    assert out.shape == (B, 256)
    print("Encoder OK")
