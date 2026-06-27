"""Physical consistency GNN for FlightMoE v2.

Projects encoder output to per-sensor features, applies a GAT-like layer using
phase-specific FP-Growth adjacency matrices, and computes a consistency residual.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))

from baselines.gdn.graph_layer_v2 import GraphLayer
from baselines.gdn.graph_struct import PHASE_NAMES, load_phase_edge_index


class PhysicalConsistencyGNN(nn.Module):
    """Physical consistency GNN.

    Input:
        features: [B, input_dim] from encoder
        phase:    [B] phase labels
    Output:
        calibrated: [B, node_num, gnn_dim]
        residual:   [B, node_num]
        edge_weights: [B, E] (attention weights averaged over heads)
    """

    def __init__(
        self,
        input_dim: int = 256,
        node_num: int = 41,
        gnn_dim: int = 128,
        num_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.1,
        learnable_edge_weight: bool = False,
    ):
        super().__init__()
        self.node_num = node_num
        self.gnn_dim = gnn_dim
        self.num_layers = num_layers
        self.learnable_edge_weight = learnable_edge_weight

        self.project = nn.Linear(input_dim, node_num * gnn_dim)

        self.gnn_layers = nn.ModuleList([
            GraphLayer(
                in_channels=gnn_dim,
                out_channels=gnn_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.residual_mlp = nn.Sequential(
            nn.Linear(gnn_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Cache edge_index per phase on first forward
        self._edge_index_cache = {}

        if learnable_edge_weight:
            # Create learnable edge weights per phase, lazy init on first forward
            self._edge_weight_params = nn.ParameterDict()
        else:
            self._edge_weight_params = None

    def _get_edge_index(self, phase: int, device: torch.device):
        if phase not in self._edge_index_cache:
            phase_name = PHASE_NAMES[phase]
            edge_index = load_phase_edge_index(phase_name, threshold=0.0)
            self._edge_index_cache[phase] = edge_index
        edge_index = self._edge_index_cache[phase].to(device)

        if self.learnable_edge_weight:
            key = str(phase)
            if key not in self._edge_weight_params:
                num_edges = edge_index.size(1)
                self._edge_weight_params[key] = nn.Parameter(
                    torch.ones(num_edges, device=device) * 0.5
                )
            edge_weight = torch.sigmoid(self._edge_weight_params[key])
            return edge_index, edge_weight
        return edge_index, None

    def forward(
        self,
        features: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, input_dim]
            phase: [B]

        Returns:
            calibrated: [B, node_num, gnn_dim]
            residual:   [B, node_num]
            edge_weights: [B, E]
        """
        B = features.size(0)
        device = features.device

        # Project to per-sensor features
        x = self.project(features)  # [B, node_num * gnn_dim]
        x = x.view(B, self.node_num, self.gnn_dim)  # [B, node_num, gnn_dim]
        x0 = x  # residual connection baseline

        # Process each sample independently (different graphs per phase)
        calibrated_list = []
        edge_weights_list = []
        for b in range(B):
            edge_index, edge_weight = self._get_edge_index(int(phase[b].item()), device)
            xb = x[b]  # [node_num, gnn_dim]
            for layer in self.gnn_layers:
                xb_out, (_, alpha) = layer(xb, edge_index, edge_weight=edge_weight, return_attention_weights=True)
                xb = xb + xb_out  # residual
                xb = F.relu(xb)

            calibrated_list.append(xb)

            # alpha: [E, heads] -> average over heads
            alpha_mean = alpha.mean(dim=-1)  # [E]
            if edge_weight is not None:
                alpha_mean = alpha_mean * edge_weight
            edge_weights_list.append(alpha_mean)

        calibrated = torch.stack(calibrated_list, dim=0)  # [B, node_num, gnn_dim]

        # Consistency residual: how much did GNN change the feature?
        # Compare calibrated with initial projection
        diff = torch.cat([x0, calibrated], dim=-1)  # [B, node_num, gnn_dim*2]
        residual = self.residual_mlp(diff).squeeze(-1)  # [B, node_num]
        residual = F.relu(residual)

        # Pad edge weights to same length per sample for batching
        max_edges = max(len(w) for w in edge_weights_list)
        padded_weights = torch.zeros(B, max_edges, device=device)
        for b, w in enumerate(edge_weights_list):
            padded_weights[b, :len(w)] = w

        return calibrated, residual, padded_weights


if __name__ == "__main__":
    B, input_dim = 4, 256
    features = torch.randn(B, input_dim)
    phase = torch.randint(0, 6, (B,))

    gnn = PhysicalConsistencyGNN(input_dim=input_dim, gnn_dim=128, num_layers=2)
    calibrated, residual, edge_weights = gnn(features, phase)
    print(f"calibrated shape: {calibrated.shape}")   # [B, 41, 128]
    print(f"residual shape: {residual.shape}")       # [B, 41]
    print(f"edge_weights shape: {edge_weights.shape}")  # [B, max_E]
    assert calibrated.shape == (B, 41, 128)
    assert residual.shape == (B, 41)
    print("GNN OK")
