"""Sparse expert router for FlightMoE v2.

Implements Top-k sparse gating with a load-balancing loss to prevent router collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseExpertRouter(nn.Module):
    """Sparse expert router with Top-k gating.

    Input:  [B, input_dim]
    Output:
        expert_weights: [B, num_experts]  (sparse, top-k non-zero, sum to 1)
        load_balance_loss: scalar
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gating = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_dim]

        Returns:
            weights: [B, num_experts]
            load_balance_loss: scalar
        """
        logits = self.gating(x)  # [B, num_experts]

        # Top-k sparse gating
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # [B, k]
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits.scatter_(1, topk_idx, topk_vals)

        weights = F.softmax(sparse_logits, dim=-1)  # [B, num_experts]

        # Load-balancing loss (Shazeer et al., 2017)
        # importance_i = sum over batch of gate probability for expert i
        # f_i = fraction of samples routed to expert i
        # loss = num_experts * sum_i(f_i * importance_i)
        router_prob = F.softmax(logits, dim=-1)  # dense probabilities for balance
        importance = router_prob.sum(dim=0)  # [num_experts]

        # f_i: fraction of samples where expert i is in top-k
        mask = torch.zeros_like(logits)
        mask.scatter_(1, topk_idx, 1.0)
        f = mask.mean(dim=0)  # [num_experts]

        load_balance_loss = self.num_experts * (f * importance).sum()

        return weights, load_balance_loss


if __name__ == "__main__":
    B, input_dim = 4, 256
    x = torch.randn(B, input_dim)
    router = SparseExpertRouter(input_dim=input_dim, num_experts=4, top_k=2)
    weights, lb_loss = router(x)
    print("weights shape:", weights.shape)
    print("weights:\n", weights)
    print("non-zero per row:", (weights > 0).sum(dim=1))
    print("row sums:", weights.sum(dim=1))
    print("load balance loss:", lb_loss.item())

    assert weights.shape == (B, 4)
    assert (weights > 0).sum(dim=1).eq(2).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(B), atol=1e-6)
    print("Router OK")
