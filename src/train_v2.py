"""End-to-end training script for FlightMoE v2.

Three-stage training:
1. Pretrain encoder + experts (router frozen / uniform)
2. Train sparse router (others frozen)
3. Joint fine-tuning with all losses
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.flightmoe_dataset import FlightMoEDataset
from models.anomaly_head import AnomalyHead
from models.encoder import PhaseAwareMultiViewEncoder
from models.expert_heads import ExpertPool
from models.gnn_consistency import PhysicalConsistencyGNN
from models.router_v2 import SparseExpertRouter
from utils.config import load_config
from utils.experiment_utils import compute_metrics, save_json


class FlightMoEv2(nn.Module):
    """End-to-end FlightMoE v2 model."""

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        enc_cfg = model_cfg["encoder"]
        gnn_cfg = model_cfg["gnn"]
        router_cfg = model_cfg["router"]
        experts_cfg = model_cfg["experts"]

        self.encoder = PhaseAwareMultiViewEncoder(
            temporal_dim=41,
            temporal_embed_dim=enc_cfg["temporal_embed_dim"],
            spectral_embed_dim=enc_cfg["spectral_embed_dim"],
            phase_embed_dim=enc_cfg["phase_embed_dim"],
            cross_attention_heads=enc_cfg["cross_attention_heads"],
            cross_attention_dropout=enc_cfg["cross_attention_dropout"],
            output_dim=enc_cfg["output_dim"],
            dropout=enc_cfg.get("dropout", 0.1),
        )

        self.gnn = PhysicalConsistencyGNN(
            input_dim=enc_cfg["output_dim"],
            node_num=gnn_cfg["node_dim"],
            gnn_dim=gnn_cfg["gnn_dim"],
            num_layers=gnn_cfg["num_layers"],
            heads=gnn_cfg["heads"],
            dropout=gnn_cfg["dropout"],
        )

        # Project consistency feature to same dim as encoder output
        self.consistency_proj = nn.Linear(gnn_cfg["gnn_dim"], enc_cfg["output_dim"])

        router_input_dim = enc_cfg["output_dim"] + 3  # + residual mean/std + mask mean
        self.router = SparseExpertRouter(
            input_dim=router_input_dim,
            num_experts=router_cfg["num_experts"],
            top_k=router_cfg["top_k"],
            hidden_dim=router_cfg["hidden_dim"],
            dropout=router_cfg["dropout"],
        )

        self.experts = ExpertPool(
            input_dim=enc_cfg["output_dim"],
            hidden_dim=experts_cfg["hidden_dim"],
            dropout=enc_cfg.get("dropout", 0.1),
        )

        self.anomaly_head = AnomalyHead(
            input_dim=enc_cfg["output_dim"],
            hidden_dim=model_cfg["anomaly_head"]["hidden_dim"],
            dropout=model_cfg["anomaly_head"].get("dropout", 0.1),
        )

    def forward(self, temporal, spectral, phase, modality_mask):
        """
        Args:
            temporal: [B, T, F]
            spectral: [B, G, F, T, C] or None
            phase: [B]
            modality_mask: [B, F]

        Returns:
            final_score: [B]
            expert_scores: [B, 4]
            expert_weights: [B, 4]
            load_balance_loss: scalar
        """
        enc_out = self.encoder(temporal, spectral, phase)  # [B, output_dim]

        calibrated, residual, edge_weights = self.gnn(enc_out, phase)
        # calibrated: [B, 41, gnn_dim], residual: [B, 41]

        consistency_feat = calibrated.mean(dim=1)  # [B, gnn_dim]
        consistency_feat = self.consistency_proj(consistency_feat)  # [B, output_dim]

        # Router input
        residual_mean = residual.mean(dim=1, keepdim=True)  # [B, 1]
        residual_std = residual.std(dim=1, keepdim=True)  # [B, 1]
        mask_mean = modality_mask.mean(dim=1, keepdim=True)  # [B, 1]
        router_input = torch.cat([enc_out, residual_mean, residual_std, mask_mean], dim=-1)

        expert_weights, load_balance_loss = self.router(router_input)

        expert_scores = self.experts(enc_out, enc_out, enc_out, consistency_feat)

        # Weighted fusion
        fused = (expert_weights * expert_scores).sum(dim=1)  # [B]

        # Optional final head
        final_score = self.anomaly_head(enc_out) + fused

        return final_score, expert_scores, expert_weights, load_balance_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_loaders(cfg):
    data_cfg = cfg["data"]
    perturb_cfg = cfg["perturbation"]
    train_ds = FlightMoEDataset(
        npz_path=data_cfg["train_npz"],
        mode="train",
        perturbation_cfg=perturb_cfg,
        apply_perturb_prob=perturb_cfg.get("apply_prob", 0.0),
        seed=cfg["training"]["seed"],
    )
    val_ds = FlightMoEDataset(
        npz_path=data_cfg["val_npz"],
        mode="val",
        perturbation_cfg=None,
        apply_perturb_prob=0.0,
        seed=cfg["training"]["seed"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, loss_weights, stage=1):
    model.train()
    total_loss = 0.0
    for batch in loader:
        temporal = batch["temporal"].to(device)
        spectral = batch.get("spectral")
        if spectral is not None:
            spectral = spectral.to(device)
        phase = batch["phase"].to(device)
        labels = batch["label"].float().to(device)
        modality_mask = batch["modality_mask"].to(device)

        optimizer.zero_grad()
        final_score, expert_scores, expert_weights, lb_loss = model(
            temporal, spectral, phase, modality_mask
        )

        loss = criterion(final_score, labels)
        if stage >= 3:
            loss = loss + loss_weights.get("load_balance", 0.01) * lb_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    for batch in loader:
        temporal = batch["temporal"].to(device)
        spectral = batch.get("spectral")
        if spectral is not None:
            spectral = spectral.to(device)
        phase = batch["phase"].to(device)
        modality_mask = batch["modality_mask"].to(device)

        final_score, _, _, _ = model(temporal, spectral, phase, modality_mask)
        all_scores.append(final_score.cpu().numpy())
        all_labels.append(batch["label"].numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return compute_metrics(labels, scores)


def main():
    parser = argparse.ArgumentParser(description="Train FlightMoE v2")
    parser.add_argument("--config", type=str, default="./configs/flightmoe_v2.yaml")
    parser.add_argument("--stage", type=int, default=0, help="0=all stages, 1/2/3=single stage")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs_override", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    train_loader, val_loader = build_loaders(cfg)
    model = FlightMoEv2(cfg).to(device)

    criterion = nn.BCEWithLogitsLoss()
    loss_weights = cfg["training"]["loss_weights"]

    output_dir = Path(cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: train encoder + experts
    if args.stage in (0, 1):
        print("\n=== Stage 1: Pretrain encoder + experts ===")
        model.router.requires_grad_(False)
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["training"]["stage1"]["lr"],
            weight_decay=cfg["training"]["stage1"]["weight_decay"],
        )
        epochs = args.epochs_override or cfg["training"]["stage1"]["epochs"]
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, opt, criterion, device, loss_weights, stage=1)
            print(f"Epoch {epoch+1}/{epochs}, loss: {loss:.4f}")
        torch.save(model.state_dict(), ckpt_dir / "stage1.pt")
        model.router.requires_grad_(True)

    # Stage 2: train router
    if args.stage in (0, 2):
        print("\n=== Stage 2: Train router ===")
        model.encoder.requires_grad_(False)
        model.gnn.requires_grad_(False)
        model.experts.requires_grad_(False)
        model.anomaly_head.requires_grad_(False)
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["training"]["stage2"]["lr"],
            weight_decay=cfg["training"]["stage2"]["weight_decay"],
        )
        epochs = args.epochs_override or cfg["training"]["stage2"]["epochs"]
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, opt, criterion, device, loss_weights, stage=2)
            print(f"Epoch {epoch+1}/{epochs}, loss: {loss:.4f}")
        torch.save(model.state_dict(), ckpt_dir / "stage2.pt")
        model.encoder.requires_grad_(True)
        model.gnn.requires_grad_(True)
        model.experts.requires_grad_(True)
        model.anomaly_head.requires_grad_(True)

    # Stage 3: joint fine-tune
    if args.stage in (0, 3):
        print("\n=== Stage 3: Joint fine-tuning ===")
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["stage3"]["lr"],
            weight_decay=cfg["training"]["stage3"]["weight_decay"],
        )
        epochs = args.epochs_override or cfg["training"]["stage3"]["epochs"]
        best_auc = -1.0
        patience = 0
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, opt, criterion, device, loss_weights, stage=3)
            metrics = evaluate(model, val_loader, device)
            auc = metrics["auc_roc"]
            print(f"Epoch {epoch+1}/{epochs}, loss: {loss:.4f}, val AUC: {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
                patience = 0
            else:
                patience += 1
            if patience >= cfg["training"]["stage3"]["early_stop_patience"]:
                print("Early stopping")
                break

    # Final evaluation
    print("\n=== Final evaluation ===")
    metrics = evaluate(model, val_loader, device)
    print(metrics)
    save_json(str(output_dir / "metrics_val.json"), metrics)


if __name__ == "__main__":
    main()
