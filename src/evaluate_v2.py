"""Evaluation script for FlightMoE v2.

Evaluates a trained FlightMoE v2 model on val/test_closed/test_open splits.
Reports overall, per-phase, and per-fault-type metrics.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.flightmoe_dataset import FlightMoEDataset
from train_v2 import FlightMoEv2
from utils.config import load_config
from utils.experiment_utils import (
    FAULT_TYPE_ID_MAP,
    compute_metrics,
    parse_case_id,
    save_json,
    save_score_npz,
)


def evaluate_split(model, loader, device):
    """Evaluate on a single split."""
    model.eval()
    all_scores = []
    all_labels = []
    all_phases = []
    all_case_ids = []
    all_expert_weights = []

    with torch.no_grad():
        for batch in loader:
            temporal = batch["temporal"].to(device)
            spectral = batch.get("spectral")
            if spectral is not None:
                spectral = spectral.to(device)
            phase = batch["phase"].to(device)
            modality_mask = batch["modality_mask"].to(device)

            final_score, expert_scores, expert_weights, _ = model(
                temporal, spectral, phase, modality_mask
            )
            all_scores.append(final_score.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_phases.append(batch["phase"].numpy())
            all_case_ids.append(batch["case_id"].numpy())
            all_expert_weights.append(expert_weights.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    phases = np.concatenate(all_phases)
    case_ids = np.concatenate(all_case_ids)
    expert_weights = np.concatenate(all_expert_weights)

    # Parse case ids for fault types
    fault_types = np.array([parse_case_id(int(cid))[2] for cid in case_ids])

    results = {
        "overall": compute_metrics(labels, scores),
        "per_phase": {},
        "per_fault_type": {},
        "router_mean_weights": expert_weights.mean(axis=0).tolist(),
        "router_phase_mean_weights": {},
    }

    for phase in range(6):
        mask = phases == phase
        if mask.sum() > 0 and len(np.unique(labels[mask])) == 2:
            results["per_phase"][str(phase)] = compute_metrics(labels[mask], scores[mask])
            results["router_phase_mean_weights"][str(phase)] = expert_weights[mask].mean(axis=0).tolist()

    for ft_id, ft_name in FAULT_TYPE_ID_MAP.items():
        mask = fault_types == ft_id
        if mask.sum() > 0 and len(np.unique(labels[mask])) == 2:
            results["per_fault_type"][ft_name] = compute_metrics(labels[mask], scores[mask])

    return results, scores, labels, phases, case_ids, expert_weights


def main():
    parser = argparse.ArgumentParser(description="Evaluate FlightMoE v2")
    parser.add_argument("--config", type=str, default="./configs/flightmoe_v2.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir or cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = FlightMoEv2(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    data_cfg = cfg["data"]
    splits = {
        "val": data_cfg["val_npz"],
        "test_closed": data_cfg["test_closed_npz"],
        "test_open": data_cfg["test_open_npz"],
    }

    all_results = {}
    for split_name, npz_path in splits.items():
        ds = FlightMoEDataset(npz_path, mode="test", perturbation_cfg=None, apply_perturb_prob=0.0)
        loader = DataLoader(
            ds,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg.get("pin_memory", True),
        )

        results, scores, labels, phases, case_ids, expert_weights = evaluate_split(model, loader, device)
        all_results[split_name] = results

        # Save scores
        save_score_npz(
            path=str(output_dir / f"flightmoe_v2_{split_name}.npz"),
            npz_path=npz_path,
            expert="flightmoe_v2",
            split=split_name,
            raw_scores=scores,
            expert_weights=expert_weights,
        )

        print(f"\n=== {split_name} ===")
        print(f"AUC: {results['overall']['auc_roc']:.4f}")
        print(f"F1:  {results['overall']['f1']:.4f}")

    save_json(str(output_dir / "flightmoe_v2_results.json"), all_results)
    print(f"\nResults saved to {output_dir / 'flightmoe_v2_results.json'}")


if __name__ == "__main__":
    main()
