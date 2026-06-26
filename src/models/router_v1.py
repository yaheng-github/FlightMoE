"""FlightMoE v1 score-level router."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.experiment_utils import compute_metrics, save_json


EXPERTS = ["madgan", "ganomaly", "hmm", "gdn_phase"]


class RouterMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_score_file(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_scores(score_dir, split, gdn_graph="phase"):
    paths = {
        "madgan": Path(score_dir) / f"madgan_{split}.npz",
        "ganomaly": Path(score_dir) / f"ganomaly_{split}.npz",
        "hmm": Path(score_dir) / f"hmm_{split}.npz",
        "gdn_phase": Path(score_dir) / f"gdn_{gdn_graph}_{split}.npz",
    }
    loaded = {name: load_score_file(path) for name, path in paths.items()}
    ref = loaded["madgan"]
    labels = ref["label"].astype(np.int32)
    phase = ref["phase_label"].astype(np.int64)
    for name, data in loaded.items():
        for key in ["label", "phase_label", "case_id"]:
            if not np.array_equal(ref[key], data[key]):
                raise ValueError(f"{name} {split} does not align on {key}")

    raw_scores = np.stack([loaded[name]["score_raw"].astype(np.float32) for name in EXPERTS], axis=1)
    residual_stats = loaded["gdn_phase"].get("residual_stats")
    if residual_stats is None:
        residual_stats = np.zeros((len(labels), 4), dtype=np.float32)
    return {
        "labels": labels,
        "phase": phase,
        "raw_scores": raw_scores,
        "residual_stats": residual_stats.astype(np.float32),
        "case_id": ref["case_id"],
        "fault_type": ref["fault_type"],
        "data_type": ref["data_type"],
    }


def temporal_stats(npz_path):
    data = np.load(npz_path)
    temporal = data["temporal"].astype(np.float32)
    mean_abs = np.mean(np.abs(temporal), axis=(1, 2))
    std = np.std(temporal, axis=(1, 2))
    value_range = np.max(temporal, axis=(1, 2)) - np.min(temporal, axis=(1, 2))
    diff = np.diff(temporal, axis=1)
    diff_std = np.std(diff, axis=(1, 2))
    diff_abs = np.mean(np.abs(diff), axis=(1, 2))
    return np.stack([mean_abs, std, value_range, diff_std, diff_abs], axis=1).astype(np.float32)


def fit_score_scaler(val_scores):
    med = np.median(val_scores, axis=0)
    q1 = np.percentile(val_scores, 25, axis=0)
    q3 = np.percentile(val_scores, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-8)
    return med.astype(np.float32), iqr.astype(np.float32)


def apply_scaler(scores, med, iqr):
    return ((scores - med) / iqr).astype(np.float32)


def fit_feature_scaler(features):
    med = np.median(features, axis=0)
    q1 = np.percentile(features, 25, axis=0)
    q3 = np.percentile(features, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-8)
    return med.astype(np.float32), iqr.astype(np.float32)


def build_features(bundle, score_med, score_iqr, stat_med, stat_iqr, npz_path):
    scores_norm = apply_scaler(bundle["raw_scores"], score_med, score_iqr)
    phase_onehot = np.eye(6, dtype=np.float32)[bundle["phase"]]
    stats = temporal_stats(npz_path)
    stats_norm = apply_scaler(stats, stat_med, stat_iqr)
    residual = bundle["residual_stats"].astype(np.float32)
    return np.concatenate([scores_norm, phase_onehot, stats_norm, residual], axis=1).astype(np.float32), scores_norm


def best_static_weight(scores, labels, seed=0):
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(scores, labels)
    logits = clf.decision_function(scores)
    weights = np.abs(clf.coef_[0])
    weights = weights / max(weights.sum(), 1e-8)
    return weights.astype(np.float32), logits.astype(np.float32)


def phase_static_weights(scores, labels, phases, seed=0):
    weights = np.zeros((6, 4), dtype=np.float32)
    for phase in range(6):
        mask = phases == phase
        if len(np.unique(labels[mask])) < 2:
            weights[phase] = np.ones(4, dtype=np.float32) / 4
            continue
        weights[phase], _ = best_static_weight(scores[mask], labels[mask], seed=seed + phase)
    return weights


def evaluate_baselines(split_name, scores_norm, labels, phases, static_weights, phase_weights):
    results = {}
    for idx, expert in enumerate(EXPERTS):
        results[f"single_{expert}"] = compute_metrics(labels, scores_norm[:, idx])
    avg_score = scores_norm.mean(axis=1)
    results["average_fusion"] = compute_metrics(labels, avg_score)
    static_score = scores_norm @ static_weights
    results["static_weighted"] = compute_metrics(labels, static_score)
    phase_score = np.sum(scores_norm * phase_weights[phases], axis=1)
    results["phase_static_weighted"] = compute_metrics(labels, phase_score)
    return results


def train_router(x, scores_norm, labels, args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    idx_train, idx_val = train_test_split(
        np.arange(len(labels)),
        test_size=args.router_val_ratio,
        random_state=args.seed,
        stratify=labels,
    )
    model = RouterMLP(x.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCELoss()
    ds = TensorDataset(
        torch.from_numpy(x[idx_train]),
        torch.from_numpy(scores_norm[idx_train]),
        torch.from_numpy(labels[idx_train].astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    best_state = None
    best_auc = -1.0
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        for xb, sb, yb in dl:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            weights = model(xb)
            fused = torch.sum(weights * sb, dim=1)
            prob = torch.sigmoid(fused)
            loss = bce(prob, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x[idx_val]).to(device)
            sv = torch.from_numpy(scores_norm[idx_val]).to(device)
            weights = model(xv)
            fused = torch.sum(weights * sv, dim=1).cpu().numpy()
        auc = roc_auc_score(labels[idx_val], fused)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= args.early_stop:
            break

    model.load_state_dict(best_state)
    return model


def router_predict(model, x, scores_norm, device_name):
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    model = model.to(device)
    model.eval()
    weights_all = []
    fused_all = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start:start + 4096]).to(device)
            sb = torch.from_numpy(scores_norm[start:start + 4096]).to(device)
            weights = model(xb)
            fused = torch.sum(weights * sb, dim=1)
            weights_all.append(weights.cpu().numpy())
            fused_all.append(fused.cpu().numpy())
    return np.concatenate(fused_all), np.concatenate(weights_all)


def run(args):
    set_seed(args.seed)
    splits = {
        "val": args.val_npz,
        "test_closed": args.test_closed,
        "test_open": args.test_open,
    }
    bundles = {split: load_scores(args.score_dir, split, gdn_graph=args.gdn_graph) for split in splits}
    score_med, score_iqr = fit_score_scaler(bundles["val"]["raw_scores"])
    val_stats = temporal_stats(args.val_npz)
    stat_med, stat_iqr = fit_feature_scaler(val_stats)

    built = {}
    for split, npz_path in splits.items():
        x, scores_norm = build_features(bundles[split], score_med, score_iqr, stat_med, stat_iqr, npz_path)
        built[split] = {"x": x, "scores_norm": scores_norm}

    static_weights, _ = best_static_weight(built["val"]["scores_norm"], bundles["val"]["labels"], seed=args.seed)
    phase_weights = phase_static_weights(
        built["val"]["scores_norm"],
        bundles["val"]["labels"],
        bundles["val"]["phase"],
        seed=args.seed,
    )
    router = train_router(built["val"]["x"], built["val"]["scores_norm"], bundles["val"]["labels"], args)

    results: Dict[str, Dict] = {
        "expert_order": EXPERTS,
        "static_weights": static_weights.tolist(),
        "phase_static_weights": phase_weights.tolist(),
        "splits": {},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": router.state_dict(),
            "input_dim": built["val"]["x"].shape[1],
            "score_median": score_med,
            "score_iqr": score_iqr,
            "stat_median": stat_med,
            "stat_iqr": stat_iqr,
            "expert_order": EXPERTS,
        },
        output_dir / "router_v1.pt",
    )

    for split in ["val", "test_closed", "test_open"]:
        labels = bundles[split]["labels"]
        phases = bundles[split]["phase"]
        scores_norm = built[split]["scores_norm"]
        split_results = evaluate_baselines(split, scores_norm, labels, phases, static_weights, phase_weights)
        router_score, router_weights = router_predict(router, built[split]["x"], scores_norm, args.device)
        split_results["mlp_router"] = compute_metrics(labels, router_score)
        split_results["router_mean_weights"] = router_weights.mean(axis=0).tolist()
        split_results["router_phase_mean_weights"] = {
            str(phase): router_weights[phases == phase].mean(axis=0).tolist()
            for phase in range(6)
            if np.any(phases == phase)
        }
        results["splits"][split] = split_results
        np.savez(
            output_dir / f"router_v1_{split}.npz",
            fused_score=router_score.astype(np.float32),
            expert_weights=router_weights.astype(np.float32),
            label=labels,
            phase_label=phases,
            case_id=bundles[split]["case_id"],
            fault_type=bundles[split]["fault_type"],
            data_type=bundles[split]["data_type"],
        )

    save_json(str(output_dir / "router_v1_results.json"), results)
    print(json.dumps(results["splits"]["test_open"], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Train FlightMoE v1 score-level router")
    parser.add_argument("--score_dir", type=str, default="./experiments/scores")
    parser.add_argument("--output_dir", type=str, default="./experiments/router_v1")
    parser.add_argument("--val_npz", type=str, default="./data/preprocessed/val.npz")
    parser.add_argument("--test_closed", type=str, default="./data/preprocessed/test_closed.npz")
    parser.add_argument("--test_open", type=str, default="./data/preprocessed/test_open.npz")
    parser.add_argument("--gdn_graph", type=str, default="phase")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--router_val_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
