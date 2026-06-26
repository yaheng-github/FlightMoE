"""GDN graph-structure ablation and window-level score export."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, './third_party/gdn')
sys.path.insert(0, './src')

from models.GDN import GDN
from util.data import get_err_median_and_iqr
from util.env import get_device, set_device

from dataset import GDNTimeDataset
from graph_struct import PHASE_NAMES, get_graph_edge_index
from utils.experiment_utils import save_score_npz


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_model(model, train_dataloader, val_dataloader, epochs=100, lr=0.001, decay=0, early_stop_win=15):
    device = get_device()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_func = nn.MSELoss(reduction='mean')
    min_loss = float('inf')
    stop_improve_count = 0

    for epoch in range(epochs):
        acu_loss = 0.0
        model.train()
        for x, y, labels, edge_index in train_dataloader:
            x, y, edge_index = [item.float().to(device) for item in [x, y, edge_index]]
            optimizer.zero_grad()
            out = model(x, edge_index).float()
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            acu_loss += loss.item()

        val_loss, _ = test_model(model, val_dataloader)
        if val_loss < min_loss:
            min_loss = val_loss
            stop_improve_count = 0
        else:
            stop_improve_count += 1

        print(f'  epoch ({epoch + 1}/{epochs}) train_loss={acu_loss/len(train_dataloader):.6f} val_loss={val_loss:.6f}')
        if stop_improve_count >= early_stop_win:
            break

    return model


def test_model(model, dataloader):
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()
    model.eval()
    t_predicted, t_ground, t_labels = [], [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y, labels, edge_index in dataloader:
            x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
            predicted = model(x, edge_index).float()
            loss = loss_func(predicted, y)
            total_loss += loss.item()
            n_batches += 1
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])
            t_predicted.append(predicted.cpu().numpy())
            t_ground.append(y.cpu().numpy())
            t_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, [np.concatenate(t_predicted), np.concatenate(t_ground), np.concatenate(t_labels)]


def score_from_predictions(test_predict, test_gt, test_labels, val_predict, val_gt, topk=5):
    scores_per_sensor = []
    for i in range(test_predict.shape[1]):
        n_err_mid, n_err_iqr = get_err_median_and_iqr(val_predict[:, i], val_gt[:, i])
        test_delta = np.abs(test_predict[:, i] - test_gt[:, i])
        scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + 1e-2)
        scores_per_sensor.append(scores)

    scores_per_sensor = np.stack(scores_per_sensor, axis=1)  # [N, node_num]
    topk_indices = np.argpartition(scores_per_sensor, -topk, axis=1)[:, -topk:]
    topk_scores = np.take_along_axis(scores_per_sensor, topk_indices, axis=1)
    total_scores = np.sum(topk_scores, axis=1)
    residual_stats = np.stack([
        scores_per_sensor.mean(axis=1),
        scores_per_sensor.std(axis=1),
        scores_per_sensor.max(axis=1),
        total_scores,
    ], axis=1).astype(np.float32)

    precision, recall, thresholds = precision_recall_curve(test_labels, total_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    preds = (total_scores >= best_th).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(test_labels, total_scores)),
        "f1": float(f1_score(test_labels, preds)),
        "precision": float(precision_score(test_labels, preds)),
        "recall": float(recall_score(test_labels, preds)),
        "best_threshold": float(best_th),
        "n_samples": int(len(test_labels)),
        "n_anomalies": int(test_labels.sum()),
    }
    return total_scores.astype(np.float32), residual_stats, metrics


def build_model(edge_index, args):
    device = get_device()
    return GDN(
        [edge_index],
        node_num=41,
        dim=args.dim,
        input_dim=args.slide_win,
        out_layer_num=args.out_layer_num,
        out_layer_inter_dim=args.out_layer_inter_dim,
        topk=args.topk,
    ).to(device)


def train_phase_model(phase_idx, phase_name, train_temporal, train_labels, train_phases, edge_index, args):
    phase_mask = train_phases == phase_idx
    normal_mask = train_labels[phase_mask] == 0
    normal_temporal = train_temporal[phase_mask][normal_mask]
    normal_labels = train_labels[phase_mask][normal_mask]

    n_normal = len(normal_temporal)
    if n_normal == 0:
        return None
    print(f"  normal windows: {n_normal}")

    indices = np.arange(n_normal)
    np.random.shuffle(indices)
    n_val = max(1, int(n_normal * args.val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_dataset = GDNTimeDataset(
        normal_temporal[train_idx],
        normal_labels[train_idx],
        edge_index,
        mode='train',
        slide_win=args.slide_win,
        slide_stride=args.slide_stride,
    )
    val_dataset = GDNTimeDataset(
        normal_temporal[val_idx],
        normal_labels[val_idx],
        edge_index,
        mode='val',
        slide_win=args.slide_win,
        slide_stride=1,
    )
    train_dl = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    model = build_model(edge_index, args)
    return train_model(model, train_dl, val_dl, epochs=args.epoch, lr=args.lr, decay=args.decay, early_stop_win=args.early_stop)


def evaluate_phase_model(model, edge_index, phase_temporal, phase_labels, val_temporal, val_labels, args):
    test_dataset = GDNTimeDataset(
        phase_temporal,
        phase_labels,
        edge_index,
        mode='test',
        slide_win=args.slide_win,
        slide_stride=1,
    )
    val_dataset = GDNTimeDataset(
        val_temporal[-min(len(val_temporal), args.norm_windows):],
        val_labels[-min(len(val_labels), args.norm_windows):],
        edge_index,
        mode='val',
        slide_win=args.slide_win,
        slide_stride=1,
    )
    test_dl = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    _, test_result = test_model(model, test_dl)
    _, val_result = test_model(model, val_dl)
    return score_from_predictions(
        np.array(test_result[0]),
        np.array(test_result[1]),
        np.array(test_result[2])[:, 0],
        np.array(val_result[0]),
        np.array(val_result[1]),
        topk=args.topk,
    )


def run_ablation(args):
    set_seed(args.seed)
    set_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.score_dir, exist_ok=True)

    train_data = np.load(args.train_npz)
    train_temporal = train_data['temporal']
    train_labels = train_data['anomaly_labels']
    train_phases = train_data['phase_labels']

    graph_types = [g.strip() for g in args.graph_types.split(',') if g.strip()]
    all_results = {}

    for graph_type in graph_types:
        print(f"\n{'=' * 70}\nGraph: {graph_type}\n{'=' * 70}")
        phase_models = {}
        phase_edges = {}
        graph_dir = Path(args.output_dir) / graph_type
        graph_dir.mkdir(parents=True, exist_ok=True)

        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            print(f"\n[Train phase {phase_idx}] {phase_name}")
            edge_index = get_graph_edge_index(graph_type, phase_name=phase_name, threshold=args.graph_threshold, seed=args.seed + phase_idx)
            phase_edges[phase_idx] = edge_index
            model = train_phase_model(phase_idx, phase_name, train_temporal, train_labels, train_phases, edge_index, args)
            if model is None:
                continue
            phase_models[phase_idx] = model
            torch.save(model.state_dict(), graph_dir / f"gdn_{phase_name}.pt")

        graph_results = {}
        for split_name, test_path in [('val', args.val_npz), ('test_closed', args.test_closed), ('test_open', args.test_open)]:
            test_data = np.load(test_path)
            test_temporal = test_data['temporal']
            test_labels = test_data['anomaly_labels']
            test_phases = test_data['phase_labels']
            split_scores = np.zeros(len(test_labels), dtype=np.float32)
            split_residual = np.zeros((len(test_labels), 4), dtype=np.float32)
            split_metrics_by_phase = {}

            for phase_idx, phase_name in enumerate(PHASE_NAMES):
                if phase_idx not in phase_models:
                    continue
                mask = test_phases == phase_idx
                if mask.sum() == 0:
                    continue
                val_mask = (train_phases == phase_idx) & (train_labels == 0)
                scores, residual_stats, metrics = evaluate_phase_model(
                    phase_models[phase_idx],
                    phase_edges[phase_idx],
                    test_temporal[mask],
                    test_labels[mask],
                    train_temporal[val_mask],
                    train_labels[val_mask],
                    args,
                )
                split_scores[mask] = scores
                split_residual[mask] = residual_stats
                split_metrics_by_phase[phase_name] = metrics
                print(f"  [{split_name}] {phase_name:10s}: AUC={metrics['auc_roc']:.4f} F1={metrics['f1']:.4f}")

            global_metrics = score_metrics_safe(test_labels, split_scores)
            print(f"  [{split_name}] global: AUC={global_metrics['auc_roc']:.4f} F1={global_metrics['f1']:.4f}")
            graph_results[split_name] = {
                "global": global_metrics,
                "by_phase": split_metrics_by_phase,
            }
            save_score_npz(
                Path(args.score_dir) / f"gdn_{graph_type}_{split_name}.npz",
                test_path,
                expert=f"gdn_{graph_type}",
                split=split_name,
                raw_scores=split_scores,
                residual_stats=split_residual,
            )

        all_results[graph_type] = graph_results

    with open(Path(args.output_dir) / "gdn_graph_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def score_metrics_safe(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    preds = (scores >= best_th).astype(int)
    return {
        "auc_roc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "best_threshold": float(best_th),
        "n_samples": int(len(labels)),
        "n_anomalies": int(labels.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="GDN graph ablation")
    parser.add_argument("--train_npz", type=str, default="./data/preprocessed/train.npz")
    parser.add_argument("--val_npz", type=str, default="./data/preprocessed/val.npz")
    parser.add_argument("--test_closed", type=str, default="./data/preprocessed/test_closed.npz")
    parser.add_argument("--test_open", type=str, default="./data/preprocessed/test_open.npz")
    parser.add_argument("--graph_types", type=str, default="phase,full,identity,global,random")
    parser.add_argument("--graph_threshold", type=float, default=0.0)
    parser.add_argument("--slide_win", type=int, default=32)
    parser.add_argument("--slide_stride", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out_layer_num", type=int, default=1)
    parser.add_argument("--out_layer_inter_dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--norm_windows", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./experiments/gdn_ablation")
    parser.add_argument("--score_dir", type=str, default="./experiments/scores")
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
