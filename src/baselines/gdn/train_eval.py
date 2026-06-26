"""
GDN 基线（Consistency Expert）
分 6 个飞行阶段训练，各用各的物理一致性邻接矩阵
"""
import sys
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, precision_recall_curve)

sys.path.insert(0, './third_party/gdn')
from models.GDN import GDN
from util.env import get_device, set_device
from util.data import get_err_median_and_iqr

from dataset import GDNTimeDataset
from graph_struct import load_phase_edge_index, PHASE_NAMES


def train_model(model, train_dataloader, val_dataloader, epochs=100, lr=0.001, decay=0, early_stop_win=15):
    device = get_device()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_func = nn.MSELoss(reduction='mean')

    min_loss = float('inf')
    stop_improve_count = 0

    for epoch in range(epochs):
        acu_loss = 0
        model.train()
        for x, y, labels, edge_index in train_dataloader:
            x, y, edge_index = [item.float().to(device) for item in [x, y, edge_index]]
            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
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

        print(f'  epoch ({epoch}/{epochs}) train_loss={acu_loss/len(train_dataloader):.6f} val_loss={val_loss:.6f}')

        if stop_improve_count >= early_stop_win:
            break

    return model


def test_model(model, dataloader):
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()
    model.eval()

    t_predicted, t_ground, t_labels = [], [], []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for x, y, labels, edge_index in dataloader:
            x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
            predicted = model(x, edge_index).float().to(device)
            loss = loss_func(predicted, y)
            total_loss += loss.item()
            n_batches += 1

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_predicted) == 0:
                t_predicted = predicted
                t_ground = y
                t_labels = labels
            else:
                t_predicted = torch.cat((t_predicted, predicted), dim=0)
                t_ground = torch.cat((t_ground, y), dim=0)
                t_labels = torch.cat((t_labels, labels), dim=0)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, [t_predicted.cpu().numpy(), t_ground.cpu().numpy(), t_labels.cpu().numpy()]


def evaluate_phase(model, test_dataloader, val_dataloader, topk=5):
    _, test_result = test_model(model, test_dataloader)
    _, val_result = test_model(model, val_dataloader)

    test_predict = np.array(test_result[0])      # [N, node_num]
    test_gt = np.array(test_result[1])           # [N, node_num]
    test_labels = np.array(test_result[2])[:, 0] # [N]

    val_predict = np.array(val_result[0])
    val_gt = np.array(val_result[1])

    # 每个传感器独立标准化
    scores_per_sensor = []
    for i in range(test_predict.shape[1]):
        n_err_mid, n_err_iqr = get_err_median_and_iqr(val_predict[:, i], val_gt[:, i])
        test_delta = np.abs(test_predict[:, i] - test_gt[:, i])
        scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + 1e-2)
        scores_per_sensor.append(scores)

    scores_per_sensor = np.stack(scores_per_sensor, axis=0)  # [node_num, N]

    # topk 聚合
    total_features = scores_per_sensor.shape[0]
    topk_indices = np.argpartition(
        scores_per_sensor,
        range(total_features - topk - 1, total_features),
        axis=0
    )[-topk:]
    total_topk_err_scores = np.sum(
        np.take_along_axis(scores_per_sensor, topk_indices, axis=0),
        axis=0
    )

    # 计算指标
    auc = roc_auc_score(test_labels, total_topk_err_scores)

    precision, recall, thresholds = precision_recall_curve(test_labels, total_topk_err_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    preds = (total_topk_err_scores >= best_th).astype(int)
    f1 = f1_score(test_labels, preds)
    prec = precision_score(test_labels, preds)
    rec = recall_score(test_labels, preds)

    return auc, f1, prec, rec, best_th, total_topk_err_scores, test_labels


def main():
    parser = argparse.ArgumentParser(description="GDN Consistency Expert")
    parser.add_argument("--train_npz", type=str, default="./data/preprocessed/train.npz")
    parser.add_argument("--test_closed", type=str, default="./data/preprocessed/test_closed.npz")
    parser.add_argument("--test_open", type=str, default="./data/preprocessed/test_open.npz")
    parser.add_argument("--slide_win", type=int, default=32)
    parser.add_argument("--slide_stride", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out_layer_num", type=int, default=1)
    parser.add_argument("--out_layer_inter_dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/gdn")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    set_device(args.device)
    device = get_device()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    train_data = np.load(args.train_npz)
    test_closed = np.load(args.test_closed)
    test_open = np.load(args.test_open)

    train_temporal = train_data['temporal']
    train_labels = train_data['anomaly_labels']
    train_phases = train_data['phase_labels']

    print("=" * 70)
    print("GDN Consistency Expert 基线（分 6 阶段训练）")
    print("=" * 70)

    phase_models = {}

    # ---------- 训练 6 个阶段 ----------
    for phase_idx, phase_name in enumerate(PHASE_NAMES):
        print(f"\n[Phase {phase_idx}] {phase_name}")
        print("-" * 50)

        edge_index = load_phase_edge_index(phase_name, threshold=0.0)
        edge_index_sets = [edge_index]

        phase_mask = train_phases == phase_idx
        phase_temporal = train_temporal[phase_mask]
        phase_labels = train_labels[phase_mask]

        normal_mask = phase_labels == 0
        normal_temporal = phase_temporal[normal_mask]
        normal_labels = phase_labels[normal_mask]

        n_normal = len(normal_temporal)
        if n_normal == 0:
            print(f"  [WARN] 无正常样本，跳过")
            continue
        print(f"  正常样本: {n_normal}")

        # train/val 切分
        indices = np.arange(n_normal)
        np.random.shuffle(indices)
        n_val = max(1, int(n_normal * args.val_ratio))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_sub = normal_temporal[train_idx]
        train_lab = normal_labels[train_idx]
        val_sub = normal_temporal[val_idx]
        val_lab = normal_labels[val_idx]

        train_dataset = GDNTimeDataset(train_sub, train_lab, edge_index,
                                       mode='train', slide_win=args.slide_win, slide_stride=args.slide_stride)
        val_dataset = GDNTimeDataset(val_sub, val_lab, edge_index,
                                     mode='val', slide_win=args.slide_win, slide_stride=1)

        train_dl = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

        model = GDN(edge_index_sets, node_num=41,
                    dim=args.dim,
                    input_dim=args.slide_win,
                    out_layer_num=args.out_layer_num,
                    out_layer_inter_dim=args.out_layer_inter_dim,
                    topk=args.topk).to(device)

        print(f"  训练...")
        model = train_model(model, train_dl, val_dl,
                            epochs=args.epoch, lr=args.lr, decay=args.decay)

        phase_models[phase_idx] = model
        torch.save(model.state_dict(), f"{args.output_dir}/gdn_{phase_name}.pt")

    # ---------- 评估 ----------
    print(f"\n{'='*70}")
    print("评估")
    print(f"{'='*70}")

    for test_name, test_npz in [("Closed", test_closed), ("Open", test_open)]:
        print(f"\n[{test_name} Set]")

        test_temporal = test_npz['temporal']
        test_labels = test_npz['anomaly_labels']
        test_phases = test_npz['phase_labels']

        all_scores = []
        all_labels = []

        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            if phase_idx not in phase_models:
                continue

            phase_mask = test_phases == phase_idx
            phase_test_temporal = test_temporal[phase_mask]
            phase_test_labels = test_labels[phase_mask]

            if len(phase_test_temporal) == 0:
                continue

            edge_index = load_phase_edge_index(phase_name, threshold=0.0)

            test_dataset = GDNTimeDataset(phase_test_temporal, phase_test_labels, edge_index,
                                          mode='test', slide_win=args.slide_win, slide_stride=1)
            test_dl = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

            # val 用于计算 error median / iqr，用该阶段训练集的正常样本
            val_temporal = train_temporal[(train_phases == phase_idx) & (train_labels == 0)]
            val_labels = train_labels[(train_phases == phase_idx) & (train_labels == 0)]
            val_dataset = GDNTimeDataset(val_temporal[-min(len(val_temporal), 200):],
                                         val_labels[-min(len(val_labels), 200):],
                                         edge_index, mode='val', slide_win=args.slide_win, slide_stride=1)
            val_dl = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

            model = phase_models[phase_idx]
            auc, f1, prec, rec, best_th, scores, labs = evaluate_phase(model, test_dl, val_dl, topk=args.topk)

            print(f"  {phase_name:10s}: AUC={auc:.4f}  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  (n={len(phase_test_temporal)})")

            all_scores.extend(scores.tolist())
            all_labels.extend(labs.tolist())

        if len(all_scores) > 0 and len(set(all_labels)) > 1:
            global_auc = roc_auc_score(all_labels, all_scores)
            precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_th = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            preds = (np.array(all_scores) >= best_th).astype(int)
            global_f1 = f1_score(all_labels, preds)
            global_prec = precision_score(all_labels, preds)
            global_rec = recall_score(all_labels, preds)
            print(f"\n  [{test_name}] 全局: AUC={global_auc:.4f}  F1={global_f1:.4f}  P={global_prec:.4f}  R={global_rec:.4f}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
