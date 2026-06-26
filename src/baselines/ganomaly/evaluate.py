"""
GANomaly 异常检测评估脚本
"""
import argparse
import json

import numpy as np
import torch

from dataset import get_dataloader
from networks import NetG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_ganomaly(test_npz: str, netg_path: str, batch_size: int = 32, image_size: int = 64):
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve

    netg = NetG(isize=image_size, nz=64, nc=3, ndf=64).to(DEVICE)
    checkpoint = torch.load(netg_path, map_location=DEVICE)
    netg.load_state_dict(checkpoint['state_dict'])
    netg.eval()

    test_dl = get_dataloader(
        test_npz,
        batch_size=batch_size,
        normal_only=False,
        shuffle=False,
        image_size=image_size,
        drop_last=False,
    )

    data_np = np.load(test_npz)
    labels = data_np['anomaly_labels']

    an_scores = []
    gt_labels = []

    with torch.no_grad():
        for data, lbls in test_dl:
            data = data.to(DEVICE)
            fake, latent_i, latent_o = netg(data)
            error = torch.mean(torch.pow(latent_i - latent_o, 2), dim=(1, 2, 3))
            an_scores.append(error.cpu())
            gt_labels.append(lbls)

    an_scores = torch.cat(an_scores).numpy()
    gt_labels = torch.cat(gt_labels).numpy()

    if len(an_scores) != len(labels):
        raise ValueError(f"scores/labels length mismatch: {len(an_scores)} vs {len(labels)}")

    # Normalize
    an_scores = (an_scores - an_scores.min()) / (an_scores.max() - an_scores.min() + 1e-10)

    auc = roc_auc_score(labels, an_scores)

    # Best F1 threshold
    precision, recall, thresholds = precision_recall_curve(labels, an_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    preds = (an_scores >= best_threshold).astype(int)
    f1 = f1_score(labels, preds)
    precision_val = precision_score(labels, preds)
    recall_val = recall_score(labels, preds)

    results = {
        "test_file": test_npz,
        "netg": netg_path,
        "auc_roc": float(auc),
        "f1": float(f1),
        "precision": float(precision_val),
        "recall": float(recall_val),
        "best_threshold": float(best_threshold),
        "n_samples": int(len(labels)),
        "n_anomalies": int(labels.sum()),
    }

    print("\n========== GANomaly 评估结果 ==========")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"  Precision: {precision_val:.4f}")
    print(f"  Recall   : {recall_val:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GANomaly Baseline Evaluation")
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--netg", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    results = evaluate_ganomaly(
        test_npz=args.test_npz,
        netg_path=args.netg,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] 结果保存至 {args.output_json}")


if __name__ == "__main__":
    main()
