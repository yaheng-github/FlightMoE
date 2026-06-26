"""
MAD-GAN 异常检测评估脚本（适配 RflyMAD 数据）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "third_party" / "madgan-pytorch"))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from madgan.models import Generator, Discriminator
from dataset import get_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dr_score(generator, discriminator, dataloader, latent_dim=32, res_weight=0.2):
    """
    计算 DR-Score (Differential Reconstruction Score)
    score = res_weight * reconstruction_error + (1 - res_weight) * discriminator_score

    其中 reconstruction_error 通过优化 latent space Z 得到最佳重构后计算。
    """
    generator.eval()
    discriminator.eval()

    all_scores = []
    all_labels = []

    for batch in dataloader:
        x = batch.to(DEVICE)  # [batch, window_size, 41]
        bs = x.size(0)

        # ---- Discriminator Score ----
        with torch.no_grad():
            d_score = discriminator(x)  # [batch, window_size, 1]
        d_score = d_score.mean(dim=(1, 2))  # [batch]

        # ---- Reconstruction Loss (优化 latent Z) ----
        # CuDNN LSTM backward 需要 train mode，因此临时切换并冻结 generator 权重
        generator.train()
        for p in generator.parameters():
            p.requires_grad = False

        Z = torch.empty((bs, x.size(1), latent_dim), device=DEVICE, requires_grad=True)
        nn.init.normal_(Z, std=0.05)
        z_optim = torch.optim.RMSprop([Z], lr=0.1)

        for _ in range(10):
            z_optim.zero_grad()
            recon = generator(Z)  # [batch, window_size, 41]
            loss = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2)).sum()
            loss.backward()
            z_optim.step()

        for p in generator.parameters():
            p.requires_grad = True
        generator.eval()

        with torch.no_grad():
            best_recon = generator(Z)
            recon_error = F.mse_loss(best_recon, x, reduction="none").mean(dim=(1, 2))  # [batch]

        # ---- DR-Score ----
        scores = res_weight * recon_error + (1 - res_weight) * d_score
        all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores)


def evaluate_madgan(
    test_npz: str,
    generator_path: str,
    discriminator_path: str,
    batch_size: int = 32,
    latent_dim: int = 32,
    res_weight: float = 0.2,
    output_json: str = None,
):
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
    import json

    # 加载模型
    generator = Generator.from_pretrained(generator_path, map_location=DEVICE)
    discriminator = Discriminator.from_pretrained(discriminator_path, map_location=DEVICE)

    generator.to(DEVICE)
    discriminator.to(DEVICE)

    # 加载测试数据（包含正常+异常）
    test_dl = get_dataloader(test_npz, batch_size=batch_size, normal_only=False, shuffle=False, drop_last=False)

    # 获取真实标签
    data = np.load(test_npz)
    labels = data['anomaly_labels']

    print(f"[INFO] 测试样本: {len(labels)}, 异常比例: {labels.mean()*100:.2f}%")

    # 计算 DR-Score
    print("[INFO] 计算 DR-Score...")
    scores = compute_dr_score(generator, discriminator, test_dl, latent_dim, res_weight)

    if len(scores) != len(labels):
        raise ValueError(f"scores/labels length mismatch: {len(scores)} vs {len(labels)}")

    # 计算指标
    auc_roc = roc_auc_score(labels, scores)

    # 找最佳阈值 (F1)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    preds = (scores >= best_threshold).astype(int)
    f1 = f1_score(labels, preds)
    precision_val = precision_score(labels, preds)
    recall_val = recall_score(labels, preds)

    results = {
        "test_file": test_npz,
        "generator": generator_path,
        "discriminator": discriminator_path,
        "auc_roc": float(auc_roc),
        "f1": float(f1),
        "precision": float(precision_val),
        "recall": float(recall_val),
        "best_threshold": float(best_threshold),
        "res_weight": res_weight,
        "n_samples": int(len(labels)),
        "n_anomalies": int(labels.sum()),
    }

    print("\n========== 评估结果 ==========")
    print(f"  AUC-ROC : {auc_roc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"  Precision: {precision_val:.4f}")
    print(f"  Recall   : {recall_val:.4f}")
    print(f"  Best Threshold: {best_threshold:.4f}")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] 结果保存至 {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MAD-GAN Baseline Evaluation")
    parser.add_argument("--test_npz", type=str, required=True, help="测试集 npz 路径")
    parser.add_argument("--generator", type=str, required=True, help="Generator checkpoint 路径")
    parser.add_argument("--discriminator", type=str, required=True, help="Discriminator checkpoint 路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--res_weight", type=float, default=0.2, help="重构误差权重 (DR-Score)")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    evaluate_madgan(
        test_npz=args.test_npz,
        generator_path=args.generator,
        discriminator_path=args.discriminator,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        res_weight=args.res_weight,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
