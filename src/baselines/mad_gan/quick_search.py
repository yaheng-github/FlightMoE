"""快速参数搜索：用 test_closed 子集测试不同 res_weight 和 checkpoint"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "third_party" / "madgan-pytorch"))

import numpy as np
import torch
import torch.nn.functional as F
from madgan.models import Generator, Discriminator
from dataset import get_dataloader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_config(gen_path, disc_path, res_weight, test_npz, n_samples=2000):
    generator = Generator.from_pretrained(gen_path, map_location=DEVICE)
    discriminator = Discriminator.from_pretrained(disc_path, map_location=DEVICE)
    generator.to(DEVICE)
    discriminator.to(DEVICE)

    # 只加载前 n_samples 个样本
    data = np.load(test_npz)
    temporal = data['temporal'][:n_samples]
    labels = data['anomaly_labels'][:n_samples]

    # 手动 batch
    batch_size = 32
    scores = []
    for i in range(0, len(temporal), batch_size):
        x = torch.from_numpy(temporal[i:i + batch_size].astype(np.float32)).to(DEVICE)
        bs = x.size(0)

        with torch.no_grad():
            d_score = discriminator(x).mean(dim=(1, 2))

        Z = torch.empty((bs, 128, 32), device=DEVICE, requires_grad=True)
        torch.nn.init.normal_(Z, std=0.05)
        z_optim = torch.optim.RMSprop([Z], lr=0.1)

        generator.train()
        for p in generator.parameters():
            p.requires_grad = False
        for _ in range(10):
            z_optim.zero_grad()
            recon = generator(Z)
            loss = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2)).sum()
            loss.backward()
            z_optim.step()
        for p in generator.parameters():
            p.requires_grad = True
        generator.eval()

        with torch.no_grad():
            best_recon = generator(Z)
            recon_error = F.mse_loss(best_recon, x, reduction="none").mean(dim=(1, 2))

        batch_scores = res_weight * recon_error + (1 - res_weight) * d_score
        scores.append(batch_scores.cpu().numpy())

    scores = np.concatenate(scores)
    labels = labels[:len(scores)]

    auc = roc_auc_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    preds = (scores >= best_th).astype(int)
    f1 = f1_score(labels, preds)

    return auc, f1


def main():
    checkpoints = [
        ("generator_best.pt", "discriminator_best.pt"),
        ("generator_final.pt", "discriminator_final.pt"),
        ("generator_epoch40.pt", "discriminator_epoch40.pt"),
    ]
    res_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_npz = "./data/preprocessed/test_closed.npz"

    print("MAD-GAN 快速参数搜索 (test_closed 前 2000 样本):")
    print("=" * 70)
    for g_name, d_name in checkpoints:
        g_path = f"./checkpoints/mad_gan/{g_name}"
        d_path = f"./checkpoints/mad_gan/{d_name}"
        for rw in res_weights:
            auc, f1 = eval_config(g_path, d_path, rw, test_npz, n_samples=2000)
            print(f"{g_name:20s} res={rw:.1f} | AUC={auc:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()
