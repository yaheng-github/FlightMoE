"""
GANomaly 训练脚本（适配 RflyMAD spectral 图像数据）
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloader
from networks import NetG, NetD, weights_init

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_ganomaly(
    train_npz: str,
    val_npz: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 2e-4,
    nz: int = 64,
    ndf: int = 64,
    w_adv: float = 1.0,
    w_con: float = 50.0,
    w_enc: float = 1.0,
    image_size: int = 64,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Data
    train_dl = get_dataloader(train_npz, batch_size=batch_size, normal_only=True, shuffle=True, image_size=image_size)
    val_dl = get_dataloader(val_npz, batch_size=batch_size, normal_only=False, shuffle=False, image_size=image_size)

    # Models
    class Opt:
        pass
    opt = Opt()
    opt.isize = image_size
    opt.nc = 3
    opt.nz = nz
    opt.ngf = ndf
    opt.ndf = ndf
    opt.ngpu = 1
    opt.extralayers = 0

    netg = NetG(opt).to(DEVICE)
    netd = NetD(opt).to(DEVICE)
    netg.apply(weights_init)
    netd.apply(weights_init)

    # Losses
    l_adv = nn.MSELoss()
    l_con = nn.L1Loss()
    l_enc = nn.MSELoss()
    l_bce = nn.BCELoss()

    # Optimizers
    optimizer_g = optim.Adam(netg.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(netd.parameters(), lr=lr, betas=(0.5, 0.999))

    real_label = torch.ones(batch_size, device=DEVICE)
    fake_label = torch.zeros(batch_size, device=DEVICE)

    best_auc = 0.0

    for epoch in range(epochs):
        netg.train()
        netd.train()
        epoch_iter = 0

        for data, _ in tqdm(train_dl, leave=False, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(DEVICE)
            bs = data.size(0)
            if bs != batch_size:
                continue

            # ---- Forward G ----
            fake, latent_i, latent_o = netg(data)

            # ---- Forward D ----
            pred_real, feat_real = netd(data)
            pred_fake, feat_fake = netd(fake.detach())

            # ---- Update D ----
            optimizer_d.zero_grad()
            err_d_real = l_bce(pred_real, real_label[:bs])
            err_d_fake = l_bce(pred_fake, fake_label[:bs])
            err_d = (err_d_real + err_d_fake) * 0.5
            err_d.backward()
            optimizer_d.step()

            # Re-init D if loss too small
            if err_d.item() < 1e-5:
                netd.apply(weights_init)

            # ---- Update G ----
            optimizer_g.zero_grad()
            pred_real, feat_real = netd(data)
            pred_fake, feat_fake = netd(fake)
            err_g_adv = l_adv(feat_real, feat_fake)
            err_g_con = l_con(fake, data)
            err_g_enc = l_enc(latent_o, latent_i)
            err_g = err_g_adv * w_adv + err_g_con * w_con + err_g_enc * w_enc
            err_g.backward()
            optimizer_g.step()

            epoch_iter += bs

        # ---- Validation ----
        netg.eval()
        netd.eval()
        an_scores = []
        gt_labels = []

        with torch.no_grad():
            for data, labels in val_dl:
                data = data.to(DEVICE)
                bs = data.size(0)
                fake, latent_i, latent_o = netg(data)
                error = torch.mean(torch.pow(latent_i - latent_o, 2), dim=(1, 2, 3))
                an_scores.append(error.cpu())
                gt_labels.append(labels)

        an_scores = torch.cat(an_scores)
        gt_labels = torch.cat(gt_labels)

        # Normalize scores
        an_scores = (an_scores - an_scores.min()) / (an_scores.max() - an_scores.min())

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(gt_labels.numpy(), an_scores.numpy())
        print(f"Epoch [{epoch+1}/{epochs}] AUC: {auc:.4f}  D_loss: {err_d.item():.4f}  G_loss: {err_g.item():.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save({'epoch': epoch + 1, 'state_dict': netg.state_dict()}, out_path / "netG_best.pth")
            torch.save({'epoch': epoch + 1, 'state_dict': netd.state_dict()}, out_path / "netD_best.pth")
            print(f"[Best] Saved at epoch {epoch+1}, AUC: {auc:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': netg.state_dict()}, out_path / f"netG_epoch{epoch+1}.pth")
            torch.save({'epoch': epoch + 1, 'state_dict': netd.state_dict()}, out_path / f"netD_epoch{epoch+1}.pth")

    torch.save({'epoch': epochs, 'state_dict': netg.state_dict()}, out_path / "netG_final.pth")
    torch.save({'epoch': epochs, 'state_dict': netd.state_dict()}, out_path / "netD_final.pth")
    print(f"\n[DONE] Best AUC: {best_auc:.4f}, Checkpoints saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GANomaly Baseline Training")
    parser.add_argument("--train_npz", type=str, default="./data/preprocessed/train.npz")
    parser.add_argument("--val_npz", type=str, default="./data/preprocessed/val.npz")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/ganomaly")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ganomaly(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        nz=args.nz,
        ndf=args.ndf,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
