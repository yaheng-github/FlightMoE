"""Export window-aligned MAD-GAN anomaly scores for FlightMoE v1."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[3] / "third_party" / "madgan-pytorch"))
sys.path.insert(0, str(Path(__file__).parents[2]))

from madgan.models import Discriminator, Generator
from utils.experiment_utils import save_score_npz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.temporal = data["temporal"].astype(np.float32)

    def __getitem__(self, index):
        return torch.from_numpy(self.temporal[index])

    def __len__(self):
        return len(self.temporal)


def compute_scores(
    generator,
    discriminator,
    dataloader,
    latent_dim=32,
    res_weight=0.9,
    z_steps=10,
    z_lr=0.1,
    progress_every=50,
):
    generator.eval()
    discriminator.eval()
    all_scores = []
    all_recon = []
    all_disc = []

    n_batches = len(dataloader)
    for batch_idx, x in enumerate(dataloader, start=1):
        x = x.to(DEVICE)
        bs = x.size(0)

        with torch.no_grad():
            d_score = discriminator(x).mean(dim=(1, 2))

        generator.train()
        for p in generator.parameters():
            p.requires_grad = False
        z = torch.empty((bs, x.size(1), latent_dim), device=DEVICE, requires_grad=True)
        nn.init.normal_(z, std=0.05)
        z_optim = torch.optim.RMSprop([z], lr=z_lr)
        for _ in range(z_steps):
            z_optim.zero_grad()
            recon = generator(z)
            loss = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2)).sum()
            loss.backward()
            z_optim.step()
        for p in generator.parameters():
            p.requires_grad = True
        generator.eval()

        with torch.no_grad():
            recon = generator(z)
            recon_error = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
            score = res_weight * recon_error + (1 - res_weight) * d_score

        all_scores.append(score.cpu().numpy())
        all_recon.append(recon_error.cpu().numpy())
        all_disc.append(d_score.cpu().numpy())
        if progress_every and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == n_batches):
            print(f"  batch {batch_idx}/{n_batches}", flush=True)

    return np.concatenate(all_scores), np.concatenate(all_recon), np.concatenate(all_disc)


def export_split(args, split_name, npz_path):
    generator = Generator.from_pretrained(args.generator, map_location=DEVICE).to(DEVICE)
    discriminator = Discriminator.from_pretrained(args.discriminator, map_location=DEVICE).to(DEVICE)
    dataloader = DataLoader(TemporalDataset(npz_path), batch_size=args.batch_size, shuffle=False, drop_last=False)
    scores, recon_error, disc_score = compute_scores(
        generator,
        discriminator,
        dataloader,
        latent_dim=args.latent_dim,
        res_weight=args.res_weight,
        z_steps=args.z_steps,
        z_lr=args.z_lr,
        progress_every=args.progress_every,
    )
    out_path = Path(args.output_dir) / f"madgan_{split_name}.npz"
    save_score_npz(
        str(out_path),
        npz_path,
        "madgan",
        split_name,
        scores,
        recon_error=recon_error.astype(np.float32),
        discriminator_score=disc_score.astype(np.float32),
        res_weight=np.array(args.res_weight, dtype=np.float32),
    )
    print(f"[SAVED] {out_path} ({len(scores)} scores)")


def main():
    parser = argparse.ArgumentParser(description="Export MAD-GAN scores")
    parser.add_argument("--val_npz", type=str, default="./data/preprocessed/val.npz")
    parser.add_argument("--test_closed", type=str, default="./data/preprocessed/test_closed.npz")
    parser.add_argument("--test_open", type=str, default="./data/preprocessed/test_open.npz")
    parser.add_argument("--generator", type=str, default="./checkpoints/mad_gan/generator_best.pt")
    parser.add_argument("--discriminator", type=str, default="./checkpoints/mad_gan/discriminator_best.pt")
    parser.add_argument("--output_dir", type=str, default="./experiments/scores")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--res_weight", type=float, default=0.9)
    parser.add_argument("--z_steps", type=int, default=10)
    parser.add_argument("--z_lr", type=float, default=0.1)
    parser.add_argument("--splits", type=str, default="val,test_closed,test_open", help="Comma-separated splits to export")
    parser.add_argument("--progress_every", type=int, default=50)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split_paths = {
        "val": args.val_npz,
        "test_closed": args.test_closed,
        "test_open": args.test_open,
    }
    for split_name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split_name not in split_paths:
            raise ValueError(f"Unknown split: {split_name}")
        print(f"[EXPORT] {split_name}", flush=True)
        npz_path = split_paths[split_name]
        export_split(args, split_name, npz_path)


if __name__ == "__main__":
    main()
