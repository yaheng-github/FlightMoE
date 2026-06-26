"""Export window-aligned GANomaly anomaly scores for FlightMoE v1."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))

from dataset import SpectralDataset
from networks import NetG
from utils.experiment_utils import save_score_npz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_scores(netg, dataloader):
    netg.eval()
    all_scores = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(DEVICE)
            _, latent_i, latent_o = netg(data)
            error = torch.mean(torch.pow(latent_i - latent_o, 2), dim=(1, 2, 3))
            all_scores.append(error.cpu().numpy())
    return np.concatenate(all_scores).astype(np.float32)


def export_split(args, split_name, npz_path):
    netg = NetG(isize=args.image_size, nz=args.nz, nc=3, ndf=args.ndf).to(DEVICE)
    checkpoint = torch.load(args.netg, map_location=DEVICE)
    netg.load_state_dict(checkpoint["state_dict"])
    dataset = SpectralDataset(npz_path, normal_only=False, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    scores = compute_scores(netg, dataloader)
    out_path = Path(args.output_dir) / f"ganomaly_{split_name}.npz"
    save_score_npz(str(out_path), npz_path, "ganomaly", split_name, scores)
    print(f"[SAVED] {out_path} ({len(scores)} scores)")


def main():
    parser = argparse.ArgumentParser(description="Export GANomaly scores")
    parser.add_argument("--val_npz", type=str, default="./data/preprocessed/val.npz")
    parser.add_argument("--test_closed", type=str, default="./data/preprocessed/test_closed.npz")
    parser.add_argument("--test_open", type=str, default="./data/preprocessed/test_open.npz")
    parser.add_argument("--netg", type=str, default="./checkpoints/ganomaly/netG_best.pth")
    parser.add_argument("--output_dir", type=str, default="./experiments/scores")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for split_name, npz_path in [("val", args.val_npz), ("test_closed", args.test_closed), ("test_open", args.test_open)]:
        export_split(args, split_name, npz_path)


if __name__ == "__main__":
    main()
