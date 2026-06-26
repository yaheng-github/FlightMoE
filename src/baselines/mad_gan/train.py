"""
MAD-GAN 训练脚本（适配 RflyMAD 数据，自包含训练循环）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "third_party" / "madgan-pytorch"))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from madgan.models import Generator, Discriminator
from dataset import get_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeviceLatentIterator:
    """Device-aware 噪声迭代器"""
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device
    def __iter__(self):
        return self
    def __next__(self):
        return torch.randn(*self.shape, device=self.device)


class DeviceDataLoader:
    """自动将 batch 送到 device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield b.to(self.device)
    def __len__(self):
        return len(self.dl)


def train_one_epoch(generator, discriminator, train_dl, latent_iter,
                    g_optim, d_optim, criterion, epoch, log_every=50):
    generator.train()
    discriminator.train()

    for i, (real, z) in enumerate(zip(train_dl, latent_iter)):
        bs = real.size(0)
        real_labels = torch.ones(bs, device=real.device)
        fake_labels = torch.zeros(bs, device=real.device)

        fake = generator(z)

        # ---- Update Discriminator ----
        d_optim.zero_grad()
        real_out = discriminator(real).mean(dim=(1, 2)).clamp(1e-7, 1 - 1e-7)    # [batch]
        fake_out = discriminator(fake.detach()).mean(dim=(1, 2)).clamp(1e-7, 1 - 1e-7)  # [batch]

        d_loss = criterion(real_out, real_labels) + criterion(fake_out, fake_labels)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        d_optim.step()

        # ---- Update Generator ----
        g_optim.zero_grad()
        fake_out = discriminator(fake).mean(dim=(1, 2)).clamp(1e-7, 1 - 1e-7)
        g_loss = criterion(fake_out, real_labels)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        g_optim.step()

        if (i + 1) % log_every == 0:
            print(f"Epoch [{epoch}] Step [{i}] D_loss:{d_loss.item():.4f} G_loss:{g_loss.item():.4f}")


@torch.no_grad()
def evaluate_epoch(generator, discriminator, val_dl, latent_iter, criterion):
    generator.eval()
    discriminator.eval()

    total_d_loss = 0.0
    total_g_loss = 0.0
    n = 0

    for real, z in zip(val_dl, latent_iter):
        bs = real.size(0)
        real_labels = torch.ones(bs, device=real.device)
        fake_labels = torch.zeros(bs, device=real.device)

        fake = generator(z)
        real_out = discriminator(real).mean(dim=(1, 2)).clamp(1e-7, 1 - 1e-7)
        fake_out = discriminator(fake).mean(dim=(1, 2)).clamp(1e-7, 1 - 1e-7)

        d_loss = criterion(real_out, real_labels) + criterion(fake_out, fake_labels)
        g_loss = criterion(fake_out, real_labels)

        total_d_loss += d_loss.item() * bs
        total_g_loss += g_loss.item() * bs
        n += bs

    return total_d_loss / n, total_g_loss / n


def train_madgan(
    train_npz: str,
    val_npz: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    hidden_dim: int = 512,
    latent_dim: int = 32,
    window_size: int = 128,
    n_lstm_layers: int = 2,
    seed: int = 42,
):
    set_seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    feature_dim = 41

    # Data
    train_dl_raw = get_dataloader(train_npz, batch_size=batch_size, normal_only=True, shuffle=True)
    val_dl_raw = get_dataloader(val_npz, batch_size=batch_size, normal_only=False, shuffle=False)
    train_dl = DeviceDataLoader(train_dl_raw, DEVICE)
    val_dl = DeviceDataLoader(val_dl_raw, DEVICE)

    # Models
    generator = Generator(
        latent_space_dim=latent_dim,
        hidden_units=hidden_dim,
        output_dim=feature_dim,
        n_lstm_layers=n_lstm_layers,
    ).to(DEVICE)

    discriminator = Discriminator(
        input_dim=feature_dim,
        hidden_units=hidden_dim,
        n_lstm_layers=n_lstm_layers,
        add_batch_mean=False,
    ).to(DEVICE)

    g_optim = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    latent_iter = DeviceLatentIterator(shape=[batch_size, window_size, latent_dim], device=DEVICE)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")
        train_one_epoch(generator, discriminator, train_dl, latent_iter,
                        g_optim, d_optim, criterion, epoch, log_every=50)

        val_d_loss, val_g_loss = evaluate_epoch(generator, discriminator, val_dl, latent_iter, criterion)
        val_loss = val_d_loss + val_g_loss
        print(f"[Val] D_loss:{val_d_loss:.4f} G_loss:{val_g_loss:.4f} Total:{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            generator.save(out_path / "generator_best.pt")
            discriminator.save(out_path / "discriminator_best.pt")
            print(f"[Best] Saved at epoch {epoch + 1}")

        if (epoch + 1) % 10 == 0:
            generator.save(out_path / f"generator_epoch{epoch + 1}.pt")
            discriminator.save(out_path / f"discriminator_epoch{epoch + 1}.pt")

    generator.save(out_path / "generator_final.pt")
    discriminator.save(out_path / "discriminator_final.pt")
    print(f"\n[DONE] Checkpoints saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MAD-GAN Baseline Training")
    parser.add_argument("--train_npz", type=str, default="./data/preprocessed/train.npz")
    parser.add_argument("--val_npz", type=str, default="./data/preprocessed/val.npz")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mad_gan")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_madgan(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        window_size=args.window_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
