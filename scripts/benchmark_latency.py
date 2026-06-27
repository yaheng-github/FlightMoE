"""Benchmark FlightMoE v2 inference latency."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from train_v2 import FlightMoEv2
from utils.config import load_config


def benchmark(batch_size, num_runs, device, cfg_path, ckpt_path):
    cfg = load_config(cfg_path)
    model = FlightMoEv2(cfg, ablation={}).to(device)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    temporal = torch.randn(batch_size, 128, 41, device=device)
    spectral = torch.randint(0, 256, (batch_size, 4, 17, 9, 3), device=device).float() / 255.0
    phase = torch.randint(0, 6, (batch_size,), device=device)
    modality_mask = torch.ones(batch_size, 41, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(temporal, spectral, phase, modality_mask)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(temporal, spectral, phase, modality_mask)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # ms
    latency_per_batch = times.mean()
    latency_per_sample = latency_per_batch / batch_size
    throughput = batch_size / (latency_per_batch / 1000)

    return {
        "batch_size": batch_size,
        "latency_ms_mean": float(latency_per_batch),
        "latency_ms_std": float(times.std()),
        "latency_per_sample_ms": float(latency_per_sample),
        "throughput_samples_per_s": float(throughput),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/flightmoe_v2.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint or 'random init'}")
    print("-" * 80)
    print(f"{'Batch':>8} {'Batch Latency (ms)':>22} {'Sample Latency (ms)':>24} {'Throughput (samp/s)':>22}")
    print("-" * 80)

    results = []
    for bs in args.batch_sizes:
        r = benchmark(bs, args.num_runs, args.device, args.config, args.checkpoint)
        results.append(r)
        print(f"{r['batch_size']:>8} {r['latency_ms_mean']:>12.3f} ± {r['latency_ms_std']:<6.3f} {r['latency_per_sample_ms']:>22.3f} {r['throughput_samples_per_s']:>22.1f}")


if __name__ == "__main__":
    main()
