"""Analyze expert activation patterns per fault type.

Usage:
    python scripts/analyze_expert_per_fault.py \
        --npz experiments/server_downloads/flightmoe_v2_test_closed.npz \
        --outdir paper_materials/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.experiment_utils import FAULT_TYPE_ID_MAP, parse_case_id

EXPERT_NAMES = ["Burst", "Drift", "Spectral", "Consistency"]


def main():
    parser = argparse.ArgumentParser(description="Analyze expert weights per fault type")
    parser.add_argument("--npz", type=str, required=True, help="Path to saved score npz")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test_closed")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    case_ids = data["case_id"]
    expert_weights = data["expert_weights"]  # [N, 4]

    fault_types = np.array([parse_case_id(int(cid))[2] for cid in case_ids])

    # Compute mean weights per fault type
    mean_weights = {}
    for ft_id, ft_name in FAULT_TYPE_ID_MAP.items():
        mask = fault_types == ft_id
        if mask.sum() > 0:
            mean_weights[ft_name] = expert_weights[mask].mean(axis=0)

    names = list(mean_weights.keys())
    weights = np.array([mean_weights[n] for n in names])  # [K, 4]

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bottom = np.zeros(len(names))
    for i, exp_name in enumerate(EXPERT_NAMES):
        ax.bar(names, weights[:, i], bottom=bottom, label=exp_name)
        bottom += weights[:, i]

    ax.set_ylabel("Mean Expert Weight")
    ax.set_ylim([0, 1.05])
    ax.set_title(f"Expert Activation by Fault Type ({args.split})")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"expert_per_fault_{args.split}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Save table
    table_path = outdir / f"expert_per_fault_{args.split}.md"
    lines = ["| Fault Type | " + " | ".join(EXPERT_NAMES) + " |", "|" + "---|" * (len(EXPERT_NAMES) + 1)]
    for n in names:
        vals = [f"{v:.3f}" for v in mean_weights[n]]
        lines.append(f"| {n} | " + " | ".join(vals) + " |")
    table_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved figure: {out_path}")
    print(f"Saved table: {table_path}")


if __name__ == "__main__":
    main()
