"""Visualize FlightMoE v2 results.

Usage:
    python scripts/visualize_results.py --results experiments/flightmoe_v2/flightmoe_v2_results.json --outdir experiments/flightmoe_v2/figures

Supports multiple result files for ablation comparison:
    python scripts/visualize_results.py \
        --results full.json --label Full \
        --results ablation_no_phase.json --label "w/o Phase" \
        --outdir figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PHASE_NAMES = ["Manual", "Takeoff", "Climb", "Cruise", "Descent", "Land"]
EXPERT_NAMES = ["Burst", "Drift", "Spectral", "Consistency"]


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_router_heatmap(phase_weights, out_path, title="Router weights per phase"):
    """phase_weights: dict phase_idx -> list of 4 weights"""
    phases = [str(i) for i in range(6)]
    data = np.array([phase_weights.get(p, [0.25] * 4) for p in phases])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(EXPERT_NAMES)))
    ax.set_xticklabels(EXPERT_NAMES)
    ax.set_yticks(np.arange(len(PHASE_NAMES)))
    ax.set_yticklabels(PHASE_NAMES)
    ax.set_title(title)

    for i in range(len(PHASE_NAMES)):
        for j in range(len(EXPERT_NAMES)):
            text = ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, label="Weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_per_phase_metrics(per_phase, out_path, title="Per-phase AUC"):
    phases = [str(i) for i in range(6)]
    aucs = [per_phase.get(p, {}).get("auc_roc", 0.0) for p in phases]
    f1s = [per_phase.get(p, {}).get("f1", 0.0) for p in phases]

    x = np.arange(len(PHASE_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, aucs, width, label="AUC")
    ax.bar(x + width / 2, f1s, width, label="F1")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_NAMES)
    ax.set_ylim([0, 1.05])
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fault_type_metrics(per_fault_type, out_path, title="Per-fault-type AUC"):
    names = list(per_fault_type.keys())
    aucs = [per_fault_type[n]["auc_roc"] for n in names]
    f1s = [per_fault_type[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax.bar(x - width / 2, aucs, width, label="AUC")
    ax.bar(x + width / 2, f1s, width, label="F1")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim([0, 1.05])
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_ablation_comparison(results_dict, out_path, split="test_closed"):
    """results_dict: label -> results JSON dict"""
    labels = list(results_dict.keys())
    aucs = [results_dict[l][split]["overall"]["auc_roc"] for l in labels]
    f1s = [results_dict[l][split]["overall"]["f1"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars1 = ax.bar(x - width / 2, aucs, width, label="AUC")
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1")

    # Annotate values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim([0, 1.05])
    ax.set_title(f"Ablation comparison on {split}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize FlightMoE v2 results")
    parser.add_argument("--results", action="append", required=True, help="Path to results JSON")
    parser.add_argument("--label", action="append", help="Label for each results file")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for figures")
    parser.add_argument("--split", type=str, default="test_closed", help="Split for ablation comparison")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.label is None:
        args.label = [f"run{i+1}" for i in range(len(args.results))]
    assert len(args.results) == len(args.label)

    all_results = {}
    for path, label in zip(args.results, args.label):
        results = load_results(path)
        all_results[label] = results

        # Per-split plots for the first (full) result set
        if len(args.results) == 1 or label == args.label[0]:
            for split_name, split_res in results.items():
                safe_split = split_name.replace("/", "_")
                plot_router_heatmap(
                    split_res["router_phase_mean_weights"],
                    outdir / f"router_heatmap_{safe_split}.png",
                    title=f"Router weights per phase ({split_name})",
                )
                plot_per_phase_metrics(
                    split_res["per_phase"],
                    outdir / f"per_phase_metrics_{safe_split}.png",
                    title=f"Per-phase metrics ({split_name})",
                )
                if split_res.get("per_fault_type"):
                    plot_fault_type_metrics(
                        split_res["per_fault_type"],
                        outdir / f"fault_type_metrics_{safe_split}.png",
                        title=f"Per-fault-type metrics ({split_name})",
                    )

    # Ablation comparison
    if len(all_results) > 1:
        plot_ablation_comparison(
            all_results,
            outdir / f"ablation_comparison_{args.split}.png",
            split=args.split,
        )

    print(f"Figures saved to {outdir}")


if __name__ == "__main__":
    main()
