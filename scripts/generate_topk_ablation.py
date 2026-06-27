"""Generate Top-k ablation comparison table and figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "paper_materials"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    results = {
        1: load_json(ROOT / "experiments/server_downloads/ablation_topk_1/flightmoe_v2_results.json"),
        2: load_json(ROOT / "experiments/server_downloads/ablation_stage1_ranking/flightmoe_v2_results.json"),
        3: load_json(ROOT / "experiments/server_downloads/ablation_topk_3/flightmoe_v2_results.json"),
        4: load_json(ROOT / "experiments/server_downloads/ablation_topk_4/flightmoe_v2_results.json"),
    }

    rows = []
    for k, data in results.items():
        rows.append({
            "k": str(k),
            "val_auc": data["val"]["overall"]["auc_roc"],
            "val_f1": data["val"]["overall"]["f1"],
            "closed_auc": data["test_closed"]["overall"]["auc_roc"],
            "closed_f1": data["test_closed"]["overall"]["f1"],
            "open_auc": data["test_open"]["overall"]["auc_roc"],
            "open_f1": data["test_open"]["overall"]["f1"],
        })

    # Markdown table
    md_lines = [
        "# Top-k Ablation Results\n",
        "| k | Val AUC | Val F1 | Closed AUC | Closed F1 | Open AUC | Open F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['k']} | {row['val_auc']:.4f} | {row['val_f1']:.4f} | "
            f"{row['closed_auc']:.4f} | {row['closed_f1']:.4f} | "
            f"{row['open_auc']:.4f} | {row['open_f1']:.4f} |"
        )
    md_path = OUTPUT_DIR / "topk_ablation.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    # LaTeX table
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Top-$k$ ablation study.}",
        "\\label{tab:topk}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "$k$ & Val AUC & Val F1 & Closed AUC & Closed F1 & Open AUC & Open F1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['k']} & {row['val_auc']:.4f} & {row['val_f1']:.4f} & "
            f"{row['closed_auc']:.4f} & {row['closed_f1']:.4f} & "
            f"{row['open_auc']:.4f} & {row['open_f1']:.4f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path = OUTPUT_DIR / "topk_ablation.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Wrote {tex_path}")

    # Figure
    ks = [str(k) for k in results.keys()]
    closed_aucs = [r["closed_auc"] for r in rows]
    open_aucs = [r["open_auc"] for r in rows]

    x = np.arange(len(ks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, closed_aucs, width, label="Closed-set AUC")
    bars2 = ax.bar(x + width / 2, open_aucs, width, label="Open-set AUC")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Effect of Top-k Sparse Routing")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.legend()
    ax.set_ylim(0.7, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig_path = OUTPUT_DIR / "figures/topk_ablation.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
