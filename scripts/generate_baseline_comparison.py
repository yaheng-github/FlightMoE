"""Generate baseline comparison table and figure from existing experiment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parents[1]

# Load results
v1_path = ROOT / "experiments/router_v1/router_v1_results.json"
gdn_path = ROOT / "experiments/gdn_ablation_fast/gdn_graph_ablation_results.json"
v2_path = ROOT / "experiments/server_downloads/flightmoe_v2_results.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(data, model_name):
    """Extract val/test_closed/test_open AUC and F1."""
    row = {"Model": model_name}
    splits = data.get("splits", data)
    for split in ["val", "test_closed", "test_open"]:
        split_data = splits[split]
        if "mlp_router" in split_data:
            metrics = split_data["mlp_router"]
        elif "overall" in split_data:
            metrics = split_data["overall"]
        elif "global" in split_data:
            metrics = split_data["global"]
        else:
            metrics = split_data
        row[f"{split}_auc"] = metrics["auc_roc"]
        row[f"{split}_f1"] = metrics["f1"]
    return row


def main():
    v1_data = load_json(v1_path)
    gdn_data = load_json(gdn_path)
    v2_data = load_json(v2_path)

    rows = [
        extract_metrics(v1_data, "FlightMoE v1 (MLP Router)"),
        {
            "Model": "GDN (phase graph)",
            "val_auc": gdn_data["phase"]["val"]["global"]["auc_roc"],
            "val_f1": gdn_data["phase"]["val"]["global"]["f1"],
            "test_closed_auc": gdn_data["phase"]["test_closed"]["global"]["auc_roc"],
            "test_closed_f1": gdn_data["phase"]["test_closed"]["global"]["f1"],
            "test_open_auc": gdn_data["phase"]["test_open"]["global"]["auc_roc"],
            "test_open_f1": gdn_data["phase"]["test_open"]["global"]["f1"],
        },
        {
            "Model": "GDN (full graph)",
            "val_auc": gdn_data["full"]["val"]["global"]["auc_roc"],
            "val_f1": gdn_data["full"]["val"]["global"]["f1"],
            "test_closed_auc": gdn_data["full"]["test_closed"]["global"]["auc_roc"],
            "test_closed_f1": gdn_data["full"]["test_closed"]["global"]["f1"],
            "test_open_auc": gdn_data["full"]["test_open"]["global"]["auc_roc"],
            "test_open_f1": gdn_data["full"]["test_open"]["global"]["f1"],
        },
        extract_metrics(v2_data, "FlightMoE v2 (Full)"),
    ]

    # Add Stage1 Ranking variant if available
    ranking_path = ROOT / "experiments/server_downloads/ablation_stage1_ranking/flightmoe_v2_results.json"
    if ranking_path.exists():
        ranking_data = load_json(ranking_path)
        rows.append(extract_metrics(ranking_data, "FlightMoE v2 (Stage1 Ranking)"))

    # Markdown table
    md_lines = [
        "# Baseline Comparison\n",
        "| Model | Val AUC | Val F1 | Closed AUC | Closed F1 | Open AUC | Open F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['Model']} | {row['val_auc']:.4f} | {row['val_f1']:.4f} | "
            f"{row['test_closed_auc']:.4f} | {row['test_closed_f1']:.4f} | "
            f"{row['test_open_auc']:.4f} | {row['test_open_f1']:.4f} |"
        )
    md_path = ROOT / "paper_materials/baseline_comparison.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    # LaTeX table
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison with baseline methods on RflyMAD.}",
        "\\label{tab:baseline}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Val AUC & Val F1 & Closed AUC & Closed F1 & Open AUC & Open F1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['Model']} & {row['val_auc']:.4f} & {row['val_f1']:.4f} & "
            f"{row['test_closed_auc']:.4f} & {row['test_closed_f1']:.4f} & "
            f"{row['test_open_auc']:.4f} & {row['test_open_f1']:.4f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path = ROOT / "paper_materials/baseline_comparison.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Wrote {tex_path}")

    # Bar chart
    models = [row["Model"].replace("FlightMoE ", "").replace(" (", "\n(").replace(")", ")") for row in rows]
    closed_aucs = [row["test_closed_auc"] for row in rows]
    open_aucs = [row["test_open_auc"] for row in rows]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, closed_aucs, width, label="Closed-set AUC")
    bars2 = ax.bar(x + width / 2, open_aucs, width, label="Open-set AUC")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Baseline Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend()
    ax.set_ylim(0.85, 1.0)
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
            fontsize=7,
        )

    fig.tight_layout()
    fig_path = ROOT / "paper_materials/figures/baseline_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
