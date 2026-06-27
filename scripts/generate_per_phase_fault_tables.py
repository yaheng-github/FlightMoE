"""Generate per-phase and per-fault-type LaTeX tables from v2 results JSON."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
RESULTS_PATH = ROOT / "experiments/server_downloads/ablation_stage1_ranking/flightmoe_v2_results.json"
OUTPUT_DIR = ROOT / "paper_materials"

PHASE_NAMES = ["hover", "waypoint", "velocity", "circling", "acce", "dece"]
PHASE_LABELS = ["Hover", "Waypoint", "Velocity", "Circling", "Accel", "Decel"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_latex_table(rows, columns, caption, label):
    """rows: list of dicts with keys matching columns."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "cc" * (len(columns) - 1) + "}",
        "\\toprule",
    ]
    header = " & ".join(columns) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for row in rows:
        vals = [row.get(col, "") for col in columns]
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    data = load_json(RESULTS_PATH)

    # Per-phase tables for each split
    for split in ["val", "test_closed", "test_open"]:
        rows = []
        for idx, phase_name in enumerate(PHASE_NAMES):
            phase_data = data[split]["per_phase"][str(idx)]
            rows.append(
                {
                    "Phase": PHASE_LABELS[idx],
                    "AUC": f"{phase_data['auc_roc']:.4f}",
                    "F1": f"{phase_data['f1']:.4f}",
                    "P": f"{phase_data['precision']:.4f}",
                    "R": f"{phase_data['recall']:.4f}",
                    "N": str(phase_data["n_samples"]),
                }
            )
        tex = make_latex_table(
            rows,
            ["Phase", "AUC", "F1", "P", "R", "N"],
            f"Per-phase metrics on {split.replace('_', '\\\\_')}.",
            f"tab:phase_{split}",
        )
        out_path = OUTPUT_DIR / f"per_phase_{split}_table.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"Wrote {out_path}")

    # Per-fault-type tables for closed and open
    for split in ["test_closed", "test_open"]:
        rows = []
        for fault_name, fault_data in data[split]["per_fault_type"].items():
            rows.append(
                {
                    "Fault Type": fault_name.replace("_", " "),
                    "AUC": f"{fault_data['auc_roc']:.4f}",
                    "F1": f"{fault_data['f1']:.4f}",
                    "P": f"{fault_data['precision']:.4f}",
                    "R": f"{fault_data['recall']:.4f}",
                    "N": str(fault_data["n_samples"]),
                }
            )
        tex = make_latex_table(
            rows,
            ["Fault Type", "AUC", "F1", "P", "R", "N"],
            f"Per-fault-type metrics on {split.replace('_', '\\\\_')}.",
            f"tab:fault_{split}",
        )
        out_path = OUTPUT_DIR / f"per_fault_{split}_table.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
