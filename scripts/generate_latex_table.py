"""Generate LaTeX tables from results JSONs.

Usage:
    python scripts/generate_latex_table.py \
        --results full.json --label Full \
        --results ablation_no_phase.json --label "w/o Phase" \
        --out table.tex
"""

import argparse
import json
from pathlib import Path


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX result table")
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--out", type=str, default="table.tex")
    args = parser.parse_args()

    assert len(args.results) == len(args.label)

    rows = []
    for path, label in zip(args.results, args.label):
        res = load_results(path)
        row = [label]
        for split in ["val", "test_closed", "test_open"]:
            overall = res[split]["overall"]
            row.append(f"{overall['auc_roc']:.4f}")
            row.append(f"{overall['f1']:.4f}")
        rows.append(row)

    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Val AUC & Val F1 & Closed AUC & Closed F1 & Open AUC & Open F1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])

    out_text = "\n".join(lines)
    Path(args.out).write_text(out_text, encoding="utf-8")
    print(out_text)
    print(f"\nLaTeX table saved to {args.out}")


if __name__ == "__main__":
    main()
