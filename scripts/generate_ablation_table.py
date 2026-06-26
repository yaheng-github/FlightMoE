"""Generate ablation comparison table from results JSONs.

Usage:
    python scripts/generate_ablation_table.py \
        --results full.json --label Full \
        --results ablation_no_phase.json --label "w/o Phase" \
        --out table.md
"""

import argparse
import json
from pathlib import Path


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate ablation comparison table")
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--out", type=str, default="ablation_table.md")
    args = parser.parse_args()

    assert len(args.results) == len(args.label)

    rows = []
    for path, label in zip(args.results, args.label):
        res = load_results(path)
        row = {"Model": label}
        for split in ["val", "test_closed", "test_open"]:
            overall = res[split]["overall"]
            row[f"{split}_auc"] = overall["auc_roc"]
            row[f"{split}_f1"] = overall["f1"]
        rows.append(row)

    cols = ["Model", "val_auc", "val_f1", "test_closed_auc", "test_closed_f1", "test_open_auc", "test_open_f1"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"

    lines = [header, sep]
    for row in rows:
        vals = [row[c] if isinstance(row[c], str) else f"{row[c]:.4f}" for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    out_text = "\n".join(lines)
    Path(args.out).write_text(out_text, encoding="utf-8")
    print(out_text)
    print(f"\nTable saved to {args.out}")


if __name__ == "__main__":
    main()
