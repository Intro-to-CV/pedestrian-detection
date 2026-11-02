#!/usr/bin/env python3
"""Summarize the predeclared 2-seed mosaic ablation on validation only."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ablation"
SEEDS = (42, 43)
CONDITIONS = ("on", "off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=OUT)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    records = {}
    for mosaic in CONDITIONS:
        records[mosaic] = []
        for seed in SEEDS:
            record = json.loads((input_root / f"mosaic_{mosaic}_seed{seed}/result.json").read_text())
            assert record["seed"] == seed and record["mosaic"] == mosaic
            records[mosaic].append(record)
            metrics = record["metrics"]
            rows.append({"mosaic": mosaic, "seed": seed,
                         "mAP50": metrics["mAP50"], "mAP50_95": metrics["mAP50_95"],
                         "precision": metrics["precision_best_f1_iou50"],
                         "recall": metrics["recall_best_f1_iou50"],
                         "training_seconds": record["training_seconds"]})
    with (output_root / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for mosaic in CONDITIONS:
        summary[mosaic] = {}
        for metric in ("mAP50", "mAP50_95", "precision_best_f1_iou50", "recall_best_f1_iou50"):
            values = [r["metrics"][metric] for r in records[mosaic]]
            summary[mosaic][metric] = {"mean": mean(values), "min": min(values), "max": max(values)}
    # Fixed before test: choose the condition by two-seed mean mAP50:95;
    # if tied, mean mAP50. Use seed 42 as the representative checkpoint.
    chosen = max(CONDITIONS, key=lambda x: (summary[x]["mAP50_95"]["mean"],
                                            summary[x]["mAP50"]["mean"]))
    selected = records[chosen][0]
    result = {"split": "val", "seeds": list(SEEDS), "summary": summary,
              "selection_rule": "higher 2-seed mean mAP50_95, then mean mAP50; seed 42 checkpoint",
              "selected_mosaic": chosen,
              "selected_seed": 42,
              "selected_checkpoint_sha256": selected["checkpoint_sha256"],
              "selected_confidence": selected["metrics"]["confidence_best_f1_iou50"]}
    (output_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
