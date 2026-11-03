#!/usr/bin/env python3
"""Build the final comparison table from locked, measured artifacts."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final_test"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    lock = load("results/test_protocol.lock.json")
    source = [
        ("YOLOv3-tiny (FP32 reproduction)", "yolov3_tiny_reproduced",
         "results/baseline_reproduction/val/metrics.json",
         "results/final_test/yolov3_tiny_reproduced/metrics.json",
         "results/benchmarks/yolov3_tiny_val_batch1.json",
         "results/baseline_reproduction/training.json"),
        ("YOLO11n", "yolo11n",
         "second_approach/results/common_evaluation/metrics.json",
         "results/final_test/yolo11n/metrics.json",
         "results/benchmarks/yolo11n_val_batch1.json",
         "second_approach/results/training_runtime.json"),
    ]
    rows = []
    for name, key, validation_path, test_path, timing_path, training_path in source:
        val, test, timing, training = map(load,
            (validation_path, test_path, timing_path, training_path))
        selected = lock[key]
        checkpoint = ROOT / selected["checkpoint"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected["checkpoint_sha256"]
        assert val["split"] == "val" and val["images"] == 25
        assert test["split"] == "test" and test["images"] == 26
        assert test["operating_confidence"] == selected["operating_confidence"]
        assert "confidence_best_f1_iou50" not in test
        assert timing["split"] == "val" and timing["batch"] == 1
        rows.append({"model": name, "val_mAP50": val["mAP50"],
                     "val_mAP50_95": val["mAP50_95"],
                     "test_mAP50": test["mAP50"],
                     "test_mAP50_95": test["mAP50_95"],
                     "test_precision_at_val_threshold": test["precision_at_operating_confidence"],
                     "test_recall_at_val_threshold": test["recall_at_operating_confidence"],
                     "validation_selected_confidence": selected["operating_confidence"],
                     "batch1_val_ms_per_image": timing["sum_ms_per_image"],
                     "training_seconds": training["training_seconds"] if key.startswith("yolov3")
                         else training["wall_seconds"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
