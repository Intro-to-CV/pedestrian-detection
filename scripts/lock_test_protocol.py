#!/usr/bin/env python3
"""Freeze validation choices before either model sees test images."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/test_protocol.lock.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Protocol is already locked: {OUT}")
    ablation_path = ROOT / "results/ablation/summary.json"
    ablation = json.loads(ablation_path.read_text())
    baseline_path = ROOT / "results/baseline_reproduction/val/metrics.json"
    baseline = json.loads(baseline_path.read_text())
    assert ablation["split"] == baseline["split"] == "val"
    assert ablation["selected_seed"] == 42
    yolo_path = ROOT / "models/yolo11n_best.pt"
    baseline_checkpoint = ROOT / "models/yolov3_tiny_best.pt"
    assert sha(yolo_path) == ablation["selected_checkpoint_sha256"]
    reproduced = json.loads((ROOT / "results/baseline_reproduction/training.json").read_text())
    assert sha(baseline_checkpoint) == reproduced["checkpoint_sha256"]
    record = {"locked_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": "validation only; no test metrics consulted",
        "test_split_manifest_sha256": sha(ROOT / "data/splits/test.txt"),
        "evaluation": {"image_size": 640, "confidence_floor_for_ap": 0.001,
                       "nms_iou": 0.6, "max_detections": 300,
                       "precision": "FP32", "shared_evaluator": "scripts/evaluate.py"},
        "yolo11n": {"mosaic": ablation["selected_mosaic"], "seed": 42,
            "checkpoint": str(yolo_path.relative_to(ROOT)),
            "checkpoint_sha256": sha(yolo_path),
            "operating_confidence": ablation["selected_confidence"],
            "validation_metric_sha256": sha(ablation_path)},
        "yolov3_tiny_reproduced": {"checkpoint": str(baseline_checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha(baseline_checkpoint),
            "operating_confidence": baseline["confidence_best_f1_iou50"],
            "validation_metric_sha256": sha(baseline_path)}}
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
