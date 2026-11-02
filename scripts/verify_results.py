#!/usr/bin/env python3
"""Rescore saved predictions and verify published numbers and checkpoint hashes."""

import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "data" if (ROOT / "data/processed/images/train").is_dir()
        else ROOT / "second_approach/data/data")


def compare(expected: dict, actual: dict):
    keys = ["mAP50", "mAP50_95"]
    if expected["split"] == "test":
        keys += ["precision_at_operating_confidence", "recall_at_operating_confidence"]
    else:
        keys += ["precision_best_f1_iou50", "recall_best_f1_iou50"]
    for key in keys:
        assert abs(expected[key] - actual[key]) < 1e-12, (key, expected[key], actual[key])
    assert expected["images"] == actual["images"]
    assert expected["ground_truth_instances"] == actual["ground_truth_instances"]


def main():
    lock = json.loads((ROOT / "results/test_protocol.lock.json").read_text())
    for model in ("yolo11n", "yolov3_tiny_reproduced"):
        selected = lock[model]
        checkpoint = ROOT / selected["checkpoint"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected["checkpoint_sha256"]
    assert hashlib.sha256((ROOT / "data/splits/test.txt").read_bytes()).hexdigest() == \
        lock["test_split_manifest_sha256"]
    cases = [
        ("val", ROOT / "second_approach/results/validation_predictions.json",
         ROOT / "second_approach/results/common_evaluation/metrics.json", None),
        ("val", ROOT / "results/baseline_reproduction/val/predictions.json",
         ROOT / "results/baseline_reproduction/val/metrics.json", None),
        ("test", ROOT / "results/final_test/yolo11n/predictions.json",
         ROOT / "results/final_test/yolo11n/metrics.json", lock["yolo11n"]["operating_confidence"]),
        ("test", ROOT / "results/final_test/yolov3_tiny_reproduced/predictions.json",
         ROOT / "results/final_test/yolov3_tiny_reproduced/metrics.json",
         lock["yolov3_tiny_reproduced"]["operating_confidence"]),
    ]
    with tempfile.TemporaryDirectory(prefix="pedestrian_verify_") as temp:
        temporary = Path(temp)
        for mode in ("on", "off"):
            for seed in (42, 43):
                name = f"mosaic_{mode}_seed{seed}"
                source = ROOT / f"results/ablation/{name}/predictions.json.gz"
                target = temporary / f"{name}.json"
                target.write_bytes(gzip.decompress(source.read_bytes()))
                expected = json.loads((ROOT / f"results/ablation/{name}/result.json").read_text())["metrics"]
                expected_file = temporary / f"{name}_expected.json"
                expected_file.write_text(json.dumps(expected))
                cases.append(("val", target, expected_file, None))
        for index, (split, predictions, expected_path, threshold) in enumerate(cases):
            output = temporary / f"eval_{index}"
            command = [sys.executable, str(ROOT / "scripts/evaluate.py"),
                "--labels", str(DATA / f"processed/labels/{split}"),
                "--predictions", str(predictions), "--output-dir", str(output),
                "--split", split, "--model", "verification"]
            if threshold is not None:
                command += ["--operating-confidence", str(threshold)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            compare(json.loads(expected_path.read_text()), json.loads((output / "metrics.json").read_text()))
    print(f"Verified {len(cases)} saved prediction sets, locked thresholds and checkpoint hashes.")


if __name__ == "__main__":
    main()
