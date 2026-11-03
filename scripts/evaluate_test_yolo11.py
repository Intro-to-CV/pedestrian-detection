#!/usr/bin/env python3
"""One-time YOLO11n test evaluation under the locked validation protocol."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from run_ablation import LOCAL, ROOT, SPLITS, data_yaml, digest, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh directory; use for an independent reproduction")
    args = parser.parse_args()
    lock = json.loads((ROOT / "results/test_protocol.lock.json").read_text())
    selected = lock["yolo11n"]
    assert digest(ROOT / "data/splits/test.txt") == lock["test_split_manifest_sha256"]
    checkpoint = ROOT / selected["checkpoint"]
    assert digest(checkpoint) == selected["checkpoint_sha256"]
    assert ultralytics.__version__ == "8.3.228" and torch.cuda.is_available()
    out = args.output_dir.resolve() if args.output_dir else ROOT / "results/final_test/yolo11n"
    run = LOCAL / "yolo11n_locked_test"
    assert not out.exists() and not run.exists(), "Test evaluation already exists; do not retune after seeing it."
    dataset = data_yaml()
    start = time.perf_counter()
    result = YOLO(str(checkpoint)).val(data=str(dataset), split="test", imgsz=640,
        batch=16, device=0, half=False, conf=0.001, iou=0.6,
        max_det=300, workers=0, project=str(LOCAL), name=run.name,
        exist_ok=False, save_txt=True, save_conf=True, plots=True, verbose=False)
    wall = time.perf_counter() - start
    out.mkdir(parents=True)
    dropped = predictions(run / "labels", "test", SPLITS, out / "predictions.json")
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
        "--labels", str(SPLITS / "processed/labels/test"),
        "--predictions", str(out / "predictions.json"),
        "--output-dir", str(out), "--model", "YOLO11n", "--split", "test",
        "--operating-confidence", str(selected["operating_confidence"])], check=True)
    speed = result.speed
    timing = {"checkpoint_sha256": selected["checkpoint_sha256"],
              "images": 26, "batch": 16, "precision": "FP32",
              "device": torch.cuda.get_device_name(0),
              "whole_validation_wall_seconds": wall,
              "preprocess_ms_per_image": float(speed["preprocess"]),
              "inference_ms_per_image": float(speed["inference"]),
              "nms_ms_per_image": float(speed["postprocess"]),
              "degenerate_boxes_discarded": dropped,
              "native_mAP50": float(result.box.map50),
              "native_mAP50_95": float(result.box.map)}
    (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
