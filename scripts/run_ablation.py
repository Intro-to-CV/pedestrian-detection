#!/usr/bin/env python3
"""Train and validate the predeclared YOLO11n mosaic ablation.

Run one condition at a time: --seed {42,43} --mosaic {on,off}. Seed 42/on
reuses the already completed second-approach checkpoint. All conditions use
the same pretrained checkpoint, split, hyperparameters, and evaluator.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import ultralytics
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "experiments" / "runs"
PUBLIC = ROOT / "results" / "ablation"
WEIGHTS = ROOT / "yolo11n.pt"
SPLITS = (ROOT / "data" if (ROOT / "data/processed/images/train").is_dir()
          else ROOT / "second_approach/data/data")
EXPECTED_WEIGHTS_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data_yaml() -> Path:
    assert (SPLITS / "splits/train.txt").exists(), "Prepared archive missing; run the second-approach notebook setup."
    assert (SPLITS / "processed/images/train").is_dir()
    LOCAL.mkdir(parents=True, exist_ok=True)
    target = LOCAL / "dataset.yaml"
    target.write_text(yaml.safe_dump({"path": str((SPLITS / "processed").resolve()),
                                     "train": "images/train", "val": "images/val",
                                     "test": "images/test", "names": {0: "person"}}))
    return target


def config(seed: int, mosaic: bool, dataset: Path, name: str) -> dict:
    return dict(data=str(dataset), imgsz=640, epochs=50, batch=16,
        optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.9,
        weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
        warmup_bias_lr=0.0, cos_lr=True, nbs=64, close_mosaic=0,
        seed=seed, deterministic=True, amp=True, pretrained=True,
        device=0, workers=0, patience=0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0,
        translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0 if mosaic else 0.0,
        mixup=0.0, copy_paste=0.0,
        project=str(LOCAL), name=name, exist_ok=False, plots=False, verbose=False)


def predictions(label_dir: Path, split: str, dataset_dir: Path, output: Path) -> int:
    """Export native validation rows; discard only zero-area edge boxes."""
    names = (dataset_dir / f"splits/{split}.txt").read_text().split()
    rows, dropped = [], 0
    for name in names:
        detections = []
        label = label_dir / f"{name}.txt"
        for line in label.read_text().splitlines() if label.exists() else []:
            cls, x, y, w, h, conf = map(float, line.split())
            xyxy = [max(0.0, min(640.0, v)) for v in
                    ((x-w/2)*640, (y-h/2)*640, (x+w/2)*640, (y+h/2)*640)]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                dropped += 1
                continue
            detections.append({"class_id": int(cls), "confidence": conf,
                               "bbox_xyxy": xyxy})
        rows.append({"image_id": name,
                     "file_name": f"{name}.png",
                     "width": 640, "height": 640,
                     "detections": detections})
    assert len(rows) == len(names) and {r["image_id"] for r in rows} == set(names)
    output.write_text(json.dumps({"schema": "person_xyxy_pixels_v1", "split": split,
        "coordinate_space": "prepared 640x640 letterboxed images", "images": rows}) + "\n")
    return dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--mosaic", choices=("on", "off"), required=True)
    parser.add_argument("--output-root", type=Path, default=PUBLIC,
                        help="Results directory; use a fresh local path to reproduce published runs")
    parser.add_argument("--train-reused", action="store_true",
                        help="Train seed-42 mosaic-on from scratch instead of reusing the saved checkpoint")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "A CUDA GPU is required for comparable ablation runtime."
    assert ultralytics.__version__ == "8.3.228", "Use the pinned experiment environment."
    assert WEIGHTS.is_file() and digest(WEIGHTS) == EXPECTED_WEIGHTS_SHA256
    dataset = data_yaml()
    name = f"mosaic_{args.mosaic}_seed{args.seed}"
    reused = args.seed == 42 and args.mosaic == "on" and not args.train_reused
    run = (ROOT / "second_approach/runs/yolo11n_seed42_gpu") if reused else LOCAL / name
    checkpoint = run / "weights/best.pt"
    published = args.output_root.resolve() / name
    if published.exists():
        raise FileExistsError(f"Result already exists: {published}")
    requested = config(args.seed, args.mosaic == "on", dataset, name)
    if reused:
        expected = json.loads((ROOT / "second_approach/results/native_validation_metrics.json").read_text())
        assert digest(checkpoint) == expected["checkpoint_sha256"]
        training_seconds = json.loads((ROOT / "second_approach/results/training_runtime.json").read_text())["wall_seconds"]
    else:
        if run.exists():
            raise FileExistsError(f"Training run already exists: {run}")
        started = time.perf_counter()
        try:
            YOLO(str(WEIGHTS)).train(**requested)
        finally:
            (LOCAL / f"{name}_elapsed.txt").write_text(str(time.perf_counter() - started))
        training_seconds = time.perf_counter() - started
    assert checkpoint.is_file()
    history = pd.read_csv(run / "results.csv")
    assert len(history) == 50 and history.select_dtypes("number").notna().all().all()
    effective = yaml.safe_load((run / "args.yaml").read_text())
    for key in ("seed", "mosaic", "batch", "imgsz", "epochs", "lr0", "optimizer", "close_mosaic"):
        assert effective[key] == requested[key], (key, effective[key], requested[key])
    published.mkdir(parents=True)
    output = LOCAL / f"{name}_val_predictions.json"
    t0 = time.perf_counter()
    if reused:
        validation_dir = ROOT / "second_approach/runs/yolo11n_seed42_gpu_validation"
    else:
        validation_dir = LOCAL / f"{name}_validation"
        assert not validation_dir.exists()
        YOLO(str(checkpoint)).val(data=str(dataset), split="val", imgsz=640,
            batch=16, device=0, half=False, conf=0.001, iou=0.6,
            max_det=300, workers=0, project=str(LOCAL), name=validation_dir.name,
            exist_ok=False, save_txt=True, save_conf=True, plots=False, verbose=False)
    dropped = predictions(validation_dir / "labels", "val", SPLITS, output)
    inference_seconds = time.perf_counter() - t0
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
        "--labels", str(SPLITS / "processed/labels/val"),
        "--predictions", str(output), "--output-dir", str(LOCAL / f"{name}_evaluation"),
        "--model", "YOLO11n", "--split", "val"], check=True, capture_output=True, text=True)
    metrics = json.loads((LOCAL / f"{name}_evaluation/metrics.json").read_text())
    record = {"seed": args.seed, "mosaic": args.mosaic,
              "reused_prior_run": reused, "pretrained_sha256": digest(WEIGHTS),
              "checkpoint_sha256": digest(checkpoint),
              "training_seconds": training_seconds,
              "validation_prediction_seconds": inference_seconds,
              "degenerate_boxes_discarded": dropped,
              "environment": {"python": platform.python_version(),
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "gpu": torch.cuda.get_device_name(0)},
              "settings": {key: value for key, value in requested.items()
                           if key not in ("data", "project", "name")},
              "metrics": metrics}
    (published / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    (published / "predictions.json.gz").write_bytes(gzip.compress(output.read_bytes(), mtime=0))
    history.to_csv(published / "epochs.csv", index=False)
    print(json.dumps({"run": name, "mAP50_95": metrics["mAP50_95"],
                      "mAP50": metrics["mAP50"], "training_seconds": training_seconds}, indent=2))


if __name__ == "__main__":
    main()
