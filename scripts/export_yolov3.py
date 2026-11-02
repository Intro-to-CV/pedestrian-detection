#!/usr/bin/env python3
"""Run the pinned YOLOv3 validator and export predictions for shared scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "experiments/runs"
DATA = (ROOT / "data" if (ROOT / "data/processed/images/train").is_dir()
        else ROOT / "second_approach/data/data")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--operating-confidence", type=float)
    p.add_argument("--output-dir", type=Path,
                   help="Fresh directory for shared metrics and predictions")
    args = p.parse_args()
    if args.split == "test" and args.operating_confidence is None:
        p.error("--operating-confidence from validation is required for test")
    source, weights = args.source.resolve(), args.weights.resolve()
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == \
        "97493de334161fe37c630a4fcf1c83c064526d63"
    assert weights.is_file() and (DATA / f"splits/{args.split}.txt").is_file()
    training_record = ROOT / "results/baseline_reproduction/training.json"
    if training_record.is_file():
        assert sha256(weights) == json.loads(training_record.read_text())["checkpoint_sha256"]
    if args.split == "test":
        lock = json.loads((ROOT / "results/test_protocol.lock.json").read_text())
        selection = lock["yolov3_tiny_reproduced"]
        assert sha256(weights) == selection["checkpoint_sha256"]
        assert args.operating_confidence == selection["operating_confidence"]
        assert sha256(ROOT / "data/splits/test.txt") == lock["test_split_manifest_sha256"]
    run = LOCAL / f"yolov3_reproduced_{args.split}"
    assert not run.exists(), f"Existing validation run: {run}"
    public = args.output_dir.resolve() if args.output_dir else (
        ROOT / "results/baseline_reproduction/val" if args.split == "val"
        else ROOT / "results/final_test/yolov3_tiny_reproduced")
    assert not (public / "metrics.json").exists(), f"Results already exist: {public}"
    data_yaml = LOCAL / "dataset.yaml"
    if not data_yaml.is_file():
        LOCAL.mkdir(parents=True, exist_ok=True)
        data_yaml.write_text(yaml.safe_dump({"path": str((DATA / "processed").resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "person"}}))
    for name in ("wandb", "clearml", "comet_ml", "albumentations"):
        sys.modules[name] = None
    sys.path.insert(0, str(source))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from models.experimental import attempt_load
    from val import run as validate
    import torch

    model = attempt_load(str(weights), device=torch.device("cpu"), fuse=False)
    parameters = sum(x.numel() for x in model.parameters())
    del model
    start = time.perf_counter()
    results, _, timings = validate(data=str(data_yaml), weights=str(weights), batch_size=16,
        imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300,
        task=args.split, device="0", workers=0, single_cls=False,
        augment=False, save_txt=True, save_conf=True, save_json=False,
        project=LOCAL, name=run.name, exist_ok=False, half=False, plots=True)
    wall = time.perf_counter() - start
    assert run.is_dir()
    names = (DATA / f"splits/{args.split}.txt").read_text().split()
    rows = []
    for name in names:
        detections = []
        label = run / "labels" / f"{name}.txt"
        for line in label.read_text().splitlines() if label.exists() else []:
            cls, x, y, w, h, confidence = map(float, line.split())
            box = [max(0.0, min(640.0, v)) for v in
                   ((x-w/2)*640, (y-h/2)*640, (x+w/2)*640, (y+h/2)*640)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append({"class_id": int(cls), "confidence": confidence,
                               "bbox_xyxy": box})
        rows.append({"image_id": name, "file_name": f"{name}.png",
                     "width": 640, "height": 640, "detections": detections})
    prediction_path = LOCAL / f"yolov3_reproduced_{args.split}_predictions.json"
    prediction_path.write_text(json.dumps({"schema": "person_xyxy_pixels_v1",
        "split": args.split, "coordinate_space": "prepared 640x640 letterboxed images",
        "images": rows}) + "\n")
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prediction_path, public / "predictions.json")
    command = [sys.executable, str(ROOT / "scripts/evaluate.py"),
        "--labels", str(DATA / f"processed/labels/{args.split}"),
        "--predictions", str(prediction_path), "--output-dir", str(public),
        "--model", "YOLOv3-tiny (reproduced)", "--split", args.split]
    if args.operating_confidence is not None:
        command += ["--operating-confidence", str(args.operating_confidence)]
    subprocess.run(command, check=True)
    timing = {"checkpoint_sha256": sha256(weights), "images": len(names),
              "batch": 16, "device": torch.cuda.get_device_name(0),
              "validation_wall_seconds": wall,
              "preprocess_ms_per_image": float(timings[0]),
              "inference_ms_per_image": float(timings[1]),
              "nms_ms_per_image": float(timings[2]),
              "parameters": parameters,
              "native_precision_best_f1": float(results[0]),
              "native_recall_best_f1": float(results[1]),
              "native_mAP50": float(results[2]),
              "native_mAP50_95": float(results[3])}
    (public / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
