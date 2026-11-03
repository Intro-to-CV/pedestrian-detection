#!/usr/bin/env python3
"""Measure native FP32 batch-1 validation speed on the same 25 images/GPU."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from run_ablation import data_yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "experiments/runs"
OUT = ROOT / "results/benchmarks"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("yolo11n", "yolov3-tiny"), required=True)
    p.add_argument("--source", type=Path, help="Pinned YOLOv3 checkout, when needed")
    p.add_argument("--output-root", type=Path, default=OUT)
    args = p.parse_args()
    assert torch.cuda.is_available()
    name = args.model.replace("-", "_")
    run = LOCAL / f"benchmark_{name}_val_batch1"
    assert not run.exists(), f"Benchmark already exists: {run}"
    dataset = data_yaml()
    start = time.perf_counter()
    if args.model == "yolo11n":
        import ultralytics
        assert ultralytics.__version__ == "8.3.228"
        from ultralytics import YOLO
        checkpoint = ROOT / "models/yolo11n_best.pt"
        result = YOLO(str(checkpoint)).val(data=str(dataset), split="val", imgsz=640,
            batch=1, conf=0.001, iou=0.6, max_det=300, device=0, half=False,
            workers=0, plots=False, project=str(LOCAL), name=run.name, exist_ok=False)
        timing = [float(result.speed[key]) for key in ("preprocess", "inference", "postprocess")]
        package = ultralytics.__version__
    else:
        assert args.source is not None
        import subprocess
        assert subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip() == \
            "97493de334161fe37c630a4fcf1c83c064526d63"
        for module in ("wandb", "clearml", "comet_ml", "albumentations"):
            sys.modules[module] = None
        sys.path.insert(0, str(args.source.resolve()))
        from val import run as validate
        checkpoint = ROOT / "models/yolov3_tiny_best.pt"
        _, _, native = validate(data=str(dataset), weights=str(checkpoint),
            batch_size=1, imgsz=640, conf_thres=0.001, iou_thres=0.6,
            max_det=300, task="val", device="0", workers=0, single_cls=False,
            augment=False, save_txt=False, save_conf=False, save_json=False,
            project=LOCAL, name=run.name, exist_ok=False, half=False, plots=False)
        timing = [float(x) for x in native]
        import ultralytics
        package = ultralytics.__version__
    wall = time.perf_counter() - start
    combined = sum(timing)
    record = {"model": args.model, "split": "val", "images": 25, "batch": 1,
              "precision": "FP32", "device": torch.cuda.get_device_name(0),
              "native_framework_version": package,
              "preprocess_ms_per_image": timing[0],
              "inference_ms_per_image": timing[1],
              "nms_ms_per_image": timing[2],
              "sum_ms_per_image": combined,
              "fps_from_native_timing": 1000 / combined,
              "whole_validation_wall_seconds": wall}
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"{name}_val_batch1.json"
    assert not target.exists(), f"Benchmark already exists: {target}"
    target.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
