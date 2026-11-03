#!/usr/bin/env python3
"""Save a YOLO11n pedestrian-detection preview using the locked checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Input image; defaults to a validation example")
    parser.add_argument("--output", type=Path, default=ROOT / "demo_output/yolo11n_example.png")
    args = parser.parse_args()
    prepared = (ROOT / "data/processed" if (ROOT / "data/processed/images/val").is_dir()
                else ROOT / "second_approach/data/data/processed")
    image = args.image or prepared / "images/val/FudanPed00025.png"
    assert image.is_file(), f"Input missing: {image}; prepare the dataset or pass --image"
    lock = json.loads((ROOT / "results/test_protocol.lock.json").read_text())
    checkpoint = ROOT / lock["yolo11n"]["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == lock["yolo11n"]["checkpoint_sha256"]
    result = YOLO(str(checkpoint)).predict(source=str(image), imgsz=640,
        conf=lock["yolo11n"]["operating_confidence"], iou=0.6,
        max_det=300, half=False, device=0 if torch.cuda.is_available() else "cpu",
        verbose=False)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.plot()[..., ::-1]).save(args.output)
    print(f"Saved {len(result.boxes)} pedestrian detections to {args.output}")


if __name__ == "__main__":
    main()
