#!/usr/bin/env python3
"""Draw the four preselected validation error cases on prepared images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate import iou, read_ground_truth

ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "data/processed" if (ROOT / "data/processed/images/train").is_dir()
        else ROOT / "second_approach/data/data/processed")
PREDICTIONS = ROOT / "second_approach/results/validation_predictions.json"
OUTPUT = ROOT / "results/failures"
CASES = ("PennPed00015", "FudanPed00025", "PennPed00044", "FudanPed00016")
CONFIDENCE = 0.25


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def rectangle(draw: ImageDraw.ImageDraw, box, color, label: str, row: int):
    draw.rectangle(tuple(box), outline=color, width=4)
    x = max(0, min(520, int(box[0])))
    y = max(0, min(610, int(box[1]) - 29 - 24 * row))
    draw.rectangle((x, y, x + min(260, 10 * len(label) + 12), y + 24), fill=(15, 23, 42))
    draw.text((x + 4, y + 2), label, fill=color, font=font(16))


def main():
    payload = json.loads(PREDICTIONS.read_text())
    predicted = {x["image_id"]: x["detections"] for x in payload["images"]}
    truth = read_ground_truth(DATA / "labels/val", 640)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    notes = []
    for name in CASES:
        image = Image.open(DATA / "images/val" / f"{name}.png").convert("RGB")
        canvas = Image.new("RGB", (640, 700), "white")
        canvas.paste(image, (0, 60))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), name, fill=(15, 23, 42), font=font(26))
        draw.text((12, 38), "green: matched  amber: missed  red: false positive  yellow: low confidence",
                  fill=(50, 65, 83), font=font(13))
        targets = truth[name]
        detections = sorted(predicted[name], key=lambda x: x["confidence"], reverse=True)
        assigned = set()
        false_positives = []
        for det in detections:
            if det["confidence"] < CONFIDENCE:
                continue
            overlaps = [(iou(np.asarray(det["bbox_xyxy"]), gt["bbox"]), idx)
                        for idx, gt in enumerate(targets) if idx not in assigned]
            best, idx = max(overlaps, default=(0.0, -1))
            if best >= 0.5:
                assigned.add(idx)
                rectangle(draw, [v + (60 if j % 2 else 0) for j, v in enumerate(det["bbox_xyxy"])],
                          "#16a34a", f"TP {det['confidence']:.2f}", 0)
            else:
                false_positives.append({"confidence": det["confidence"], "best_iou": best})
                rectangle(draw, [v + (60 if j % 2 else 0) for j, v in enumerate(det["bbox_xyxy"])],
                          "#dc2626", f"FP {det['confidence']:.2f}", 0)
        missed = []
        for idx, gt in enumerate(targets):
            if idx in assigned:
                continue
            box = gt["bbox"].tolist()
            rectangle(draw, [v + (60 if j % 2 else 0) for j, v in enumerate(box)],
                      "#f59e0b", "FN ground truth", 1)
            candidates = [(iou(np.asarray(det["bbox_xyxy"]), gt["bbox"]), det["confidence"], det)
                          for det in detections if det["confidence"] < CONFIDENCE]
            overlap, confidence, candidate = max(candidates, default=(0.0, 0.0, None),
                                                  key=lambda x: x[0])
            if overlap >= 0.5 and candidate is not None:
                shifted = [v + (60 if j % 2 else 0)
                           for j, v in enumerate(candidate["bbox_xyxy"])]
                draw.rectangle(tuple(shifted), outline="#facc15", width=2)
                rectangle(draw, shifted, "#facc15", f"low {confidence:.2f}", 2)
            missed.append({"best_low_confidence_iou": overlap,
                           "best_low_confidence_score": confidence})
        canvas.save(OUTPUT / f"{name}.png")
        notes.append({"image": name, "false_positives": false_positives, "missed": missed})
    (OUTPUT / "measurements.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
