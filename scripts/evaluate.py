#!/usr/bin/env python3
"""Evaluate detector predictions against YOLO labels with one shared protocol.

Predictions must use the portable ``person_xyxy_pixels_v1`` JSON schema written
by the YOLOv3-tiny notebook::

    {"images": [{"image_id": "FudanPed00001", "detections": [
        {"class_id": 0, "confidence": 0.91, "bbox_xyxy": [x1, y1, x2, y2]}
    ]}]}

The same schema can be exported from YOLO11n, which keeps model comparison out
of framework-specific validators.  The script writes metrics.json, metrics.csv,
pr_curve.png and error_review.csv to --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


IOU_THRESHOLDS = tuple(round(0.5 + step * 0.05, 2) for step in range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Directory of YOLO .txt labels.")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions JSON in person_xyxy_pixels_v1 schema.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics and plots.")
    parser.add_argument("--image-size", type=int, default=640, help="Side length of prepared square images (default: 640).")
    parser.add_argument("--split", default="val", help="Name stored in result metadata (default: val).")
    parser.add_argument("--model", default="unknown", help="Model name stored in result metadata.")
    parser.add_argument("--error-confidence", type=float, default=0.25,
                        help="Confidence floor used for the FP/FN review (default: 0.25).")
    return parser.parse_args()


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    left_top = np.maximum(box_a[:2], box_b[:2])
    right_bottom = np.minimum(box_a[2:], box_b[2:])
    overlap = np.maximum(right_bottom - left_top, 0)
    intersection = float(overlap[0] * overlap[1])
    area_a = float(np.prod(np.maximum(box_a[2:] - box_a[:2], 0)))
    area_b = float(np.prod(np.maximum(box_b[2:] - box_b[:2], 0)))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def read_ground_truth(labels_dir: Path, image_size: int) -> dict[str, list[dict[str, Any]]]:
    files = sorted(labels_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"No YOLO labels found in {labels_dir}")
    ground_truth: dict[str, list[dict[str, Any]]] = {}
    for path in files:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 YOLO values")
            class_id = int(values[0])
            xc, yc, width, height = map(float, values[1:])
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"{path}:{line_number}: invalid normalized box")
            rows.append({
                "class_id": class_id,
                "bbox": np.array(((xc - width / 2) * image_size, (yc - height / 2) * image_size,
                                  (xc + width / 2) * image_size, (yc + height / 2) * image_size), dtype=float),
            })
        ground_truth[path.stem] = rows
    return ground_truth


def read_predictions(path: Path, known_images: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_rows = payload["images"] if isinstance(payload, dict) else payload
    if not isinstance(image_rows, list):
        raise ValueError("predictions JSON must be a list or an object with an 'images' list")
    detections: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for image in image_rows:
        image_id = str(image["image_id"])
        if image_id not in known_images:
            raise ValueError(f"Prediction references image absent from labels: {image_id}")
        if image_id in seen_images:
            raise ValueError(f"Predictions contain duplicate image_id: {image_id}")
        seen_images.add(image_id)
        for detection in image.get("detections", []):
            box = np.asarray(detection["bbox_xyxy"], dtype=float)
            confidence = float(detection.get("confidence", detection.get("score")))
            class_id = int(detection.get("class_id", detection.get("class", 0)))
            if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"{image_id}: invalid bbox_xyxy {box.tolist()}")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{image_id}: confidence must be in [0, 1]")
            detections.append({"image_id": image_id, "class_id": class_id, "confidence": confidence, "bbox": box})
    missing = known_images - seen_images
    if missing:
        raise ValueError(f"Predictions JSON must include empty detections lists too; missing {len(missing)} images")
    return detections


def sorted_detections(detections: list[dict[str, Any]], class_id: int) -> list[dict[str, Any]]:
    return sorted((d for d in detections if d["class_id"] == class_id), key=lambda d: d["confidence"], reverse=True)


def match(detections: list[dict[str, Any]], ground_truth: dict[str, list[dict[str, Any]]],
          class_id: int, threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    ordered = sorted_detections(detections, class_id)
    count_gt = sum(1 for boxes in ground_truth.values() for box in boxes if box["class_id"] == class_id)
    assigned: dict[str, set[int]] = defaultdict(set)
    true_positive = np.zeros(len(ordered), dtype=float)
    for index, detection in enumerate(ordered):
        candidates = ground_truth[detection["image_id"]]
        best_index, best_iou = -1, 0.0
        for gt_index, target in enumerate(candidates):
            if target["class_id"] != class_id or gt_index in assigned[detection["image_id"]]:
                continue
            overlap = iou(detection["bbox"], target["bbox"])
            if overlap > best_iou:
                best_index, best_iou = gt_index, overlap
        if best_index >= 0 and best_iou >= threshold:
            assigned[detection["image_id"]].add(best_index)
            true_positive[index] = 1.0
    return true_positive, np.asarray([d["confidence"] for d in ordered]), count_gt


def average_precision(true_positive: np.ndarray, count_gt: int) -> float:
    if count_gt == 0:
        return float("nan")
    if not len(true_positive):
        return 0.0
    false_positive = 1 - true_positive
    recall = np.cumsum(true_positive) / count_gt
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive) + np.cumsum(false_positive), 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    # COCO's 101 recall levels: for each level use the greatest interpolated
    # precision available at that recall or above.  This formulation also
    # correctly gives AP=1.0 to a perfect detector.
    recall_grid = np.linspace(0, 1, 101)
    interpolated = [float(envelope[recall >= level].max()) if np.any(recall >= level) else 0.0
                    for level in recall_grid]
    return float(np.mean(interpolated))


def precision_recall_at_best_f1(true_positive: np.ndarray, confidences: np.ndarray,
                                count_gt: int) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    if not len(true_positive):
        return 0.0, 0.0, 1.0, np.array([0.0]), np.array([0.0])
    cumulative_tp = np.cumsum(true_positive)
    precision = cumulative_tp / np.arange(1, len(true_positive) + 1)
    recall = cumulative_tp / count_gt
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(float).eps)
    index = int(np.argmax(f1))
    return float(precision[index]), float(recall[index]), float(confidences[index]), precision, recall


def error_rows(detections: list[dict[str, Any]], ground_truth: dict[str, list[dict[str, Any]]],
               confidence_floor: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assigned: dict[str, set[int]] = defaultdict(set)
    for detection in sorted((d for d in detections if d["confidence"] >= confidence_floor),
                            key=lambda d: d["confidence"], reverse=True):
        targets = ground_truth[detection["image_id"]]
        best_index, best_iou = -1, 0.0
        for target_index, target in enumerate(targets):
            if target["class_id"] != detection["class_id"] or target_index in assigned[detection["image_id"]]:
                continue
            overlap = iou(detection["bbox"], target["bbox"])
            if overlap > best_iou:
                best_index, best_iou = target_index, overlap
        if best_index >= 0 and best_iou >= 0.5:
            assigned[detection["image_id"]].add(best_index)
        else:
            rows.append({"type": "FP", "image_id": detection["image_id"], "class_id": detection["class_id"],
                         "confidence": detection["confidence"], "iou_with_best_unmatched_gt": round(best_iou, 6),
                         "bbox_xyxy": json.dumps(detection["bbox"].round(3).tolist())})
    for image_id, targets in ground_truth.items():
        for target_index, target in enumerate(targets):
            if target_index not in assigned[image_id]:
                rows.append({"type": "FN", "image_id": image_id, "class_id": target["class_id"], "confidence": "",
                             "iou_with_best_unmatched_gt": "", "bbox_xyxy": json.dumps(target["bbox"].round(3).tolist())})
    return rows


def write_pr_curve(path: Path, precision: np.ndarray, recall: np.ndarray, ap50: float) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, color="#2563eb", linewidth=2, label=f"person, AP@0.5={ap50:.3f}")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Validation PR curve (IoU=0.50)")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 <= args.error_confidence <= 1:
        raise ValueError("--error-confidence must be in [0, 1]")
    ground_truth = read_ground_truth(args.labels, args.image_size)
    detections = read_predictions(args.predictions, set(ground_truth))
    class_ids = sorted({target["class_id"] for targets in ground_truth.values() for target in targets})
    if not class_ids:
        raise ValueError("No ground-truth objects found")

    ap_by_iou: dict[float, list[float]] = defaultdict(list)
    for class_id in class_ids:
        for threshold in IOU_THRESHOLDS:
            tp, _, count_gt = match(detections, ground_truth, class_id, threshold)
            ap_by_iou[threshold].append(average_precision(tp, count_gt))
    tp50, confidences50, count_gt50 = match(detections, ground_truth, class_ids[0], 0.5)
    precision, recall, best_confidence, pr_precision, pr_recall = precision_recall_at_best_f1(tp50, confidences50, count_gt50)
    map50 = float(np.nanmean(ap_by_iou[0.5]))
    map50_95 = float(np.mean([np.nanmean(ap_by_iou[threshold]) for threshold in IOU_THRESHOLDS]))
    errors = error_rows(detections, ground_truth, args.error_confidence)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema": "detector_evaluation_v1",
        "model": args.model,
        "split": args.split,
        "images": len(ground_truth),
        "ground_truth_instances": sum(len(targets) for targets in ground_truth.values()),
        "detections": len(detections),
        "iou_thresholds_for_map": list(IOU_THRESHOLDS),
        "ap_method": "101-point interpolated AP",
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision_best_f1_iou50": precision,
        "recall_best_f1_iou50": recall,
        "confidence_best_f1_iou50": best_confidence,
        "error_review_iou": 0.5,
        "error_review_confidence": args.error_confidence,
        "false_positives": sum(row["type"] == "FP" for row in errors),
        "false_negatives": sum(row["type"] == "FN" for row in errors),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    with (args.output_dir / "error_review.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["type", "image_id", "class_id", "confidence", "iou_with_best_unmatched_gt", "bbox_xyxy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    write_pr_curve(args.output_dir / "pr_curve.png", pr_precision, pr_recall, map50)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
