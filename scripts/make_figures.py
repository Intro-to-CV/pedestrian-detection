#!/usr/bin/env python3
"""Create compact comparison figures from saved metrics (no model inference)."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"


def main():
    with (ROOT / "results/final_test/comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    labels = ["YOLOv3-tiny", "YOLO11n"]
    colors = ["#64748b", "#2563eb"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, row in enumerate(rows):
        for j, split in enumerate(("val", "test")):
            value = float(row[f"{split}_mAP50_95"])
            x = j * 2.5 + i * 0.75
            ax.bar(x, value, width=0.68, color=colors[i], label=labels[i] if j == 0 else None)
            ax.text(x, value + 0.015, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_xticks([0.375, 2.875], ["Validation (25 images)", "Test (26 images)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Shared-evaluator pedestrian detection")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)

    with (ROOT / "results/ablation/runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for mode, color in (("on", "#2563eb"), ("off", "#dc2626")):
        selected = sorted((r for r in rows if r["mosaic"] == mode), key=lambda r: int(r["seed"]))
        x = [int(r["seed"]) for r in selected]
        y = [float(r["mAP50_95"]) for r in selected]
        ax.plot(x, y, marker="o", linewidth=2, markersize=7, color=color,
                label=f"Mosaic {mode} (mean {sum(y)/len(y):.3f})")
        for xi, yi in zip(x, y):
            offset = (-24 if xi == 42 else 8) if mode == "on" else (8 if xi == 42 else -18)
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, offset),
                        textcoords="offset points", ha="center", color=color)
    ax.set_xticks([42, 43])
    ax.set_xlim(41.7, 43.3)
    ax.set_ylim(0.82, 0.90)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Validation mAP@0.5:0.95")
    ax.set_title("YOLO11n mosaic ablation: small, seed-dependent effect")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "mosaic_ablation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
