#!/usr/bin/env python3
"""Look at the RAW data: draw each mask (coloured) and its bounding box on top of the image.

If the mask colours sit exactly on the people, then image<->mask pairing is right.

  python scripts/visualize_masks.py --raw data/raw                  # 12 random pairs -> one grid
  python scripts/visualize_masks.py --raw data/raw --name FudanPed00001   # one pair: image | mask | overlay
Output: results/figures/mask_overlay/
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import ROOT, mask_to_boxes, pair_images_and_masks  # noqa: E402

COLORS = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 200, 0), (255, 0, 255),
          (0, 220, 220), (255, 128, 0), (150, 80, 255)]


def overlay(img_path, mask_path):
    img = Image.open(img_path).convert("RGB")
    m = np.array(Image.open(mask_path))
    arr = np.array(img).astype(float)
    for k, inst in enumerate(i for i in np.unique(m) if i != 0):
        sel = m == inst
        arr[sel] = 0.5 * arr[sel] + 0.5 * np.array(COLORS[k % len(COLORS)])
    out = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(out)
    boxes, _ = mask_to_boxes(mask_path)
    for b in boxes:
        d.rectangle(b, outline=(255, 255, 255), width=2)
    return img, m, out


def fit(img, cell):
    img = img.copy()
    img.thumbnail((cell, cell))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--name", help="one image name without .png, e.g. FudanPed00001")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0, help="change to see other random images")
    args = ap.parse_args()
    names, images, masks = pair_images_and_masks(args.raw)
    out_dir = ROOT / "results/figures/mask_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.name:
        img, m, ov = overlay(images[args.name], masks[args.name])
        mask_vis = Image.fromarray((m * (255 // max(m.max(), 1))).astype(np.uint8)).convert("RGB")
        sheet = Image.new("RGB", (img.width * 3, img.height), "white")
        for i, im in enumerate((img, mask_vis, ov)):
            sheet.paste(im, (i * img.width, 0))
        path = out_dir / f"{args.name}_triple.png"
    else:
        pick = random.Random(args.seed).sample(names, min(args.n, len(names)))
        cell, cols = 400, 4
        rows = (len(pick) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
        for i, n in enumerate(pick):
            sheet.paste(fit(overlay(images[n], masks[n])[2], cell), ((i % cols) * cell, (i // cols) * cell))
        print("Shown:", ", ".join(pick))
        path = out_dir / "overlay_grid.png"
    sheet.save(path)
    print("Saved", path)


if __name__ == "__main__":
    main()
