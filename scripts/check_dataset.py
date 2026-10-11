#!/usr/bin/env python3
"""Checks the PROCESSED dataset on disk and draws boxes for visual inspection.

  * every image has a label file (and vice versa), all values valid, all in [0,1]
  * manifests (data/splits/*.txt) match the folders and do not overlap
  * saves contact sheets with the boxes drawn on top:
        results/figures/dataset_examples/{train,val,test}_grid.png
Open the pictures and LOOK: every pedestrian should sit inside a green box.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
PROC, SPL = ROOT / "data/processed", ROOT / "data/splits"
OUT = ROOT / "results/figures/dataset_examples"
SPLITS = ("train", "val", "test")


def read_label(path):
    rows = [list(map(float, ln.split())) for ln in path.read_text().splitlines() if ln.strip()]
    for r in rows:
        assert len(r) == 5 and r[0] == 0, f"bad row in {path.name}: {r}"
        assert all(0 <= v <= 1 for v in r[1:]), f"coordinate out of [0,1] in {path.name}: {r}"
        assert r[3] > 0 and r[4] > 0, f"empty box in {path.name}"
    return rows


def draw(img_path, rows, cell=320):
    img = Image.open(img_path).convert("RGB")
    d, (w, h) = ImageDraw.Draw(img), img.size
    for _, xc, yc, bw, bh in rows:
        d.rectangle([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h], outline=(0, 255, 0), width=3)
    return img.resize((cell, cell))


def main():
    manifests = {s: (SPL / f"{s}.txt").read_text().split() for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not set(manifests[a]) & set(manifests[b]), f"LEAKAGE between {a} and {b}"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for s in SPLITS:
        imgs = sorted((PROC / "images" / s).glob("*.png"))
        labs = sorted((PROC / "labels" / s).glob("*.txt"))
        assert [p.stem for p in imgs] == [p.stem for p in labs] == sorted(manifests[s]), f"{s}: folders and manifest differ"
        n_boxes = sum(len(read_label(p)) for p in labs)
        print(f"{s:5s}: {len(imgs):3d} images, {n_boxes:3d} boxes - OK")
        pick = rng.sample(imgs, min(12, len(imgs)))
        sheet = Image.new("RGB", (4 * 320, 3 * 320), "white")
        for i, p in enumerate(pick):
            sheet.paste(draw(p, read_label(PROC / "labels" / s / f"{p.stem}.txt")), ((i % 4) * 320, (i // 4) * 320))
        sheet.save(OUT / f"{s}_grid.png")
    print(f"All checks passed. Look at the pictures in {OUT}")


if __name__ == "__main__":
    main()
