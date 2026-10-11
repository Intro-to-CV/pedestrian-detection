#!/usr/bin/env python3
"""Penn-Fudan -> YOLO dataset (data preparation, Stage 2).

What this script does
  1. Finds PNGImages/ and PedMasks/ inside --raw and pairs every image with its mask.
  2. Converts each instance mask into bounding boxes (one box per pedestrian).
  3. Letterbox-resizes every image to 640x640 (aspect ratio kept, grey padding)
     and transforms the boxes the same way.
  4. Writes YOLO labels:  0 x_center y_center width height   (all normalized 0..1).
  5. Makes a reproducible split (seed 42, sorted names, 70/15/rest) -> 119/25/26.
  6. Asserts there is no overlap between splits.
  7. Saves split manifests, statistics.json and configs/dataset.yaml.

Augmentation is NOT done here. Ultralytics augments on the fly and only the
training set, so val/test stay untouched.

Run from anywhere:  python scripts/prepare_dataset.py --raw data/raw
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
IMGSZ = 640
PAD_COLOR = (114, 114, 114)  # the grey Ultralytics uses for letterboxing
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------- 1. pairing
def find_dir(root: Path, name: str) -> Path:
    hits = [p for p in root.rglob(name) if p.is_dir() and "__MACOSX" not in p.parts]
    if not hits:
        raise SystemExit(f"Could not find a folder named '{name}' inside {root}")
    return min(hits, key=lambda p: len(p.parts))


def pair_images_and_masks(raw: Path):
    """image  FudanPed00001.png  <->  mask  FudanPed00001_mask.png"""
    img_dir, mask_dir = find_dir(raw, "PNGImages"), find_dir(raw, "PedMasks")
    images = {p.stem: p for p in img_dir.glob("*.png")}
    masks = {p.stem[: -len("_mask")]: p for p in mask_dir.glob("*_mask.png")}
    if set(images) != set(masks):
        print("Images without mask:", sorted(set(images) - set(masks))[:10])
        print("Masks without image:", sorted(set(masks) - set(images))[:10])
        raise SystemExit("Image/mask names do not match.")
    names = sorted(images)  # sorted BEFORE shuffling -> reproducible
    print(f"Found {len(names)} image/mask pairs in {img_dir.parent}")
    if len(names) != 170:
        print(f"WARNING: expected 170 images, got {len(names)}")
    return names, images, masks


# ------------------------------------------------------- 2. mask -> boxes
def mask_to_boxes(mask_path: Path):
    """Mask pixel value 0 = background, 1..K = pedestrian number k.
    Box = smallest rectangle around all pixels with that value.
    Returns pixel boxes (x1, y1, x2, y2) with exclusive upper edge."""
    m = np.array(Image.open(mask_path))  # do NOT .convert('RGB'): it would destroy the ids
    if m.ndim == 3:
        assert np.all(m[..., 0] == m[..., 1]) and np.all(m[..., 0] == m[..., 2]), \
            f"{mask_path.name} is a colour mask; expected instance ids"
        m = m[..., 0]
    boxes = []
    for inst_id in np.unique(m):
        if inst_id == 0:
            continue
        ys, xs = np.nonzero(m == inst_id)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes, m.shape[::-1]  # (W, H)


# -------------------------------------------------- 3. resize + 4. YOLO format
def letterbox(img: Image.Image, boxes, size=IMGSZ):
    w, h = img.size
    r = size / max(w, h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BICUBIC), (dx, dy))
    new_boxes = [(x1 * r + dx, y1 * r + dy, x2 * r + dx, y2 * r + dy) for x1, y1, x2, y2 in boxes]
    return canvas, new_boxes


def to_yolo_line(box, size=IMGSZ):
    x1, y1, x2, y2 = (min(max(v, 0), size) for v in box)
    xc, yc = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size
    bw, bh = (x2 - x1) / size, (y2 - y1) / size
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


# ------------------------------------------------------------- 5. split
def make_split(names, seed=SEED):
    n = len(names)
    n_train, n_val = n * 70 // 100, n * 15 // 100  # integer arithmetic, no float rounding
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }


def assert_no_leakage(splits, all_names):
    sets = {k: set(v) for k, v in splits.items()}
    for a in SPLITS:
        assert len(sets[a]) == len(splits[a]), f"duplicate names inside {a}"
        for b in SPLITS:
            if a < b:
                assert not (sets[a] & sets[b]), f"LEAKAGE: {a} and {b} share {sets[a] & sets[b]}"
    assert set().union(*sets.values()) == set(all_names), "split does not cover all images"
    print("OK: no filename overlap between train/val/test; all images covered")


# ------------------------------------------------------ 7. statistics
def source_of(name: str) -> str:
    return "Penn" if name.startswith("Penn") else "Fudan" if name.startswith("Fudan") else "other"


def build_statistics(splits, orig_boxes):
    all_h = np.array([y2 - y1 for b in orig_boxes.values() for (_, y1, _, y2) in b])
    lo, hi = np.percentile(all_h, [100 / 3, 200 / 3])  # tertiles -> small / medium / large

    def size_bin(h):
        return "small" if h < lo else "medium" if h < hi else "large"

    stats = {
        "seed": SEED,
        "split_rule": "sorted names -> random.Random(42).shuffle -> train n*70//100, val n*15//100, test = rest",
        "image_size_after_preprocessing": [IMGSZ, IMGSZ],
        "resize_method": "letterbox (aspect ratio kept, grey 114 padding)",
        "total_images": sum(len(v) for v in splits.values()),
        "total_instances": int(len(all_h)),
        "box_height_px_original": {"min": int(all_h.min()), "median": float(np.median(all_h)), "max": int(all_h.max())},
        "box_height_size_bins_px": {"small": f"< {lo:.0f}", "medium": f"{lo:.0f} - {hi:.0f}", "large": f">= {hi:.0f}"},
        "splits": {},
    }
    for s, names in splits.items():
        counts = [len(orig_boxes[n]) for n in names]
        heights = [y2 - y1 for n in names for (_, y1, _, y2) in orig_boxes[n]]
        bins = {"small": 0, "medium": 0, "large": 0}
        for h in heights:
            bins[size_bin(h)] += 1
        stats["splits"][s] = {
            "images": len(names),
            "instances": int(sum(counts)),
            "instances_per_image": {"mean": round(float(np.mean(counts)), 3), "min": min(counts), "max": max(counts)},
            "images_with_no_boxes": int(sum(c == 0 for c in counts)),
            "box_height_px_original": {"min": min(heights), "median": float(np.median(heights)), "max": max(heights)},
            "boxes_by_size_bin": bins,
            "images_by_source": {src: sum(source_of(n) == src for n in names) for src in ("Penn", "Fudan")},
        }
    return stats


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw", help="folder with the unzipped Kaggle dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--yaml", type=Path, default=ROOT / "configs/dataset.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite existing data/processed")
    args = ap.parse_args()

    names, images, masks = pair_images_and_masks(args.raw)

    # boxes in ORIGINAL pixels (used for statistics) + sanity checks
    orig_boxes = {}
    for n in names:
        boxes, (w, h) = mask_to_boxes(masks[n])
        assert Image.open(images[n]).size == (w, h), f"{n}: image and mask sizes differ"
        assert boxes, f"{n}: mask has no pedestrians"
        assert all(0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h for x1, y1, x2, y2 in boxes), f"{n}: bad box"
        orig_boxes[n] = boxes

    splits = make_split(names)
    print({k: len(v) for k, v in splits.items()})
    assert_no_leakage(splits, names)

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists. Re-run with --force to overwrite.")
        shutil.rmtree(args.out)
    for s, split_names in splits.items():
        (args.out / "images" / s).mkdir(parents=True)
        (args.out / "labels" / s).mkdir(parents=True)
        for n in split_names:
            img, boxes = letterbox(Image.open(images[n]).convert("RGB"), orig_boxes[n])
            img.save(args.out / "images" / s / f"{n}.png")
            (args.out / "labels" / s / f"{n}.txt").write_text("\n".join(to_yolo_line(b) for b in boxes) + "\n")

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    for s, split_names in splits.items():
        (args.splits_dir / f"{s}.txt").write_text("\n".join(split_names) + "\n")
    stats = build_statistics(splits, orig_boxes)
    (args.splits_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")

    args.yaml.parent.mkdir(parents=True, exist_ok=True)
    rel = Path(args.out).resolve().relative_to(ROOT).as_posix()
    args.yaml.write_text(
        f"# Run Ultralytics from the repository root so this relative path resolves.\n"
        f"path: {rel}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: person\n"
    )
    print(f"Done. Instances: {stats['total_instances']}. Wrote {args.out}, {args.splits_dir}, {args.yaml}")


if __name__ == "__main__":
    main()
