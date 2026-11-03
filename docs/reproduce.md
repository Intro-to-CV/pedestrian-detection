# Reproducing the experiments

Run these commands from the repository root. The recorded runs used Python 3.13.9, PyTorch 2.11.0+cu130 and one NVIDIA GeForce RTX 3050 Laptop GPU. A compatible CUDA driver is needed to reproduce their speed measurements. The two trainers use **separate environments** because YOLOv3's pinned helper package differs from YOLO11n's.

## 1. Prepare the same dataset

Download [Penn–Fudan](https://www.cis.upenn.edu/~jshi/ped_html/) and extract its `PNGImages/` and `PedMasks/` directories below `data/raw/`. Then:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-yolo11.txt
.venv/bin/python scripts/prepare_dataset.py --raw data/raw
.venv/bin/python scripts/check_dataset.py
```

Check for **119 train, 25 validation and 26 test images**, with 423 boxes total. The preparation script checks disjoint manifests. It saves 640×640 letterboxed images and YOLO labels under `data/processed/`. The original prepared archive also works: extract `pedestrian_data.zip` in the repository root to populate `data/processed/` and `data/splits/`, then run `check_dataset.py`. Generated images are ignored by Git.

## 2. YOLO11n mosaic ablation

Place the official COCO-pretrained `yolo11n.pt` in the repository root. With Ultralytics installed, running `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"` downloads it when absent. The ablation script verifies its SHA-256 (`0ebbc80d…ae6417644ee1`) before training.

The repository contains the original seed-42 mosaic-on [checkpoint](../models/README.md). To rerun all four training conditions independently, use a **fresh clone** and a separate result directory:

```bash
for mode in on off; do
  for seed in 42 43; do
    .venv/bin/python scripts/run_ablation.py --seed "$seed" --mosaic "$mode" \
      --train-reused --output-root experiments/runs/reproduced_ablation
  done
done
.venv/bin/python scripts/summarize_ablation.py \
  --input-root experiments/runs/reproduced_ablation \
  --output-root experiments/runs/reproduced_ablation
```

Every run starts from the same pretrained weights and uses 50 epochs, batch 16, 640 px, AdamW, LR 0.001, seed 42 or 43, and the same augmentation settings except for `mosaic=1.0` versus `mosaic=0.0`. The published [ablation folder](../results/ablation/README.md) contains per-epoch histories, settings, shared-evaluator metrics and compressed predictions. Retraining can yield slightly different numbers across hardware/software versions.

## 3. YOLOv3-tiny baseline reproduction

The report's original MPS checkpoint was unavailable. This repository includes a new, validated FP32 GPU checkpoint. To retrain it from the same pinned source and pretrained weights:

```bash
mkdir -p experiments/runs/vendor
git clone https://github.com/ultralytics/yolov3.git experiments/runs/vendor/yolov3
git -C experiments/runs/vendor/yolov3 checkout --detach 97493de334161fe37c630a4fcf1c83c064526d63
curl -fL -o experiments/runs/yolov3-tiny.pt \
  https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-tiny.pt
python -m venv .venv-yolov3
.venv-yolov3/bin/python -m pip install -r requirements-yolov3.txt \
  -r experiments/runs/vendor/yolov3/requirements.txt
.venv-yolov3/bin/python scripts/reproduce_yolov3.py \
  --source experiments/runs/vendor/yolov3 \
  --weights experiments/runs/yolov3-tiny.pt \
  --output-root experiments/runs/reproduced_baseline_results
.venv-yolov3/bin/python scripts/export_yolov3.py \
  --source experiments/runs/vendor/yolov3 \
  --weights experiments/runs/yolov3_tiny_reproduced/weights/best.pt \
  --split val --output-dir experiments/runs/reproduced_baseline_results/val
```

The reproduction script checks the source commit and pretrained weight hash. It disables CUDA AMP because the legacy trainer's AMP run produced nearly unchanged validation metrics on this GPU; the FP32 run learned normally. [The diagnosis and scores](../results/baseline_reproduction/README.md) are recorded separately from the original MPS result.

## 4. Verify the published final test result

The [lock file](../results/test_protocol.lock.json) stores the selected checkpoint hashes, validation confidence thresholds, split manifest hash and evaluator settings **before** test evaluation. The selected binaries are committed under `models/`. In a fresh clone with the prepared dataset, the following commands rescore the 26-image test split in fresh local output directories:

```bash
.venv/bin/python scripts/evaluate_test_yolo11.py \
  --output-dir experiments/runs/reproduced_test/yolo11n
.venv-yolov3/bin/python scripts/export_yolov3.py \
  --source experiments/runs/vendor/yolov3 \
  --weights models/yolov3_tiny_best.pt --split test \
  --operating-confidence 0.386606 \
  --output-dir experiments/runs/reproduced_test/yolov3_tiny
```

Both scripts verify the locked checkpoint and threshold, and refuse to overwrite their local run directories. The shared evaluator reports AP from detections above 0.001; test precision/recall use the **fixed validation thresholds**, never a threshold optimized on test. [Published comparison](../results/final_test/README.md) includes full predictions and metrics for exact re-scoring without running the models.

For local batch-1 speed measurements on the validation split, use `scripts/benchmark_validation.py --model yolo11n` in `.venv` and `--model yolov3-tiny --source experiments/runs/vendor/yolov3` in `.venv-yolov3`. The [saved timing records](../results/benchmarks/) identify precision, batch size, image count, framework version and GPU. These are native pipeline timings and exclude model loading and disk I/O.
