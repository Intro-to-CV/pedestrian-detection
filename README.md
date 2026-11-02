# Pedestrian detection on Penn–Fudan

**Question:** How does a fine-tuned YOLO11n compare with YOLOv3-tiny for detecting pedestrians, and does mosaic augmentation help YOLO11n?

We converted Penn–Fudan instance masks to one `person` box per pedestrian, letterboxed images to 640×640, and fixed a seeded **119 train / 25 validation / 26 test** split. Both detectors start from COCO-pretrained weights. The test split was used only after checkpoint and confidence-threshold choices were [locked from validation](results/test_protocol.lock.json).

## Results at a glance

Both test rows use the same 26 images (66 people), the same [shared evaluator](scripts/evaluate.py), and each model's **validation-selected** confidence threshold.

| Model | Test mAP@0.5 | Test mAP@0.5:0.95 | Precision | Recall | Batch-1 time/image* |
|---|---:|---:|---:|---:|---:|
| YOLOv3-tiny, FP32 reproduction | 0.917 | 0.549 | 0.881 | **0.894** | 10.55 ms |
| YOLO11n | **0.936** | **0.780** | **0.966** | 0.864 | **9.53 ms** |

\*FP32 native validation timing on the same RTX 3050 GPU and 25 validation images; it is separate from the test scoring pass. [Full test results and caveats](results/final_test/README.md).

For YOLO11n, a controlled two-seed mosaic ablation gave mean validation mAP@0.5:0.95 of **0.868 with mosaic** and **0.862 without**. The ranges overlap, so this is a small and uncertain difference. [Ablation details](results/ablation/README.md).

## Quick start

1. Install Python 3.10–3.13 and the YOLO11n dependencies in an environment with a suitable PyTorch build:

   ```bash
   python -m venv .venv
   .venv/bin/python -m pip install -r requirements-yolo11.txt
   ```

2. Put the official Penn–Fudan `PNGImages/` and `PedMasks/` directories somewhere below `data/raw/`, then prepare and check the dataset:

   ```bash
   .venv/bin/python scripts/prepare_dataset.py --raw data/raw
   .venv/bin/python scripts/check_dataset.py
   ```

   The [official dataset page](https://www.cis.upenn.edu/~jshi/ped_html/) and [data preparation notes](docs/data_preparation.md) explain the source and conversion. Raw and processed images are intentionally absent from Git.

3. Inspect the executed [YOLO11n notebook](second_approach/YOLO11n_PennFudan_second_approach.ipynb), the [selected checkpoints](models/README.md), and the [four validation failure examples](results/failures/README.md). See [reproduction commands](docs/reproduce.md) to rerun training, evaluation, timing, and the ablation.

   To preview the selected model on a validation image, run `.venv/bin/python scripts/demo.py`; it writes `demo_output/yolo11n_example.png`. A [ready-made preview](results/demo/example.png) is included.

## What is in the repository?

| Path | Purpose |
|---|---|
| `scripts/prepare_dataset.py`, `scripts/check_dataset.py` | Convert masks, make the fixed split, verify labels |
| `baseline/` | Original YOLOv3-tiny notebook and progress-check validation result |
| `second_approach/` | Executed YOLO11n notebook and its validation artifacts |
| `scripts/run_ablation.py`, `results/ablation/` | Mosaic on/off, seeds 42 and 43, with mean and range |
| `scripts/evaluate.py`, `results/test_protocol.lock.json` | Common scoring and validation-only test choices |
| `models/`, `results/final_test/` | Exact selected checkpoints, test predictions, metrics, and timing |
| `results/failures/` | Four visual validation error cases with measured causes |
| `results/figures/`, `results/demo/` | Comparison charts and a ready-to-show detection preview |

## Comparison limits

The original report's YOLOv3-tiny checkpoint was unavailable, so test evaluation uses a [new FP32 reproduction](results/baseline_reproduction/README.md) whose validation score is close to the reported one. The models have different architectures, losses, framework versions, and learning-rate schedules; the YOLOv3 reproduction also needed FP32 on this GPU after CUDA AMP failed to learn. The mosaic ablation is the controlled experiment; the cross-model comparison is not an architecture-only causal test. With 25 validation and 26 test images, estimates are noisy, and the image-level split can contain related scenes across sets.
