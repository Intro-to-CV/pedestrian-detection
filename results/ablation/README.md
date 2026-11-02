# Mosaic ablation (validation only)

YOLO11n was trained with mosaic **on** (`mosaic=1.0`) and **off** (`mosaic=0.0`) at seeds 42 and 43. Both conditions used the same Penn–Fudan 119/25 train/validation split, COCO-pretrained weight hash, image size 640, 50 epochs, batch 16, AdamW, learning rate 0.001, cosine schedule, augmentation settings other than mosaic, and [shared evaluator](../../scripts/evaluate.py). The completed seed-42 mosaic-on run was reused; its checkpoint hash was verified. The other three runs were made on the same RTX 3050 GPU with Ultralytics 8.3.228.

| Mosaic | Seed 42 mAP@0.5:0.95 | Seed 43 mAP@0.5:0.95 | Mean | Range |
|---|---:|---:|---:|---:|
| On | 0.8655 | 0.8702 | **0.8679** | 0.8655–0.8702 |
| Off | 0.8733 | 0.8502 | 0.8618 | 0.8502–0.8733 |

The mean difference is **0.0061** in favor of mosaic-on, while seed 42 alone favors mosaic-off. With just two seeds and 25 validation images, this is weak evidence about the augmentation's general benefit. The predeclared selection rule was higher mean mAP@0.5:0.95, with mean mAP@0.5 as a tie-breaker, then the seed-42 checkpoint of the chosen condition. It selects **mosaic-on, seed 42** for final test evaluation.

[Per-run values](runs.csv), [summary and selection](summary.json), and each run's `result.json`, `epochs.csv`, and compressed `predictions.json.gz` are included. The raw checkpoints stay local under `experiments/runs/` (or `second_approach/runs/` for the reused run). Reproduce with `python scripts/run_ablation.py --seed 43 --mosaic off` and the other seed/condition combinations; the script refuses to overwrite an existing run. The compressed predictions allow exact shared-evaluator rescoring without the checkpoints.
