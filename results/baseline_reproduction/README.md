# YOLOv3-tiny baseline reproduction

The report's original YOLOv3-tiny checkpoint was trained on Apple MPS and was not available locally or in Git. To evaluate both approaches on test, we reproduced the 50-epoch run from the [baseline notebook](../../baseline/YOLOv3_tiny_PennFudan_baseline.ipynb) with its pinned `ultralytics/yolov3` commit, v9.6.0 COCO weights, Penn–Fudan split, batch 16, AdamW, LR 0.001, weight decay 0.0005, seed 42, low augmentation and 640 px input.

On this RTX 3050 GPU, the legacy trainer's automatic CUDA AMP check passed, but the first 50-epoch run's validation metrics stayed nearly fixed and mAP@0.5:0.95 ended at 0.0184. A 10-epoch FP32 diagnostic reached native mAP@0.5:0.95 of 0.4304, confirming that the setup could learn without CUDA AMP. The full reproduction therefore **disabled AMP** while leaving the source revision and requested hyperparameters unchanged. The failed run and diagnostic remain local under `experiments/runs/diagnostics/`; they are not used in the final comparison.

| Validation result | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| Original MPS checkpoint, progress report | 0.9658 | 0.6211 |
| New FP32 GPU reproduction | 0.9733 | 0.6264 |

The close scores do not make the checkpoints identical. Final test metrics use the new checkpoint in [models/](../../models/README.md), with its **own** validation-selected confidence threshold 0.386606. The saved [training record](training.json), [hyperparameters](hyperparameters.yaml), [epoch history](epochs.csv), [validation metrics](val/metrics.json), [predictions](val/predictions.json), and [timing](val/timing.json) support the result. The baseline's saved `opt.yaml` has `cos_lr: false`; YOLO11n uses `cos_lr: true`.
