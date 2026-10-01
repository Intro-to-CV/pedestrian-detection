# Baseline Detection Approach

## Model

The baseline detector is **Ultralytics YOLO11n**.

- Architecture: YOLO11n
- Task: object detection
- Pretrained checkpoint: `yolo11n.pt`
- Pretraining dataset: COCO
- Target class: pedestrian (`person`)
- Input resolution: 640 × 640

YOLO11n was selected because it is a lightweight pretrained detector that is fast to train and evaluate while still providing a strong baseline for pedestrian detection.

## Initial Training Configuration

| Parameter | Value |
|---|---|
| Model | `yolo11n.pt` |
| Image size | 640 × 640 |
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0005 |
| Training seed | 42 |
| Mixed precision | Enabled |
| Pretrained weights | Enabled |

No custom augmentation strategy is introduced in the baseline.

The dataset split itself is defined separately in Issue #1. The test split will not be used for training or hyperparameter selection.

## Initial Training Command

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data/penn_fudan.yaml",
    imgsz=640,
    epochs=50,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    seed=42,
    amp=True,
    device=0,
)
```

The dataset configuration path may be adjusted after Issue #1 is finalized.

## Baseline Implementation Plan

1. Use the train/validation/test split prepared in Issue #1.
2. Convert annotations to YOLO format if needed.
3. Create the dataset YAML configuration.
4. Load the pretrained `yolo11n.pt` checkpoint.
5. Fine-tune YOLO11n on the training split.
6. Use the validation split for development and model selection.
7. Save the best validation checkpoint.
8. Evaluate the final model on the held-out test split.
9. Save predictions and metrics for the common evaluation protocol.
10. Keep this configuration fixed when comparing with the second approach, except for the experimental variable being tested.

## Reproducibility

The baseline configuration will be kept fixed. Package versions and runtime environment will be recorded during implementation.

Evaluation metrics, inference-speed measurement, and false-positive/false-negative analysis are defined in the separate evaluation-protocol issue.
