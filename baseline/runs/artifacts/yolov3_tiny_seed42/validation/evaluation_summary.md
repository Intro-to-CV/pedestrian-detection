# YOLOv3-tiny validation summary

Dataset: Penn–Fudan validation split (25 images, 67 pedestrian instances).
Checkpoint: `training/weights/best.pt`.

| Metric | Shared evaluator result |
|---|---:|
| mAP@0.5 | 0.9658 |
| mAP@0.5:0.95 | 0.6211 |
| Precision at best F1, IoU 0.5 | 0.8889 |
| Recall at best F1, IoU 0.5 | 0.9552 |
| Best-F1 confidence | 0.2935 |

The accompanying `metrics.json`/`metrics.csv` are the source of these values.
`pr_curve.png` is the PR curve. `error_review.csv` records false positives and
false negatives at IoU 0.5 and confidence >= 0.25.
