# Held-out test results

The [protocol lock](../test_protocol.lock.json) was written before these 26 test images (66 pedestrians) were evaluated. It fixes both checkpoint hashes and the confidence thresholds chosen on validation. Both rows use the same prepared images and [shared 101-point AP evaluator](../../scripts/evaluate.py). AP uses a 0.001 detection floor; precision and recall use each model's **validation-selected** threshold.

| Model | Val mAP@0.5:0.95 | Test mAP@0.5 | Test mAP@0.5:0.95 | Test precision | Test recall | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv3-tiny, FP32 reproduction | 0.6264 | 0.9172 | 0.5493 | 0.8806 | **0.8939** | 0.386606 |
| YOLO11n, mosaic-on seed 42 | **0.8655** | **0.9360** | **0.7800** | **0.9661** | 0.8636 | 0.361187 |

YOLO11n's test mAP@0.5:0.95 is **0.2306** higher. At the selected thresholds it has fewer false positives (2 versus 8), while YOLOv3-tiny finds two more of the 66 annotated pedestrians (59 versus 57 true positives). These are single-checkpoint estimates from a small image-level split; related scenes may cross splits.

## Runtime on one GPU

FP32 batch-1 native validation timing on the **same 25 validation images** and NVIDIA GeForce RTX 3050 Laptop GPU:

| Model | Preprocess | Inference | NMS | Sum per image | Approx. FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv3-tiny | 0.70 ms | 6.98 ms | 2.87 ms | 10.55 ms | 94.8 |
| YOLO11n | 0.48 ms | 7.90 ms | 1.14 ms | **9.53 ms** | **105.0** |

These are each framework's native stage timings, excluding model load and file I/O. They are useful local runtime measurements, not a hardware-independent speed claim. The 50-epoch training calls took 347.89 s for FP32 YOLOv3-tiny and 161.05 s for YOLO11n with AMP; the differing precision and trainer implementations also affect training time.

The original MPS YOLOv3 checkpoint from the progress report was unavailable. Its published validation result is retained in `baseline/`, while the **new FP32 reproduction** is used consistently for the test row above. The original YOLOv3 trainer uses linear LR decay; the YOLO11n run uses cosine decay. Architecture, losses, pretrained checkpoints and trainer details differ, so the two-model comparison is observational. The mosaic experiment isolates only the mosaic setting within YOLO11n.

[Machine-readable comparison](comparison.csv), [YOLO11n predictions and metrics](yolo11n/metrics.json), [YOLOv3-tiny predictions and metrics](yolov3_tiny_reproduced/metrics.json), and [batch-1 timing records](../benchmarks/) are included. The error CSVs in each model folder use **0.25** as a separate review threshold; their FP/FN totals should not be confused with the fixed-threshold precision/recall above.
