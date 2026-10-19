# YOLO11n second approach

[Executed notebook](YOLO11n_PennFudan_second_approach.ipynb) fine-tunes COCO-pretrained YOLO11n on the same Penn–Fudan split as the [YOLOv3-tiny baseline](../baseline/YOLOv3_tiny_PennFudan_baseline.ipynb). The train, validation, and test sets contain 119, 25, and 26 images. The test set was not evaluated for this checkpoint.

## Reproduce

Use Python 3.10–3.13 and install `ultralytics==8.3.228`, `torch`, `torchvision`, `numpy`, `pandas`, `PyYAML`, `matplotlib`, `Pillow`, `jupyter`, and `ipykernel`. Run the notebook from the repository root. Place the baseline's `pedestrian_data.zip` in the repository root or your home directory, or set `DATA_ROOT` in the notebook to the prepared dataset folder. Set `YOLO11_WEIGHTS` to a local COCO-pretrained `yolo11n.pt`, or allow Ultralytics to download it. All generated files under `second_approach/runs/` are local and ignored by Git; a second run needs a new `RUN_NAME`.

The completed run used Python 3.13.9, PyTorch 2.11.0+cu130, Ultralytics 8.3.228, and an NVIDIA GeForce RTX 3050 Laptop GPU. The full training call took **161.05 s**, including startup, checkpoint saving, and the trainer's final validation. A separate FP32 best-checkpoint validation took **3.66 s**. The selected checkpoint has 2,582,347 parameters.

## Settings and differences

Both runs use prepared 640×640 images, 50 epochs, batch 16, AdamW, initial learning rate 0.001, weight decay 0.0005, seed 42, one `person` class, 100-step minimum warmup, nominal batch 64, and COCO-pretrained weights. The baseline's code uses a **cosine** learning-rate schedule, despite its notebook text saying linear; this notebook sets `cos_lr=True`. It also requests matching HSV, translation, scale, flip, mosaic, mixup, and copy-paste values from `hyp.scratch-low.yaml`, and sets `close_mosaic=0` so mosaic continues through the final epochs. Validation requests FP32, confidence floor 0.001, NMS IoU 0.6, and at most 300 detections per image.

The model architecture, loss function, pretrained checkpoint, and trainer/evaluator implementations differ. The YOLOv3 notebook explicitly disables optional Albumentations, while YOLO11n uses whatever compatible augmentation packages are installed. The recorded `args.yaml` and environment are the definitive effective settings. The shared `scripts/evaluate.py` results below provide the directly comparable validation metrics; native Ultralytics metrics are also stored in the executed notebook.

## Validation results

Both rows use the same 25 images and the repository's shared 101-point AP evaluator. Precision and recall use each model's best-F1 confidence threshold at IoU 0.5.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|
| YOLOv3-tiny | 0.9658 | 0.6211 | 0.8889 | 0.9552 |
| YOLO11n | **0.9832** | **0.8655** | **0.9692** | 0.9403 |

YOLO11n improved mAP@0.5:0.95 by **0.2445** absolute on validation. Its recall at the selected confidence threshold is **0.0149** lower. See [comparison.csv](results/comparison.csv), [shared evaluator metrics](results/common_evaluation/metrics.json), [error review](results/common_evaluation/error_review.csv), and [prediction export](results/validation_predictions.json). The export excludes 82 zero-area boxes at image edges after clipping; their removal is recorded in the notebook output.
