# YOLOv3-tiny baseline

[Training notebook](YOLOv3_tiny_PennFudan_baseline.ipynb) for fine-tuning COCO-pretrained YOLOv3-tiny on Penn–Fudan with one class: `person`.

Requires Python 3.10–3.13, Git and Jupyter. A CUDA GPU is recommended. The notebook installs training dependencies and records the installed versions. Run the cells in order.

Input is the prepared root-level `data/processed` and `data/splits` folders. Set `DATA_ROOT` only if they live elsewhere. Alternatively, put `pedestrian_data.zip` in the repository root or home directory.

Training uses 119 images, 640 px, 50 epochs, batch 16, AdamW, initial learning rate 0.001 and seed 42. Validation uses 25 images. The 26 test images are excluded from training and validation. Augmentation applies only to training.

Outputs are saved under `baseline/runs/artifacts/<RUN_NAME>/`:

| File | Contents |
|---|---|
| `training/weights/best.pt` | Best validation checkpoint |
| `training/results.csv` | Epoch losses, metrics and learning rates |
| `train_*.log` | Training console output and warnings |
| `train_runtime.json` | Timestamps, duration, status and exit code |
| `errors.jsonl` | Exceptions and failed commands, if any |
| `training_summary.json` | Completed epochs, saved parameters, runtime and checkpoint hash |
| `training/opt.yaml`, `training/hyp.yaml` | Trainer settings |
| `validation/predictions.json`, `validation/labels/` | Validation predictions |
| `environment.json`, `requirements-freeze.txt` | Hardware and package versions |

The last cell bundles the outputs into a ZIP. Use a new `RUN_NAME` for each experiment. Logs and checkpoints are generated when training runs.

The recorded `yolov3_tiny_seed42` run completed all 50 epochs on Apple Metal
(MPS) in 738.94 seconds. Its shared evaluator result on the 25-image validation
split was mAP@0.5 **0.9658**, mAP@0.5:0.95 **0.6211**, precision **0.8889** and
recall **0.9552** at the best-F1 confidence threshold. See
`runs/artifacts/yolov3_tiny_seed42/validation/evaluation_summary.md` for the
table and corresponding machine-readable artifacts.

## Common evaluation

After the notebook has written `validation/predictions.json`, evaluate it with the framework-independent protocol used for both YOLOv3-tiny and YOLO11n:

```bash
python scripts/evaluate.py \
  --labels data/processed/labels/val \
  --predictions baseline/runs/artifacts/<RUN_NAME>/validation/predictions.json \
  --output-dir baseline/runs/artifacts/<RUN_NAME>/validation/common_evaluation \
  --model YOLOv3-tiny --split val
```

It saves `metrics.json`, `metrics.csv`, `pr_curve.png`, and `error_review.csv`. YOLO11n must export the same `person_xyxy_pixels_v1` prediction schema before using this command.
