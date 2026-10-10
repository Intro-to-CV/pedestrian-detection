# YOLOv3-tiny baseline

[Training notebook](YOLOv3_tiny_PennFudan_baseline.ipynb) for fine-tuning COCO-pretrained YOLOv3-tiny on Penn–Fudan with one class: `person`.

Requires Python 3.10–3.13, Git and Jupyter. A CUDA GPU is recommended. The notebook installs training dependencies and records the installed versions. Run the cells in order.

Input is the prepared `data/processed` and `data/splits` folders. Set `DATA_ROOT` if they live elsewhere. Alternatively, put `pedestrian_data.zip` in the repository root or home directory.

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

The last cell bundles the outputs into a ZIP. Use a new `RUN_NAME` for each experiment. Logs and checkpoints are generated when training runs; the full 50-epoch run has not been executed yet.
