# Pedestrian detection

Penn–Fudan pedestrian detection experiments: reproducible data preparation,
YOLOv3-tiny fine-tuning, and a shared evaluation protocol for YOLOv3-tiny and
YOLO11n.

## Dataset preparation

Download and extract Penn–Fudan so `PNGImages/` and `PedMasks/` are below
`data/raw/`, then run:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py --raw data/raw
python scripts/check_dataset.py
```

This creates a 640×640 letterboxed dataset, one `person` class in YOLO format,
and a deterministic `119 / 25 / 26` train/validation/test split (seed 42).
The raw archive and processed images are intentionally ignored by Git.

See [data preparation](docs/data_preparation.md) and the
[YOLOv3-tiny baseline](baseline/README.md) for the complete procedures.

## Shared evaluation

`scripts/evaluate.py` consumes 640×640 YOLO labels and predictions in the
portable `person_xyxy_pixels_v1` JSON schema. It calculates mAP@0.5,
mAP@0.5:0.95, precision and recall at the best F1 operating point, and emits a
PR curve plus an FP/FN review CSV. Use the same script and inputs for both
models.
