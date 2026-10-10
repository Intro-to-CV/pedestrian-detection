# YOLOv3-tiny baseline

Fine-tune COCO-pretrained YOLOv3-tiny on Penn–Fudan for one-class `person` detection.

Open [the notebook](YOLOv3_tiny_PennFudan_baseline.ipynb) in Jupyter or Colab and run the cells in order. Use Python 3.10–3.13 and preferably a CUDA GPU. The setup cell installs training dependencies and pins the source revision. For local use, install Jupyter with `python -m pip install jupyterlab`, then run `jupyter lab` from the repository root.

Expected prepared data:

```text
data/processed/images/{train,val,test}/
data/processed/labels/{train,val,test}/
data/splits/{train,val,test}.txt
data/splits/statistics.json
```

Set `DATA_ROOT` if these folders live elsewhere. Alternatively, put `pedestrian_data.zip` in the repository root or your home directory; in Colab, upload it to `/content`. The notebook extracts it under `baseline/runs/` and verifies the existing labels and split.

Training uses 640 px, 50 epochs, batch 16, AdamW, initial learning rate 0.001 and seed 42. There are 119 training images, 25 validation images and 26 held-out test images. Augmentation applies only to training. The test set is checked for data integrity but is not used for inference or model selection.

Outputs include the best checkpoint, epoch logs, validation predictions, native validation metrics, actual settings, environment details, training time and recorded errors. The notebook bundles these into a ZIP. Generated runs stay out of Git. Choose a new `RUN_NAME` for each experiment and copy outputs out of Colab before ending the session.

The notebook documents gradient accumulation and warmup settings for comparison with YOLO11n. The full 50-epoch experiment remains to be run.
