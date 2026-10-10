# YOLOv3-tiny baseline

Abdulloh's part for 13 October: fine-tune the baseline and share its checkpoint, logs and validation predictions.

Open [the notebook](YOLOv3_tiny_PennFudan_baseline.ipynb) in Jupyter or Colab and run the cells in order. Use a GPU runtime if available. The setup cell installs the training dependencies and pins the YOLOv3 source revision. For a local notebook editor, install Jupyter with `python -m pip install jupyterlab` and start it with `jupyter lab` from the repository root.

The notebook uses Marat's prepared folders:

```text
data/processed/images/{train,val,test}/
data/processed/labels/{train,val,test}/
data/splits/{train,val,test}.txt
data/splits/statistics.json
```

These come from the `data-prep` branch's preparation workflow. If the folders are not available yet, put the supplied `pedestrian_data.zip` in the repository root or your home directory. In Colab, upload it to `/content`. The notebook extracts it under `baseline/runs/`; it does not rebuild the labels or split. Set `DATA_ROOT` if the prepared folders live elsewhere.

The training settings are 640 px, 50 epochs, batch 16, AdamW, initial learning rate 0.001 and seed 42. The COCO-pretrained YOLOv3-tiny is adapted to one `person` class. Training uses 119 images, validation uses 25, and the 26 test images are only checked for data integrity. Augmentation is applied only to training.

The notebook explains upstream defaults that matter for the later YOLO11n comparison, including gradient accumulation and the minimum 100-step warmup. Actual settings, package versions, timings and errors are saved with each run. Choose a new `RUN_NAME` for a new experiment; existing training outputs are not overwritten.

After training, send the generated handoff ZIP to Ksenia and Timur. It includes `best.pt`, epoch logs, validation predictions, native validation diagnostics, configuration files and environment details. Ksenia owns the shared evaluation and failure analysis; Timur owns the report. The full 50-epoch experiment has not been run yet.

The original notebook passed a reduced one-epoch training smoke test and a 640 px validation/export check. The repository version also has syntax and dataset-input checks. These checks are not baseline experiment results.

Data preparation belongs to Marat's branch. This folder adds no dataset files, preprocessing scripts or trained weights. Keep generated runs out of Git.
