#!/usr/bin/env python3
"""Reproduce the missing YOLOv3-tiny baseline checkpoint on the local GPU.

The published validation metrics came from an MPS run whose checkpoint is not
in Git. This script uses its pinned source, COCO weights, data, hyperparameters,
and seed, while recording that the hardware/run is different.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "experiments/runs"
PUBLIC = ROOT / "results/baseline_reproduction"
SOURCE_COMMIT = "97493de334161fe37c630a4fcf1c83c064526d63"
PRETRAINED_SHA256 = "01342ff2a552509c9c8f1cf52b2f78ab40b9de9ae606478cdb993385ba525ad4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Pinned ultralytics/yolov3 checkout")
    p.add_argument("--weights", type=Path, required=True, help="v9.6.0 yolov3-tiny.pt")
    p.add_argument("--output-root", type=Path, default=PUBLIC,
                   help="Metadata directory; choose a fresh local path when reproducing published results")
    args = p.parse_args()
    source, weights = args.source.resolve(), args.weights.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == SOURCE_COMMIT
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    assert weights.is_file() and sha256(weights) == PRETRAINED_SHA256
    assert torch.cuda.is_available()
    public = args.output_root.resolve()
    assert not (public / "training.json").exists(), f"Result already exists: {public}"
    run = LOCAL / "yolov3_tiny_reproduced"
    assert not run.exists(), f"Already exists: {run}"
    prepared = (ROOT / "data" if (ROOT / "data/processed/images/train").is_dir()
                else ROOT / "second_approach/data/data")
    data_dir = prepared / "processed"
    assert (data_dir / "images/train").is_dir()
    LOCAL.mkdir(parents=True, exist_ok=True)
    data_yaml = LOCAL / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(data_dir.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 1, "names": {0: "person"}}))
    hyp = yaml.safe_load((source / "data/hyps/hyp.scratch-low.yaml").read_text())
    hyp.update(lr0=0.001, lrf=0.01, momentum=0.9, weight_decay=0.0005,
               warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0)
    hyp_path = LOCAL / "yolov3_hyp.yaml"
    hyp_path.write_text(yaml.safe_dump(hyp, sort_keys=False))
    launcher = LOCAL / "launch_yolov3.py"
    launcher.write_text("""import os, sys, runpy
from pathlib import Path
repo, script = Path(sys.argv[1]).resolve(), sys.argv[2]
sys.argv = [str(repo / script)] + sys.argv[3:]
sys.path.insert(0, str(repo))
for name in ('wandb', 'clearml', 'comet_ml', 'albumentations'):
    sys.modules[name] = None
os.chdir(repo)
if os.environ.get('YOLOV3_DISABLE_AMP') == '1':
    import utils.general as general
    general.check_amp = lambda model: False
runpy.run_path(str(repo / script), run_name='__main__')
""")
    command = [sys.executable, str(launcher), str(source), "train.py",
        "--weights", str(weights), "--cfg", str(source / "models/yolov3-tiny.yaml"),
        "--data", str(data_yaml), "--hyp", str(hyp_path), "--imgsz", "640",
        "--epochs", "50", "--batch-size", "16", "--optimizer", "AdamW",
        "--seed", "42", "--device", "0", "--workers", "0", "--patience", "0",
        "--project", str(LOCAL), "--name", run.name]
    env = os.environ.copy()
    env.update(WANDB_MODE="disabled", COMET_MODE="DISABLED", MPLBACKEND="Agg",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", YOLO_AUTOINSTALL="false",
               PYTHONUNBUFFERED="1", YOLOV3_DISABLE_AMP="1")
    start = time.perf_counter()
    with (LOCAL / "yolov3_training.log").open("w") as log:
        result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - start
    assert result.returncode == 0, f"Training failed; see {LOCAL / 'yolov3_training.log'}"
    best = run / "weights/best.pt"
    assert best.is_file()
    history = pd.read_csv(run / "results.csv")
    assert len(history) == 50
    public.mkdir(parents=True, exist_ok=True)
    history.to_csv(public / "epochs.csv", index=False)
    (public / "hyperparameters.yaml").write_text(hyp_path.read_text())
    record = {"model": "YOLOv3-tiny", "reproduced": True,
        "source_commit": commit, "pretrained_sha256": sha256(weights),
        "checkpoint_sha256": sha256(best), "epochs": 50, "batch": 16,
        "imgsz": 640, "seed": 42, "optimizer": "AdamW", "lr0": 0.001,
        "training_seconds": seconds, "device": torch.cuda.get_device_name(0),
        "mixed_precision": False,
        "precision_reason": "CUDA AMP diagnostic produced static validation metrics; FP32 diagnostic learned normally",
        "python": platform.python_version(), "torch": torch.__version__,
        "command": command}
    (public / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"checkpoint_sha256": record["checkpoint_sha256"],
                      "training_seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
