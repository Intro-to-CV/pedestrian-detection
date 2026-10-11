# pedestrian-detection

Fine-tuning COCO-pretrained YOLOv3-tiny (baseline) and YOLO11n on Penn-Fudan.

## Dataset

Penn-Fudan Database for Pedestrian Detection and Segmentation: 170 images, one class (`person`).
Annotations are instance masks (pixel value 0 = background, k = k-th pedestrian).

### Get the data
Download from Kaggle (`psvishnu/pennfudan-database-for-pedestrian-detection-zip`) or from the
[official page](https://www.cis.upenn.edu/~jshi/ped_html/) and unzip so that `PNGImages/` and `PedMasks/`
are somewhere inside `data/raw/`, e.g. `data/raw/PennFudanPed/PNGImages/...`.

### Prepare
```bash
pip install -r requirements.txt
python scripts/prepare_dataset.py --raw data/raw   # masks -> boxes -> YOLO labels, 640x640, split, stats
python scripts/check_dataset.py                    # re-checks files on disk, draws boxes
```
Then open `results/figures/dataset_examples/*_grid.png` and verify that boxes sit on pedestrians.

### What is produced
| Path | Content | 
|---|---|
| `data/processed/images,labels/{train,val,test}` | 640x640 images + YOLO labels | 
| `data/splits/{train,val,test}.txt` | split manifests (image names) | 
| `data/splits/statistics.json` | counts, boxes per image, box heights, size bins | 
| `configs/dataset.yaml` | Ultralytics dataset config (run from repo root) | 

### Protocol
* Sorted filenames -> `random.Random(42).shuffle` -> train `N*70//100`, val `N*15//100`, test = rest
  (170 -> 119 / 25 / 26). The script asserts that the splits are disjoint and cover all images.
* Resize: letterbox to 640x640 (aspect ratio preserved, grey padding); boxes are transformed identically.
* Boxes: tight rectangle around all pixels of each instance mask.
* Augmentation: not stored on disk. Ultralytics augments on the fly, training split only.
* Note: neighbouring photos can show the same scene, so an image-level split does not guarantee scene independence.
