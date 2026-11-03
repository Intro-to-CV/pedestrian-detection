# YOLO11n demo

[Example output](example.png) shows the selected YOLO11n checkpoint on **FudanPed00025 from validation**, using its validation-selected confidence 0.361187. It is a preview, separate from test evaluation. One crowded pedestrian is missed at this threshold; see the [ground-truth failure overlay](../failures/FudanPed00025.png).

After preparing the dataset and installing the YOLO11n environment, regenerate it with:

```bash
.venv/bin/python scripts/demo.py
```

Pass `--image path/to/photo.png --output demo_output/my_photo.png` to try another image. The script verifies the checkpoint against the locked hash and writes a visualization without changing any reported metrics.
