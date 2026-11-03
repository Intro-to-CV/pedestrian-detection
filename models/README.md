# Selected checkpoints

These are the exact checkpoints used for the locked test results and the demo. Both were selected using **validation only**; the test set was not used for checkpoint or threshold selection.

| File | Training run | SHA-256 |
|---|---|---|
| [yolo11n_best.pt](yolo11n_best.pt) | COCO-pretrained YOLO11n; mosaic-on, seed 42, 50 epochs | `d5aadfabcaa5b2c1409abcb9ad52b793bccd0abf3a5361c585cfb192dc28fc20` |
| [yolov3_tiny_best.pt](yolov3_tiny_best.pt) | Reproduced YOLOv3-tiny; seed 42, 50 epochs, FP32 on RTX 3050 | `041227e92f882177f663f09830d3e77955d905db69d869621eb17ac3a94a96db` |

The original report's YOLOv3-tiny checkpoint was trained on Apple MPS and was not available in the repository. The committed YOLOv3-tiny checkpoint is a new reproduction. Its validation mAP@0.5:0.95 is 0.6264 using the shared evaluator, close to the report's 0.6211, but it is not the same binary. See [the locked protocol](../results/test_protocol.lock.json) for the confidence thresholds and evaluator settings used on test.
