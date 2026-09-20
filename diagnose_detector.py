"""Find out why the detector misses an animal. Read-only: changes nothing in the project.

Runs the same detector the web app uses on each image, at several input sizes, with the confidence
threshold dropped to 0.01, and prints the top raw detections. Then you can tell whether the animal is
(a) not seen at all, (b) seen but below the app's 0.25 threshold, or (c) only seen at a larger input size.

Usage:  .venv/bin/python diagnose_detector.py path/to/image.jpg [another.jpg ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolov5"))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords  # noqa: E402

APP_SIZE = 224          # what webapp.py uses (196 is rounded up to 224)
APP_CONF = 0.25         # webapp.py CONF_THRES
SIZES = (224, 320, 416, 640)
FLOOR_CONF = 0.01
TOP_N = 3

model = DetectMultiBackend(str(ROOT / "model" / "best.pt"), device=torch.device("cpu"), dnn=False, data=None, fp16=False)
names = model.names


def detect(image, size):
    imgsz = check_img_size((size, size), s=model.stride)
    model.warmup(imgsz=(1, 3, *imgsz))
    prepared = letterbox(image, imgsz, stride=model.stride, auto=model.pt)[0]
    prepared = np.ascontiguousarray(prepared.transpose((2, 0, 1))[::-1])  # BGR -> RGB, HWC -> CHW
    tensor = torch.from_numpy(prepared).float()[None] / 255.0
    with torch.no_grad():
        prediction = model(tensor, augment=False, visualize=False)
        found = non_max_suppression(prediction, FLOOR_CONF, 0.45, None, False, max_det=20)[0]
    if len(found):
        found[:, :4] = scale_coords(tensor.shape[2:], found[:, :4], image.shape).round()
    return found


def main(paths):
    if not paths:
        sys.exit(__doc__)
    for path in paths:
        # YOLOv5 patches cv2.imread to raise on a missing file, so check first for a readable message.
        if not Path(path).expanduser().is_file():
            print(f"\n{path}: file not found. Use the real path to the image (drag it into Terminal to paste it).")
            continue
        image = cv2.imread(str(Path(path).expanduser()))
        if image is None:
            print(f"\n{path}: found, but it could not be decoded as an image")
            continue
        print(f"\n=== {Path(path).name}  ({image.shape[1]}x{image.shape[0]}) ===")
        for size in SIZES:
            found = detect(image, size)
            tag = "  <- what the app uses" if size == APP_SIZE else ""
            print(f"input {size}px{tag}")
            if not len(found):
                print("    nothing above 0.01")
                continue
            for *box, conf, cls in found[:TOP_N].tolist():
                verdict = "ALERT-LEVEL" if conf >= APP_CONF else f"below app threshold ({APP_CONF})"
                x1, y1, x2, y2 = (int(v) for v in box)
                print(f"    {names[int(cls)]:11s} {conf:.2f}  box {x1},{y1}-{x2},{y2}  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])