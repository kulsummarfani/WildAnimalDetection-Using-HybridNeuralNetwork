"""Flask web application for the VanRakshak AI detector and dashboard."""

import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MEDIA_DIR = PROJECT_ROOT / ".runtime" / "web-media"
UPLOAD_DIR = PROJECT_ROOT / ".runtime" / "web-uploads"
ALERTS_LOG = PROJECT_ROOT / ".runtime" / "alerts_log.json"
YOLO_ROOT = PROJECT_ROOT / "yolov5"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords

from classify import predict_species
from gradcam import make_gradcam_overlay
from threat import get_threat_tier, should_alert

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"
IMG_SIZE = (196, 196)
CONF_THRES = 0.25
IOU_THRES = 0.45
_device = torch.device("cpu")
_model = None
_names = None
_img_size = IMG_SIZE

DEMO_USERS = {
    ("admin", "demo123"): "admin",
    ("watchman1", "demo123"): "watchman",
    ("resident1", "demo123"): "resident",
}


def load_model():
    global _model, _names, _img_size
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Detector weights not found: {MODEL_PATH}")
        _model = DetectMultiBackend(str(MODEL_PATH), device=_device, dnn=False, data=None, fp16=False)
        _names = _model.names
        _img_size = check_img_size(IMG_SIZE, s=_model.stride)
        _model.warmup(imgsz=(1, 3, *_img_size))
    return _model, _names


def estimate_movement_and_distance(box, frame_shape):
    x1, y1, x2, y2 = box
    height = frame_shape[0]
    box_height_ratio = (y2 - y1) / float(height) if height else 0
    if box_height_ratio > 0.5:
        return "approaching", 15
    if box_height_ratio > 0.25:
        return "approaching", 60
    if box_height_ratio > 0.1:
        return "stationary", 150
    return "moving away", 300


def read_alerts():
    if not ALERTS_LOG.exists():
        return []
    try:
        return json.loads(ALERTS_LOG.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def write_alerts(alerts):
    ALERTS_LOG.write_text(json.dumps(alerts, indent=2))


def build_alert(detection, role):
    species = detection["label"]
    confidence = detection["confidence"]
    distance = detection["distance_m"]
    movement = detection["movement"]
    tier = "watchman" if role in ("watchman", "admin") else "resident"
    if movement == "approaching":
        if tier == "watchman":
            message = (f"{species} detected at {confidence * 100:.1f}% confidence, "
                       f"approximately {distance}m from the boundary and moving toward the village.")
        else:
            message = (f"{species} spotted approximately {distance}m from the village boundary. "
                       "Stay indoors and alert neighbours.")
    else:
        message = f"{species} detected approximately {distance}m away. Stay alert and monitor the feed."
    return {"tier": tier, "message": message}


def detect_image(image_path):
    model, names = load_model()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    original_shape = image.shape[:2]
    prepared = letterbox(image, _img_size, stride=model.stride, auto=model.pt)[0]
    prepared = prepared.transpose((2, 0, 1))[::-1]
    prepared = np.ascontiguousarray(prepared)
    tensor = torch.from_numpy(prepared).to(_device).float() / 255.0
    tensor = tensor[None]

    with torch.no_grad():
        predictions = model(tensor, augment=False, visualize=False)
        predictions = non_max_suppression(predictions, CONF_THRES, IOU_THRES, None, False, max_det=50)

    detections = []
    annotated = image.copy()
    for batch_detections in predictions:
        if not len(batch_detections):
            continue
        batch_detections[:, :4] = scale_coords(tensor.shape[2:], batch_detections[:, :4], image.shape).round()
        for *coordinates, detector_confidence, detector_class in reversed(batch_detections):
            box = [int(value) for value in coordinates]
            x1, y1, x2, y2 = box
            crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                continue
            try:
                evidence, species, confidence = make_gradcam_overlay(crop)
            except (OSError, ValueError, RuntimeError):
                species, confidence = predict_species(crop)
                evidence = crop
            evidence_name = f"gradcam-{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(MEDIA_DIR / evidence_name), evidence)
            movement, distance = estimate_movement_and_distance(box, original_shape)
            detections.append({
                "label": species,
                "detector_label": str(names[int(detector_class)]),
                "confidence": round(float(confidence), 4),
                "detector_confidence": round(float(detector_confidence), 4),
                "bbox": [float(value) for value in coordinates],
                "movement": movement,
                "distance_m": distance,
                "threat_tier": get_threat_tier(species),
                "evidence_url": f"/media/{evidence_name}",
            })
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (46, 184, 114), 2)
            cv2.putText(annotated, f"{species} {confidence * 100:.1f}%", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 184, 114), 2)

    annotated_name = f"detection-{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(MEDIA_DIR / annotated_name), annotated)
    return detections, f"/media/{annotated_name}"


@app.post("/api/detect")
def api_detect():
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "No image uploaded"}), 400
    role = request.form.get("role", "resident")
    if role not in {"resident", "watchman", "admin"}:
        role = "resident"
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
    request.files["image"].save(upload_path)
    try:
        detections, annotated_url = detect_image(upload_path)
    except (OSError, ValueError, RuntimeError) as exc:
        return jsonify({"error": f"Detection failed: {exc}"}), 500
    finally:
        upload_path.unlink(missing_ok=True)

    alert = None
    alert_candidates = [item for item in detections if should_alert(item["label"], item["confidence"])]
    if alert_candidates:
        top = max(alert_candidates, key=lambda item: item["confidence"])
        alert = build_alert(top, role)
        alerts = read_alerts()
        alerts.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "species": top["label"],
            "confidence": top["confidence"],
            "distance_m": top["distance_m"],
            "movement": top["movement"],
            "tier": alert["tier"],
            "message": alert["message"],
        })
        write_alerts(alerts)
    return jsonify({"detections": detections, "alert": alert, "annotated_url": annotated_url})


@app.get("/api/alerts")
def api_alerts():
    return jsonify({"alerts": read_alerts()})


@app.post("/api/login")
def api_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    role = DEMO_USERS.get((username, data.get("password", "")))
    if role is None:
        return jsonify({"success": False, "error": "Invalid username or password"}), 401
    if role != data.get("role"):
        return jsonify({"success": False, "error": f"That account is a '{role}' account"}), 401
    return jsonify({"success": True, "role": role, "username": username})


@app.get("/media/<path:filename>")
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    print("Loading detection model...")
    load_model()
    print("VanRakshak AI running at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
