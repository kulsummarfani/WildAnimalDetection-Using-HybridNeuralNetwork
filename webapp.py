"""Flask web application for the authenticated VanRakshak AI dashboard."""

import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, g, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MEDIA_DIR = PROJECT_ROOT / ".runtime" / "web-media"
UPLOAD_DIR = PROJECT_ROOT / ".runtime" / "web-uploads"
YOLO_ROOT = PROJECT_ROOT / "yolov5"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords

import supa
from alert import send_sms_alert
from classify import predict_species
from gradcam import make_gradcam_overlay
from threat import BIG_CATS, get_threat_tier, normalize_species, should_alert

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"
IMG_SIZE = (640, 640)
CONF_THRES = 0.25
IOU_THRES = 0.45
ALERT_HIGH_CONF = 0.40
ALERT_MEDIUM_CONF = 0.60
_device = torch.device("cpu")
_model = None
_names = None
_img_size = IMG_SIZE


def _bearer_token():
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def require_role(*roles):
    """Require a Supabase identity and optionally one of the listed roles."""
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            try:
                principal = supa.get_principal(_bearer_token())
            except supa.InvalidToken:
                return jsonify({"error": "Login required"}), 401
            except supa.SupabaseConfigError as exc:
                return jsonify({"error": str(exc)}), 503
            except Exception as exc:
                print(f"Auth service error: {exc}")
                return jsonify({"error": "Authentication service unavailable"}), 503
            if roles and principal["role"] not in roles:
                return jsonify({"error": "Your role is not allowed to do this"}), 403
            g.principal = principal
            return view(*args, **kwargs)
        return wrapper
    return decorator


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
    _, y1, _, y2 = box
    height = frame_shape[0]
    box_height_ratio = (y2 - y1) / float(height) if height else 0
    if box_height_ratio > 0.5:
        return "approaching", 15
    if box_height_ratio > 0.25:
        return "approaching", 60
    if box_height_ratio > 0.1:
        return "stationary", 150
    return "moving away", 300


def sms_enabled():
    return os.environ.get("SEND_SMS_ALERTS", "").strip().lower() in {"1", "true", "yes"}


_alert_lock = threading.Lock()


def sms_cooldown_seconds():
    """Return the per-species SMS cooldown; zero disables it."""
    try:
        return max(0, int(os.environ.get("SMS_COOLDOWN_SECONDS", "300")))
    except ValueError:
        return 300


def sms_cooldown_remaining(species):
    """Return seconds until this species may be texted again."""
    window = sms_cooldown_seconds()
    if window == 0:
        return 0
    now = datetime.now(timezone.utc)
    last = supa.last_sms_time(species, (now - timedelta(seconds=window)).isoformat())
    if not last:
        return 0
    sent_at = datetime.fromisoformat(last.replace("Z", "+00:00"))
    return max(0, math.ceil((sent_at + timedelta(seconds=window) - now).total_seconds()))


def maybe_send_sms(top):
    """Send once per species per cooldown window and report the outcome."""
    if not sms_enabled():
        return "disabled", False, 0
    try:
        remaining = sms_cooldown_remaining(top["label"])
    except Exception as exc:
        print(f"SMS cooldown check failed, sending anyway: {exc}")
        remaining = 0
    if remaining > 0:
        return "cooldown", False, remaining
    try:
        sent = bool(send_sms_alert(top["label"], top["confidence"]))
    except Exception as exc:
        print(f"SMS alert failed: {exc}")
        sent = False
    return ("sent" if sent else "failed"), sent, 0


def alert_view(row, role):
    viewer = "watchman" if role in ("watchman", "admin") else "resident"
    created = row.get("created_at") or ""
    try:
        shown_time = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        shown_time = created
    return {
        "time": shown_time,
        "species": row.get("species"),
        "confidence": row.get("confidence"),
        "distance_m": row.get("distance_m"),
        "movement": row.get("movement"),
        "threat_tier": row.get("threat_tier"),
        "tier": viewer,
        "message": row.get(f"message_{viewer}") or "",
        "sms_sent": row.get("sms_sent", False),
    }


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
    prepared = np.ascontiguousarray(prepared.transpose((2, 0, 1))[::-1])
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
            species = normalize_species(names[int(detector_class)])
            confidence = detector_confidence
            evidence, classifier_label, classifier_confidence = crop, None, None
            if species in BIG_CATS:
                try:
                    evidence, classifier_label, classifier_confidence = make_gradcam_overlay(crop)
                except (OSError, ValueError, RuntimeError):
                    try:
                        classifier_label, classifier_confidence = predict_species(crop)
                    except (OSError, ValueError, RuntimeError):
                        pass
            evidence_name = f"gradcam-{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(MEDIA_DIR / evidence_name), evidence)
            movement, distance = estimate_movement_and_distance(box, original_shape)
            detections.append({
                "label": species,
                "detector_label": str(names[int(detector_class)]),
                "confidence": round(float(confidence), 4),
                "detector_confidence": round(float(detector_confidence), 4),
                "classifier_label": classifier_label,
                "classifier_confidence": None if classifier_confidence is None else round(float(classifier_confidence), 4),
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
@require_role("watchman", "admin")
def api_detect():
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "No image uploaded"}), 400
    role = g.principal["role"]
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
    request.files["image"].save(upload_path)
    try:
        detections, annotated_url = detect_image(upload_path)
    except (OSError, ValueError, RuntimeError) as exc:
        return jsonify({"error": f"Detection failed: {exc}"}), 500
    finally:
        upload_path.unlink(missing_ok=True)

    alert = None
    alert_candidates = [item for item in detections if should_alert(
        item["label"], item["confidence"], ALERT_HIGH_CONF, ALERT_MEDIUM_CONF)]
    if alert_candidates:
        top = max(alert_candidates, key=lambda item: item["confidence"])
        alert = build_alert(top, role)
        with _alert_lock:
            sms_status, sms_sent, retry_in = maybe_send_sms(top)
            alert["sms_sent"] = sms_sent
            alert["sms_status"] = sms_status
            if sms_status == "cooldown":
                alert["sms_retry_in_seconds"] = retry_in
            try:
                supa.insert_alert({
                    "species": top["label"],
                    "confidence": top["confidence"],
                    "distance_m": top["distance_m"],
                    "movement": top["movement"],
                    "threat_tier": top["threat_tier"],
                    "message_watchman": build_alert(top, "watchman")["message"],
                    "message_resident": build_alert(top, "resident")["message"],
                    "sms_sent": sms_sent,
                    "triggered_by": g.principal["id"],
                })
                alert["logged"] = True
            except Exception as exc:
                print(f"Could not store alert in Supabase: {exc}")
                alert["logged"] = False
    return jsonify({"detections": detections, "alert": alert, "annotated_url": annotated_url})


@app.get("/api/config")
def api_config():
    try:
        return jsonify(supa.public_config())
    except supa.SupabaseConfigError as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/api/me")
@require_role()
def api_me():
    return jsonify({"username": g.principal["username"], "role": g.principal["role"], "email": g.principal["email"]})


@app.get("/api/alerts")
@require_role()
def api_alerts():
    try:
        rows = supa.list_alerts()
    except Exception as exc:
        print(f"Could not load alerts from Supabase: {exc}")
        return jsonify({"error": "Could not load alerts"}), 503
    role = g.principal["role"]
    return jsonify({"alerts": [alert_view(row, role) for row in rows]})


@app.get("/media/<path:filename>")
@require_role("watchman", "admin")
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
