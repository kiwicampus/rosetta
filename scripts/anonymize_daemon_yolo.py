#!/opt/conda/envs/anon_env_yolo/bin/python
# -*- coding: utf-8 -*-
"""
Fast anonymization daemon using YOLOv8 + OpenCV YuNet + PyTorch (GPU).

Face detection:  OpenCV YuNet (primary)  +  YOLOv8n face (secondary ensemble)
Plate detection: YOLOv8n  with aspect-ratio / edge-proximity sanity filter
Blur:            OpenCV GaussianBlur (sigma=12)

Quality fixes vs naive YOLO approach:
  - BGR conversion before YOLO (ultralytics expects OpenCV BGR, not RGB)
  - YuNet face detector: much better recall than arnabdhar YOLOv8n at distance
  - Ensemble: union of YuNet + YOLO face boxes keeps both models' strengths
  - Plate filter: width/height >= 1.5, min 20px wide, center not within 5% of edge
  - 15% box padding to cover faces/plates that are slightly off
  - Temporal gap-fill: frames with 0 detections inherit boxes from ±3 neighbours

Protocol (binary, big-endian uint32 header):
  Request  → stdin:  [4B N][4B H][4B W][4B C][N×H×W×C uint8 RGB bytes]
  Response ← stdout: [N×H×W×C uint8 RGB bytes]
  Shutdown → stdin:  [0 0 0 0] (poison pill)
"""

import collections
import math
import os
import struct
import sys
import traceback

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CUDA LD_LIBRARY_PATH fix for GeForce / consumer GPUs
# ---------------------------------------------------------------------------
_LD = os.environ.get("LD_LIBRARY_PATH", "")
_real_driver = "/usr/lib/x86_64-linux-gnu"
if _real_driver not in _LD:
    os.environ["LD_LIBRARY_PATH"] = f"{_real_driver}:{_LD}" if _LD else _real_driver

# ---------------------------------------------------------------------------
# Configurable paths / thresholds
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "weights")
WEIGHTS_DIR = os.environ.get("ANON_WEIGHTS_DIR", _DEFAULT_WEIGHTS_DIR)

YOLO_FACE_WEIGHTS  = os.path.join(WEIGHTS_DIR, "yolov8n_face.pt")
PLATE_WEIGHTS_V5   = os.path.join(WEIGHTS_DIR, "yolov5m_plate.pt")   # primary (better recall)
PLATE_WEIGHTS      = os.path.join(WEIGHTS_DIR, "yolov8n_plate.pt")   # fallback
YUNET_WEIGHTS      = os.path.join(WEIGHTS_DIR, "face_detection_yunet_2023mar.onnx")

YOLO_FACE_THRESHOLD = float(os.environ.get("ANON_FACE_THRESHOLD", "0.20"))
PLATE_THRESHOLD     = float(os.environ.get("ANON_PLATE_THRESHOLD", "0.15"))   # yolov5m threshold
YUNET_THRESHOLD     = float(os.environ.get("ANON_YUNET_THRESHOLD", "0.70"))

# Plate sanity filter: must be wider than tall, not near edge, not too large
PLATE_MIN_RATIO   = float(os.environ.get("ANON_PLATE_MIN_RATIO",  "1.5"))
PLATE_MIN_WIDTH   = int(os.environ.get("ANON_PLATE_MIN_WIDTH",    "20"))
PLATE_MAX_WIDTH_F = float(os.environ.get("ANON_PLATE_MAX_WIDTH_F","0.35"))  # max 35% of frame width
PLATE_EDGE_MARGIN = float(os.environ.get("ANON_PLATE_EDGE_MARGIN","0.08"))  # 8% of frame dim

# Face sanity filter: cap at 30% of frame width/height to kill scene-wide false detections
FACE_MAX_W_F = float(os.environ.get("ANON_FACE_MAX_W_F", "0.30"))
FACE_MAX_H_F = float(os.environ.get("ANON_FACE_MAX_H_F", "0.30"))

BATCH_SIZE      = int(os.environ.get("ANON_BATCH_SIZE",     "32"))
BOX_PADDING     = float(os.environ.get("ANON_BOX_PADDING",  "0.15"))
TEMPORAL_WINDOW = int(os.environ.get("ANON_TEMPORAL_WINDOW","2"))

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models():
    import torch
    from ultralytics import YOLO  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[anonymize_daemon_yolo] Device: {device}", file=sys.stderr, flush=True)

    for path, label in [
        (YOLO_FACE_WEIGHTS, "yolo_face"),
        (YUNET_WEIGHTS,     "yunet_face"),
    ]:
        if not os.path.exists(path):
            print(f"[anonymize_daemon_yolo] ERROR: {label} weights not found: {path}",
                  file=sys.stderr, flush=True)
            sys.exit(1)

    print(f"[anonymize_daemon_yolo] Loading YOLO face model …", file=sys.stderr, flush=True)
    yolo_face = YOLO(YOLO_FACE_WEIGHTS)
    yolo_face.to(device)

    # Prefer yolov5m plate model (better recall); fall back to yolov8n
    print(f"[anonymize_daemon_yolo] Loading plate model …", file=sys.stderr, flush=True)
    use_v5_plate = os.path.exists(PLATE_WEIGHTS_V5)
    if use_v5_plate:
        import torch as _torch
        plate_model = _torch.hub.load(
            "ultralytics/yolov5", "custom",
            path=PLATE_WEIGHTS_V5,
            trust_repo=True, verbose=False,
        )
        plate_model.conf = PLATE_THRESHOLD
        plate_model.to(device)
        if device == "cuda":
            plate_model.half()
        print(f"[anonymize_daemon_yolo] Plate model: yolov5m (better recall, {'fp16' if device == 'cuda' else 'fp32'})", file=sys.stderr, flush=True)
    else:
        plate_model = YOLO(PLATE_WEIGHTS)
        plate_model.to(device)
        print(f"[anonymize_daemon_yolo] Plate model: yolov8n (fallback)", file=sys.stderr, flush=True)

    # YuNet is loaded per-batch (needs frame size) — just validate here
    cv2.FaceDetectorYN.create(
        YUNET_WEIGHTS, "", (64, 64),
        score_threshold=YUNET_THRESHOLD, nms_threshold=0.3,
    )
    print(f"[anonymize_daemon_yolo] YuNet OK. Warming up YOLO …", file=sys.stderr, flush=True)

    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    yolo_face.predict(dummy, verbose=False, device=device, half=(device == "cuda"))
    if use_v5_plate:
        plate_model([dummy])
    else:
        plate_model.predict(dummy, verbose=False, device=device)

    print(
        f"[anonymize_daemon_yolo] Ready  "
        f"(yunet_thr={YUNET_THRESHOLD} yolo_face_thr={YOLO_FACE_THRESHOLD} "
        f"plate_thr={PLATE_THRESHOLD} pad={BOX_PADDING} temporal±{TEMPORAL_WINDOW} "
        f"plate_model={'yolov5m' if use_v5_plate else 'yolov8n'}).",
        file=sys.stderr, flush=True,
    )
    return yolo_face, plate_model, device, use_v5_plate


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def _expand_box(x0, y0, x1, y1, H, W):
    dx = (x1 - x0) * BOX_PADDING
    dy = (y1 - y0) * BOX_PADDING
    return (
        max(0.0, x0 - dx),
        max(0.0, y0 - dy),
        min(float(W), x1 + dx),
        min(float(H), y1 + dy),
    )


def _is_valid_face(x0, y0, x1, y1, H, W):
    """Reject detections that are too large to be a real face in frame."""
    w = x1 - x0
    h = y1 - y0
    if w > W * FACE_MAX_W_F:
        return False
    if h > H * FACE_MAX_H_F:
        return False
    return True


def _is_valid_plate(x0, y0, x1, y1, H, W):
    """Return True if the box looks geometrically like a license plate."""
    w = x1 - x0
    h = y1 - y0
    if w < PLATE_MIN_WIDTH or h <= 0:
        return False
    if w / h < PLATE_MIN_RATIO:
        return False
    if w > W * PLATE_MAX_WIDTH_F:
        return False
    # Reject boxes whose centre is within PLATE_EDGE_MARGIN of any frame edge
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    margin_x = W * PLATE_EDGE_MARGIN
    margin_y = H * PLATE_EDGE_MARGIN
    if cx < margin_x or cx > W - margin_x:
        return False
    if cy < margin_y or cy > H - margin_y:
        return False
    return True


def _smooth_boxes_temporal(all_boxes: list) -> list:
    """Fill zero-detection frames by borrowing boxes from nearby frames."""
    N = len(all_boxes)
    result = [list(b) for b in all_boxes]
    for i in range(N):
        if result[i]:
            continue
        lo = max(0, i - TEMPORAL_WINDOW)
        hi = min(N, i + TEMPORAL_WINDOW + 1)
        for j in range(lo, hi):
            if j != i:
                result[i].extend(all_boxes[j])
    return result


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

def _gaussian_blur_regions(frame: np.ndarray, boxes: list, sigma: float = 12.0) -> np.ndarray:
    if not boxes:
        return frame.copy()

    from scipy.ndimage import gaussian_filter  # type: ignore

    result = frame.copy()
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype=bool)

    for x0, y0, x1, y1 in boxes:
        x0c = max(0, int(math.floor(x0)))
        y0c = max(0, int(math.floor(y0)))
        x1c = min(W, int(math.ceil(x1)))
        y1c = min(H, int(math.ceil(y1)))
        if x1c > x0c and y1c > y0c:
            mask[y0c:y1c, x0c:x1c] = True

    if not mask.any():
        return result

    ys, xs = np.where(mask)
    y_lo = max(0, int(ys.min()) - 20)
    y_hi = min(H, int(ys.max()) + 20)
    x_lo = max(0, int(xs.min()) - 20)
    x_hi = min(W, int(xs.max()) + 20)

    patch = frame[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
    blurred = gaussian_filter(patch, sigma=[sigma, sigma, 0])
    local_mask = mask[y_lo:y_hi, x_lo:x_hi]
    result_patch = result[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
    result_patch[local_mask] = blurred[local_mask]
    result[y_lo:y_hi, x_lo:x_hi] = np.clip(result_patch, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Batched anonymization
# ---------------------------------------------------------------------------

_prev_face_tail: collections.deque = collections.deque(maxlen=TEMPORAL_WINDOW)
_prev_plate_tail: collections.deque = collections.deque(maxlen=TEMPORAL_WINDOW)
_yunet_detector = None
_yunet_size = (0, 0)


def _get_yunet(W, H):
    """Return a YuNet detector, reinitialising only when the frame size changes."""
    global _yunet_detector, _yunet_size
    if _yunet_detector is None or _yunet_size != (W, H):
        _yunet_detector = cv2.FaceDetectorYN.create(
            YUNET_WEIGHTS, "", (W, H),
            score_threshold=YUNET_THRESHOLD,
            nms_threshold=0.3,
            top_k=5000,
        )
        _yunet_size = (W, H)
    return _yunet_detector


def _anonymize_batch(yolo_face, plate_model, device, use_v5_plate: bool, batch: np.ndarray) -> np.ndarray:
    global _prev_face_tail, _prev_plate_tail

    N = batch.shape[0]
    H, W = batch.shape[1], batch.shape[2]
    yunet = _get_yunet(W, H)

    all_face_boxes  = [[] for _ in range(N)]
    all_plate_boxes = [[] for _ in range(N)]

    # ---- Batch YOLO inference (needs BGR input) ----
    # Frames arrive as RGB from the IPC protocol; convert to BGR for OpenCV/YOLO.
    frames_bgr = [batch[i, :, :, ::-1] for i in range(N)]

    for sub_start in range(0, N, BATCH_SIZE):
        sub_end = min(sub_start + BATCH_SIZE, N)
        sub_bgr = frames_bgr[sub_start:sub_end]

        face_results = yolo_face.predict(sub_bgr, verbose=False, device=device, conf=YOLO_FACE_THRESHOLD, half=(device == "cuda"))

        # yolov5 hub API vs yolov8 ultralytics API
        if use_v5_plate:
            plate_results_v5 = plate_model(sub_bgr)  # returns Results with .xyxy list
            plate_boxes_per_img = [plate_results_v5.xyxy[j].cpu().numpy() for j in range(len(sub_bgr))]
        else:
            plate_results_v8 = plate_model.predict(sub_bgr, verbose=False, device=device, conf=PLATE_THRESHOLD, half=(device == "cuda"))
            plate_boxes_per_img = [
                (pr.boxes.xyxy.cpu().numpy() if pr.boxes is not None and len(pr.boxes) > 0 else np.zeros((0, 4)))
                for pr in plate_results_v8
            ]

        for j, fr in enumerate(face_results):
            idx = sub_start + j
            if fr.boxes is not None and len(fr.boxes) > 0:
                for box in fr.boxes.xyxy.cpu().numpy():
                    x0, y0, x1, y1 = box
                    if _is_valid_face(x0, y0, x1, y1, H, W):
                        all_face_boxes[idx].append(_expand_box(x0, y0, x1, y1, H, W))

        for j, pboxes in enumerate(plate_boxes_per_img):
            idx = sub_start + j
            for box in pboxes:
                x0, y0, x1, y1 = box[:4]
                if _is_valid_plate(x0, y0, x1, y1, H, W):
                    all_plate_boxes[idx].append(_expand_box(x0, y0, x1, y1, H, W))

    # ---- YuNet face detection (needs BGR, per-frame) ----
    for i in range(N):
        _, faces = yunet.detect(frames_bgr[i])
        if faces is not None:
            for f in faces:
                x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                if _is_valid_face(x, y, x + w, y + h, H, W):
                    all_face_boxes[i].append(_expand_box(x, y, x + w, y + h, H, W))

    # ---- Temporal gap-fill ----
    face_ext  = list(_prev_face_tail)  + all_face_boxes
    plate_ext = list(_prev_plate_tail) + all_plate_boxes
    tail_len = len(_prev_face_tail)

    smoothed_face  = _smooth_boxes_temporal(face_ext)[tail_len:]
    smoothed_plate = _smooth_boxes_temporal(plate_ext)[tail_len:]

    for boxes in all_face_boxes[-TEMPORAL_WINDOW:]:
        _prev_face_tail.append(boxes)
    for boxes in all_plate_boxes[-TEMPORAL_WINDOW:]:
        _prev_plate_tail.append(boxes)

    # ---- Apply blur ----
    results = batch.copy()
    for i in range(N):
        boxes = smoothed_face[i] + smoothed_plate[i]
        if boxes:
            results[i] = _gaussian_blur_regions(batch[i], boxes)

    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_STDIN  = sys.stdin.buffer
_STDOUT = sys.stdout.buffer


def _read_exactly(n: int) -> bytes:
    buf  = bytearray(n)
    view = memoryview(buf)
    pos  = 0
    while pos < n:
        chunk = _STDIN.read(n - pos)
        if not chunk:
            raise EOFError("stdin closed unexpectedly")
        view[pos : pos + len(chunk)] = chunk
        pos += len(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

    yolo_face, plate_model, device, use_v5_plate = _load_models()
    print("[anonymize_daemon_yolo] Ready.", file=sys.stderr, flush=True)

    while True:
        try:
            header = _read_exactly(16)
        except EOFError:
            break

        N, H, W, C = struct.unpack(">4I", header)

        if N == 0:
            print("[anonymize_daemon_yolo] Shutdown signal.", file=sys.stderr, flush=True)
            break

        try:
            raw = _read_exactly(N * H * W * C)
        except EOFError:
            break

        batch = np.frombuffer(raw, dtype=np.uint8).reshape(N, H, W, C)

        try:
            result = _anonymize_batch(yolo_face, plate_model, device, use_v5_plate, batch)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            result = batch

        _STDOUT.write(result.tobytes())
        _STDOUT.flush()


if __name__ == "__main__":
    main()
