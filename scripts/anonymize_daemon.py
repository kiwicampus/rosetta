#!/opt/conda/envs/anon_env/bin/python
# -*- coding: utf-8 -*-
"""
Persistent anonymization daemon.

This script runs inside the ``anon_env`` conda environment (Python 3.7,
TF1.15, CUDA 10.0) and is spawned **once** per bag_to_lerobot.py run by
:class:`rosetta.common.anonymizer.FrameAnonymizer`.

Protocol (binary, big-endian uint32 header):
  Request  → stdin:  [4B N][4B H][4B W][4B C][N×H×W×C uint8 RGB bytes]
  Response ← stdout: [N×H×W×C uint8 RGB bytes]

A poison-pill request with N=0 causes a clean exit.

Optimisation notes
------------------
Face and plate detectors run on separate TF sessions.  Both ``session.run()``
calls release the Python GIL, so a shared ThreadPoolExecutor(max_workers=2)
allows true concurrent GPU execution when the GPU scheduler can service both
simultaneously.  This halves the per-frame detection time on well-provisioned
hardware.
"""

import math
import os
import struct
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ---------------------------------------------------------------------------
# Locate the offline-anonymization repo
# ---------------------------------------------------------------------------

ANON_REPO_PATH = os.environ.get("ANON_REPO_PATH", "")
if not ANON_REPO_PATH:
    print(
        "[anonymize_daemon] ERROR: ANON_REPO_PATH env var not set.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

_UAI_ROOT = os.path.join(
    ANON_REPO_PATH, "anonymization", "tf_anonymizer", "src", "anonymizer"
)
sys.path.insert(0, _UAI_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Configurable paths / thresholds
# ---------------------------------------------------------------------------

WEIGHTS_DIR = os.environ.get(
    "ANON_WEIGHTS_DIR", os.path.join(ANON_REPO_PATH, "weights")
)
FACE_THRESHOLD = float(os.environ.get("ANON_FACE_THRESHOLD", "0.3"))
PLATE_THRESHOLD = float(os.environ.get("ANON_PLATE_THRESHOLD", "0.001"))

os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_anonymizer():
    from anonymizer.detection import Detector, download_weights, get_weights_path  # type: ignore
    from anonymizer.obfuscation import Obfuscator  # type: ignore
    from anonymizer.anonymization.anonymizer import Anonymizer  # type: ignore

    print(
        f"[anonymize_daemon] Checking / downloading weights to {WEIGHTS_DIR} …",
        file=sys.stderr,
        flush=True,
    )
    download_weights(download_directory=WEIGHTS_DIR)

    detectors = {
        "face": Detector(
            kind="face",
            weights_path=get_weights_path(WEIGHTS_DIR, kind="face"),
        ),
        "plate": Detector(
            kind="plate",
            weights_path=get_weights_path(WEIGHTS_DIR, kind="plate"),
        ),
    }
    obfuscator = Obfuscator(kernel_size=21, sigma=2.0, box_kernel_size=9)
    anon = Anonymizer(detectors=detectors, obfuscator=obfuscator)
    print("[anonymize_daemon] Models loaded. Ready.", file=sys.stderr, flush=True)
    return anon


# ---------------------------------------------------------------------------
# Parallel per-frame processing
# ---------------------------------------------------------------------------

# Shared executor: reuse threads across frames to avoid construction overhead.
# max_workers=2 runs face and plate detectors concurrently.
_DETECTOR_POOL: ThreadPoolExecutor = None  # initialised in main()


def _run_detector(detector, frame, threshold):
    """Run one detector on one frame.  Called from a thread."""
    return detector.detect(frame, threshold)


def _anonymize_frame_parallel(anonymizer, frame: np.ndarray) -> np.ndarray:
    """Detect faces and plates concurrently, then obfuscate.

    Both detectors have separate TF sessions, so session.run() releases the
    GIL and both GPU kernels can be in-flight simultaneously.
    """
    face_det = anonymizer.detectors["face"]
    plate_det = anonymizer.detectors["plate"]

    # Submit both detectors to the shared pool
    face_future = _DETECTOR_POOL.submit(_run_detector, face_det, frame, FACE_THRESHOLD)
    plate_future = _DETECTOR_POOL.submit(_run_detector, plate_det, frame, PLATE_THRESHOLD)
    face_boxes = face_future.result()
    plate_boxes = plate_future.result()

    all_boxes = face_boxes + plate_boxes

    if not all_boxes:
        return frame.copy()

    result = anonymizer.obfuscator.obfuscate(frame, all_boxes)
    if not isinstance(result, np.ndarray):
        result = np.array(result, dtype=np.uint8)
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    if result.shape != frame.shape:
        print(
            f"[anonymize_daemon] WARN: shape mismatch {result.shape} vs {frame.shape}",
            file=sys.stderr, flush=True,
        )
        return frame.copy()
    return result


# ---------------------------------------------------------------------------
# Fallback: single-threaded per-frame (kept for error recovery)
# ---------------------------------------------------------------------------

def _anonymize_frame_single(anonymizer, frame: np.ndarray) -> np.ndarray:
    result, _ = anonymizer.anonymize_image(
        frame,
        detection_thresholds={"face": FACE_THRESHOLD, "plate": PLATE_THRESHOLD},
    )
    if not isinstance(result, np.ndarray):
        result = np.array(result, dtype=np.uint8)
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    if result.shape != frame.shape:
        return frame
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_STDIN = sys.stdin.buffer
_STDOUT = sys.stdout.buffer


def _read_exactly(n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
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
    global _DETECTOR_POOL
    anonymizer = _load_anonymizer()

    # One persistent pool for face+plate concurrent detection
    _DETECTOR_POOL = ThreadPoolExecutor(max_workers=2)

    try:
        while True:
            try:
                header = _read_exactly(16)
            except EOFError:
                break

            N, H, W, C = struct.unpack(">4I", header)

            if N == 0:
                print("[anonymize_daemon] Received shutdown signal.", file=sys.stderr, flush=True)
                break

            try:
                raw = _read_exactly(N * H * W * C)
            except EOFError:
                break

            batch = np.frombuffer(raw, dtype=np.uint8).reshape(N, H, W, C)
            out_frames = []

            for i in range(N):
                try:
                    anon_frame = _anonymize_frame_parallel(anonymizer, batch[i])
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    # On per-frame error: try single-threaded fallback
                    try:
                        anon_frame = _anonymize_frame_single(anonymizer, batch[i])
                    except Exception:
                        traceback.print_exc(file=sys.stderr)
                        anon_frame = batch[i]
                out_frames.append(anon_frame)

            result = np.stack(out_frames, axis=0)
            _STDOUT.write(result.tobytes())
            _STDOUT.flush()

    finally:
        _DETECTOR_POOL.shutdown(wait=False)


if __name__ == "__main__":
    main()
