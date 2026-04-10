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

API assumption
--------------
The offline-anonymization repo exposes an ``Anonymizer`` class whose
constructor accepts ``face_weights_path``, ``plate_weights_path``,
``face_threshold``, and ``plate_threshold``, and provides a method::

    anonymizer.anonymize_image(image: np.ndarray) -> np.ndarray

where ``image`` is HWC uint8 RGB and the return value has the same shape
and dtype.  If the actual API differs, update ``_load_anonymizer()`` and
``_anonymize_frame()`` below.
"""

import os
import struct
import sys
import traceback

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

# The understand-ai anonymizer submodule lives at:
#   <ANON_REPO_PATH>/anonymization/tf_anonymizer/src/anonymizer/
# Add it to sys.path so we can import anonymizer.* directly without
# going through models.py (which needs the python_path package).
_UAI_ROOT = os.path.join(
    ANON_REPO_PATH, "anonymization", "tf_anonymizer", "src", "anonymizer"
)
sys.path.insert(0, _UAI_ROOT)

# Suppress TF1.x deprecation noise on stderr
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Configurable paths / thresholds (can be overridden via env vars)
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
    """Build an understand-ai Anonymizer directly, bypassing models.py.

    Uses the understand-ai submodule at
    ``<ANON_REPO_PATH>/anonymization/tf_anonymizer/src/anonymizer/``.
    Weights are downloaded from Google Drive on first run if not present.
    """
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


def _anonymize_frame(anonymizer, frame: np.ndarray) -> np.ndarray:
    """Run anonymization on a single HWC uint8 RGB frame."""
    result, _ = anonymizer.anonymize_image(
        frame,
        detection_thresholds={"face": FACE_THRESHOLD, "plate": PLATE_THRESHOLD},
    )
    if not isinstance(result, np.ndarray):
        result = np.array(result, dtype=np.uint8)
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    if result.shape != frame.shape:
        print(
            f"[anonymize_daemon] WARN: output shape {result.shape} != "
            f"input shape {frame.shape}; returning original.",
            file=sys.stderr,
            flush=True,
        )
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
    anonymizer = _load_anonymizer()

    while True:
        try:
            header = _read_exactly(16)
        except EOFError:
            break

        N, H, W, C = struct.unpack(">4I", header)

        if N == 0:
            # Poison pill — clean shutdown
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
                anon_frame = _anonymize_frame(anonymizer, batch[i])
            except Exception:
                traceback.print_exc(file=sys.stderr)
                # On error keep the original frame so the run isn't aborted
                anon_frame = batch[i]
            out_frames.append(anon_frame)

        result = np.stack(out_frames, axis=0)  # [N, H, W, C] uint8
        _STDOUT.write(result.tobytes())
        _STDOUT.flush()


if __name__ == "__main__":
    main()
