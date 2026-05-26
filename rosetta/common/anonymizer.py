# -*- coding: utf-8 -*-
"""
FrameAnonymizer — pool-based wrapper for the offline-anonymization daemon.

Usage
-----
The anonymizer is designed to be initialised **once** at the start of
``bag_to_lerobot.py`` and reused across all episodes::

    anonymizer = FrameAnonymizer.get_instance(gpu=True)

    # Per episode: collect image frames, then anonymize in one batch
    anonymizer.anonymize_episode(frame_dicts, image_keys)  # in-place

    # At end of script
    anonymizer.shutdown()

Architecture
------------
FrameAnonymizer spawns one ``anonymize_daemon.py`` subprocess **per camera
stream** and runs all camera streams in parallel using ``ThreadPoolExecutor``.
Each daemon loads the face+plate models once at start-up.

Workers are created lazily on the first ``anonymize_episode`` call that
needs N cameras — a single worker is created at construction time and
additional workers are spun up as needed.

Binary protocol (big-endian uint32):
  Request  → stdin:  [4B N][4B H][4B W][4B C][N×H×W×C uint8 RGB]
  Response ← stdout: [N×H×W×C uint8 RGB]
  Shutdown → stdin:  [0 0 0 0] (16-byte zero header, poison pill)
"""

import os
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Resolved paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()

def _find_workspace_src() -> Path:
    for parent in _THIS_FILE.parents:
        candidate = parent / "ros2_workspace" / "src"
        if candidate.is_dir():
            return candidate
    return _THIS_FILE.parent.parent.parent.parent

_WORKSPACE_SRC = _find_workspace_src()
_ROSETTA_ROOT = _WORKSPACE_SRC / "rosetta"

_GPU_ENV_PYTHON = Path("/home/user_lerobot/anon_env_gpu/bin/python")
_CPU_ENV_PYTHON = Path("/opt/conda/envs/anon_env/bin/python")
CONDA_PYTHON = _GPU_ENV_PYTHON if _GPU_ENV_PYTHON.exists() else _CPU_ENV_PYTHON

# YOLO daemon uses the anon_env_yolo conda environment (PyTorch + ultralytics)
_YOLO_ENV_PYTHON = Path("/opt/conda/envs/anon_env_yolo/bin/python")
YOLO_DAEMON_SCRIPT = _ROSETTA_ROOT / "scripts" / "anonymize_daemon_yolo.py"
WEIGHTS_DIR = _ROSETTA_ROOT / "weights"

def _resolve_daemon() -> tuple:
    """Return (python_path, daemon_script) for the YOLO anonymization daemon.

    Raises RuntimeError if the daemon environment or weights are not available.
    Run bag_to_lerobot.py with --anonymize to trigger automatic weight download.
    """
    yolo_face = WEIGHTS_DIR / "yolov8n_face.pt"
    yolo_plate_v5 = WEIGHTS_DIR / "yolov5m_plate.pt"
    yolo_plate_v8 = WEIGHTS_DIR / "yolov8n_plate.pt"

    missing = []
    if not _YOLO_ENV_PYTHON.exists():
        missing.append(f"anon_env_yolo not found at {_YOLO_ENV_PYTHON}")
    if not YOLO_DAEMON_SCRIPT.exists():
        missing.append(f"daemon script not found at {YOLO_DAEMON_SCRIPT}")
    if not yolo_face.exists():
        missing.append(f"yolov8n_face.pt not found in {WEIGHTS_DIR}")
    if not (yolo_plate_v5.exists() or yolo_plate_v8.exists()):
        missing.append(f"no plate model found in {WEIGHTS_DIR}")
    if missing:
        raise RuntimeError(
            "Anonymization daemon unavailable:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nRun bag_to_lerobot.py with --anonymize to download missing weights."
        )

    return _YOLO_ENV_PYTHON, YOLO_DAEMON_SCRIPT


# Chunk size per daemon IPC call.  For the YOLO daemon this is a GPU batch size.
# For the TF1.15 daemon it only affects IPC framing.
DEFAULT_CHUNK_SIZE = 128

# Number of parallel daemon workers.  YOLO daemon is fast enough that 1 worker
# handles all cameras sequentially without throughput loss, and avoids VRAM
# exhaustion on 6 GB GPUs.  Increase via ANON_NUM_WORKERS if VRAM allows.
DEFAULT_NUM_WORKERS = int(os.environ.get("ANON_NUM_WORKERS", "1"))


# ---------------------------------------------------------------------------
# Internal per-daemon worker
# ---------------------------------------------------------------------------

class _DaemonWorker:
    """Manages a single anonymization daemon subprocess."""

    def __init__(self, gpu: bool = True, chunk_size: int = DEFAULT_CHUNK_SIZE, worker_id: int = 0):
        self._chunk_size = chunk_size
        self._worker_id = worker_id

        conda_python, daemon_script = _resolve_daemon()

        env = os.environ.copy()
        env["ANON_WEIGHTS_DIR"] = str(WEIGHTS_DIR)
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        # Prepend real driver lib so CUDA forward-compat libs don't shadow it
        real_driver_lib = "/usr/lib/x86_64-linux-gnu"
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{real_driver_lib}:{existing_ld}" if existing_ld else real_driver_lib
        )
        if not gpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
        env["MPLCONFIGDIR"] = "/tmp"
        # Point torch.hub to the pre-cached source baked into the image at build time.
        # Avoids a GitHub download on every daemon startup and works in offline environments.
        # Use the pre-cached hub source baked into the image; fall back to the
        # default (~/.cache/torch) when running in older containers without it.
        if os.path.isdir("/opt/torch_hub"):
            env["TORCH_HOME"] = "/opt/torch_hub"

        self._proc = subprocess.Popen(
            [str(conda_python), str(daemon_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=env,
        )

    def anonymize_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Anonymize frames in chunks and return results in order."""
        if not frames:
            return frames

        result: List[np.ndarray] = []
        n_chunks = max(1, (len(frames) + self._chunk_size - 1) // self._chunk_size)
        for chunk_idx, chunk_start in enumerate(range(0, len(frames), self._chunk_size)):
            chunk = frames[chunk_start : chunk_start + self._chunk_size]
            t0 = time.perf_counter()
            anon_chunk = self._send_chunk(chunk)
            elapsed = time.perf_counter() - t0
            result.extend(anon_chunk)
            print(
                f"    [Anon w{self._worker_id}] batch {chunk_idx + 1}/{n_chunks}: "
                f"{len(chunk)} frames in {elapsed:.2f}s "
                f"({len(chunk) / elapsed:.1f} fps)",
                flush=True,
            )
        return result

    def shutdown(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(struct.pack(">4I", 0, 0, 0, 0))
                self._proc.stdin.flush()
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
        except Exception:
            pass

    def _send_chunk(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        arr = np.stack(frames, axis=0)
        N, H, W, C = arr.shape
        header = struct.pack(">4I", N, H, W, C)
        self._proc.stdin.write(header + arr.tobytes())
        self._proc.stdin.flush()
        n_bytes = N * H * W * C
        raw = self._read_exactly(n_bytes)
        out = np.frombuffer(raw, dtype=np.uint8).reshape(N, H, W, C)
        return [out[i] for i in range(N)]

    def _read_exactly(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        pos = 0
        while pos < n:
            chunk = self._proc.stdout.read(n - pos)
            if not chunk:
                raise RuntimeError(
                    f"[FrameAnonymizer worker {self._worker_id}] Daemon closed stdout "
                    "unexpectedly. Check stderr for TF/model errors."
                )
            view[pos : pos + len(chunk)] = chunk
            pos += len(chunk)
        return bytes(buf)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FrameAnonymizer:
    """Pool-based anonymizer that processes camera streams in parallel.

    One daemon worker per camera stream is spawned lazily on the first
    ``anonymize_episode`` call.  The pool persists and is reused across
    all episodes.

    Parameters
    ----------
    gpu:
        Use GPU-enabled conda Python for each daemon.
    chunk_size:
        Max frames per IPC round-trip per worker.
    num_workers:
        Maximum parallel daemon processes.  Defaults to ``DEFAULT_NUM_WORKERS``
        (4).  Set ``ANON_NUM_WORKERS=1`` env var to revert to single-worker
        (non-parallel) mode.
    """

    _instance: Optional["FrameAnonymizer"] = None

    def __init__(
        self,
        gpu: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        _, daemon_script = _resolve_daemon()
        print(
            f"[FrameAnonymizer] Using YOLO daemon ({daemon_script.name}) — "
            "fast GPU inference (~10 ms/frame).",
            flush=True,
        )

        self._gpu = gpu
        self._chunk_size = chunk_size
        self._max_workers = max(1, num_workers)
        self._workers: List[_DaemonWorker] = []

        # Spin up the first worker immediately so startup errors surface early.
        print(
            f"[FrameAnonymizer] Starting primary daemon "
            f"({'GPU' if gpu else 'CPU'}) … this may take a few seconds.",
            flush=True,
        )
        self._workers.append(_DaemonWorker(gpu=gpu, chunk_size=chunk_size, worker_id=0))

    @classmethod
    def get_instance(
        cls,
        gpu: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> "FrameAnonymizer":
        """Return (and lazily create) the singleton anonymizer."""
        if cls._instance is None:
            cls._instance = cls(gpu=gpu, chunk_size=chunk_size, num_workers=num_workers)
        return cls._instance

    # ------------------------------------------------------------------

    def _ensure_workers(self, n: int) -> None:
        """Spin up daemon workers until we have at least n."""
        needed = min(n, self._max_workers)
        while len(self._workers) < needed:
            wid = len(self._workers)
            print(
                f"[FrameAnonymizer] Starting additional daemon worker {wid + 1}/{needed} …",
                flush=True,
            )
            self._workers.append(
                _DaemonWorker(gpu=self._gpu, chunk_size=self._chunk_size, worker_id=wid)
            )

    # ------------------------------------------------------------------
    # Public API (backward-compatible)
    # ------------------------------------------------------------------

    def anonymize_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Anonymize frames using the primary (first) worker."""
        return self._workers[0].anonymize_batch(frames)

    def anonymize_episode(
        self, frame_dicts: List[Dict], image_keys: List[str]
    ) -> List[Dict]:
        """Anonymize all camera streams in parallel (one worker per stream).

        Each camera stream is dispatched to its own daemon worker so all
        cameras are anonymized concurrently rather than sequentially.

        Parameters
        ----------
        frame_dicts:
            Frame dicts as assembled by bag_to_lerobot — modified in-place.
        image_keys:
            Feature keys that contain image data (dtype "video" or "image").
        """
        n_cameras = len(image_keys)
        if n_cameras == 0:
            return frame_dicts

        self._ensure_workers(n_cameras)

        episode_t0 = time.perf_counter()
        total_frames_processed = 0

        # Build per-camera frame lists (index into frame_dicts preserved)
        camera_data: Dict[str, tuple] = {}
        for key in image_keys:
            imgs: List[np.ndarray] = []
            valid_indices: List[int] = []
            for idx, fd in enumerate(frame_dicts):
                arr = fd.get(key)
                if arr is not None and isinstance(arr, np.ndarray) and arr.ndim == 3:
                    imgs.append(arr)
                    valid_indices.append(idx)
            if imgs:
                print(f"  [Anon] stream '{key}': {len(imgs)} frames", flush=True)
                camera_data[key] = (imgs, valid_indices)

        if not camera_data:
            return frame_dicts

        # Dispatch one worker per camera stream, run in parallel
        actual_workers = min(len(camera_data), self._max_workers)

        def _process_camera(worker: _DaemonWorker, key: str, imgs: List[np.ndarray], valid_indices: List[int]):
            key_t0 = time.perf_counter()
            anon_imgs = worker.anonymize_batch(imgs)
            key_elapsed = time.perf_counter() - key_t0
            print(
                f"  [Anon] stream '{key}' done in {key_elapsed:.2f}s "
                f"({len(imgs) / key_elapsed:.1f} fps)",
                flush=True,
            )
            return key, anon_imgs, valid_indices

        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = []
            for i, (key, (imgs, valid_indices)) in enumerate(camera_data.items()):
                worker = self._workers[i % len(self._workers)]
                futures.append(
                    pool.submit(_process_camera, worker, key, imgs, valid_indices)
                )

            for future in as_completed(futures):
                key, anon_imgs, valid_indices = future.result()
                total_frames_processed += len(anon_imgs)
                for frame_idx, anon in zip(valid_indices, anon_imgs):
                    frame_dicts[frame_idx][key] = anon

        episode_elapsed = time.perf_counter() - episode_t0
        print(
            f"  [Anon] episode total: {total_frames_processed} frame-streams "
            f"across {n_cameras} camera(s) in {episode_elapsed:.2f}s "
            f"[{actual_workers} parallel workers]",
            flush=True,
        )
        return frame_dicts

    def shutdown(self) -> None:
        """Shut down all daemon workers cleanly."""
        for worker in self._workers:
            try:
                worker.shutdown()
            except Exception:
                pass
        FrameAnonymizer._instance = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
