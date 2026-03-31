# -*- coding: utf-8 -*-
"""
FrameAnonymizer — singleton wrapper for the offline-anonymization daemon.

Usage
-----
The anonymizer is designed to be initialised **once** at the start of
``bag_to_lerobot.py`` and reused across all episodes::

    anonymizer = FrameAnonymizer.get_instance(gpu=True)

    # Per episode: collect image frames, then anonymize in one batch
    anon_frames = anonymizer.anonymize_batch(frames)   # List[np.ndarray HWC uint8]

    # At end of script
    anonymizer.shutdown()

Architecture
------------
The anonymizer spawns ``anonymize_daemon.py`` as a persistent subprocess
running inside the ``anon_env`` conda environment (Python 3.7, TF1.15,
CUDA 10.0).  The daemon loads the face+plate models **once** at start-up
and serves requests over a simple binary stdin/stdout protocol for the
entire lifetime of the parent process.

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
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Resolved paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# Works for both the source tree and the colcon-installed layout:
#   source:    <ws>/ros2_workspace/src/rosetta/rosetta/common/anonymizer.py
#   installed: <ws>/ros2_workspace/install/rosetta/lib/.../site-packages/rosetta/common/anonymizer.py
# Walk up until we find a parent that contains ros2_workspace/src/.
def _find_workspace_src() -> Path:
    for parent in _THIS_FILE.parents:
        candidate = parent / "ros2_workspace" / "src"
        if candidate.is_dir():
            return candidate
    # Fallback to the old relative assumption (source-tree layout).
    return _THIS_FILE.parent.parent.parent.parent

_WORKSPACE_SRC = _find_workspace_src()
_ROSETTA_ROOT = _THIS_FILE.parent.parent.parent          # site-packages/ or src/rosetta/

# Prefer the GPU-capable env (nvidia-tensorflow, CUDA 12, Ampere-compatible).
# Fall back to the original anon_env (TF-GPU 1.15.5 + CUDA 10.0) if not found.
_GPU_ENV_PYTHON = Path("/home/user_lerobot/anon_env_gpu/bin/python")
_CPU_ENV_PYTHON = Path("/opt/conda/envs/anon_env/bin/python")
CONDA_PYTHON = _GPU_ENV_PYTHON if _GPU_ENV_PYTHON.exists() else _CPU_ENV_PYTHON
DAEMON_SCRIPT = _ROSETTA_ROOT / "scripts" / "anonymize_daemon.py"
ANON_REPO = _WORKSPACE_SRC / "offline-anonymization"

# Chunk size for breaking very large batches to manage peak GPU memory.
# At 640×360×3 ≈ 691 KB per frame, 64 frames ≈ 44 MB — comfortably fits
# in GPU VRAM.  Increase if you have ≥8 GB VRAM.
DEFAULT_CHUNK_SIZE = 64


class FrameAnonymizer:
    """Singleton that manages a persistent anonymization daemon subprocess.

    Parameters
    ----------
    gpu:
        ``True`` uses the ``anon_env`` conda Python which has
        ``tensorflow-gpu=1.15.5 + cudatoolkit=10.0``.  ``False`` forces
        CPU-only mode by setting ``CUDA_VISIBLE_DEVICES=-1`` in the
        daemon's environment.
    chunk_size:
        Maximum number of frames sent to the daemon per IPC call.  Larger
        chunks improve GPU utilisation; smaller chunks reduce peak memory.
    """

    _instance: Optional["FrameAnonymizer"] = None

    def __init__(self, gpu: bool = True, chunk_size: int = DEFAULT_CHUNK_SIZE):
        if not CONDA_PYTHON.exists():
            raise RuntimeError(
                f"conda anon_env not found at {CONDA_PYTHON}.\n"
                "Rebuild the Docker image — the Dockerfile creates this env "
                "automatically (see .devcontainer/Dockerfile)."
            )
        if not DAEMON_SCRIPT.exists():
            raise RuntimeError(
                f"anonymize_daemon.py not found at {DAEMON_SCRIPT}.\n"
                "Check that the rosetta package is properly installed."
            )
        if not ANON_REPO.exists():
            raise RuntimeError(
                f"offline-anonymization submodule not found at {ANON_REPO}.\n"
                "Run: git submodule update --init ros2_workspace/src/offline-anonymization"
            )

        self._chunk_size = chunk_size
        env = os.environ.copy()
        env["ANON_REPO_PATH"] = str(ANON_REPO)
        # TF1.15 proto files are incompatible with protobuf >= 4.x
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        # The nvidia/cuda base image puts /usr/local/cuda-X.Y/compat/ first in
        # ld.so.conf.d, which makes the CUDA forward-compat libcuda.so shadow the
        # real driver library.  Forward compat is only supported on Data Center
        # GPUs (A100/V100), NOT on GeForce (RTX 30xx/40xx), causing cuInit to
        # fail with CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE.  Prepend the real
        # driver library path so the daemon sees the correct libcuda.so first.
        real_driver_lib = "/usr/lib/x86_64-linux-gnu"
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{real_driver_lib}:{existing_ld}" if existing_ld else real_driver_lib
        )
        if not gpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"

        print(
            f"[FrameAnonymizer] Starting daemon "
            f"({'GPU' if gpu else 'CPU'}) … this may take a few seconds.",
            flush=True,
        )
        self._proc = subprocess.Popen(
            [str(CONDA_PYTHON), str(DAEMON_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,   # forward daemon logs to our stderr
            env=env,
        )

    # ------------------------------------------------------------------
    # Singleton interface
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, gpu: bool = True, chunk_size: int = DEFAULT_CHUNK_SIZE) -> "FrameAnonymizer":
        """Return (and lazily create) the singleton anonymizer.

        Parameters are only used on first call; subsequent calls return the
        existing instance regardless of the arguments passed.
        """
        if cls._instance is None:
            cls._instance = cls(gpu=gpu, chunk_size=chunk_size)
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymize_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Anonymize a list of HWC uint8 RGB frames.

        Frames are sent to the daemon in chunks of ``chunk_size`` to
        balance GPU throughput and memory usage.

        Parameters
        ----------
        frames:
            List of ``np.ndarray`` with shape ``(H, W, C)`` and dtype
            ``uint8``.  All frames must have the same spatial dimensions.

        Returns
        -------
        List[np.ndarray]
            Anonymized frames in the same order, same shape and dtype.
        """
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
                f"    [Anon] batch {chunk_idx + 1}/{n_chunks}: "
                f"{len(chunk)} frames in {elapsed:.2f}s "
                f"({len(chunk) / elapsed:.1f} fps)",
                flush=True,
            )
        return result

    def anonymize_episode(
        self, frame_dicts: List[Dict], image_keys: List[str]
    ) -> List[Dict]:
        """Anonymize all image fields across an episode's frame list in-place.

        Collects all images for each camera key into one batch, anonymizes
        them in a single round-trip per key, then writes the results back
        into the frame dicts.

        Parameters
        ----------
        frame_dicts:
            List of frame dicts as assembled by the bag_to_lerobot tick loop.
        image_keys:
            Keys in each frame dict that hold ``np.ndarray`` image data
            (dtype == "video" or "image" features).

        Returns
        -------
        The same ``frame_dicts`` list with anonymized images written back.
        """
        episode_t0 = time.perf_counter()
        total_frames_processed = 0

        for key in image_keys:
            imgs: List[np.ndarray] = []
            valid_indices: List[int] = []
            for idx, fd in enumerate(frame_dicts):
                arr = fd.get(key)
                if arr is not None and isinstance(arr, np.ndarray) and arr.ndim == 3:
                    imgs.append(arr)
                    valid_indices.append(idx)

            if not imgs:
                continue

            print(f"  [Anon] stream '{key}': {len(imgs)} frames", flush=True)
            key_t0 = time.perf_counter()
            anon_imgs = self.anonymize_batch(imgs)
            key_elapsed = time.perf_counter() - key_t0
            print(
                f"  [Anon] stream '{key}' done in {key_elapsed:.2f}s "
                f"({len(imgs) / key_elapsed:.1f} fps)",
                flush=True,
            )
            total_frames_processed += len(imgs)

            for frame_idx, anon in zip(valid_indices, anon_imgs):
                frame_dicts[frame_idx][key] = anon

        episode_elapsed = time.perf_counter() - episode_t0
        print(
            f"  [Anon] episode total: {total_frames_processed} frame-streams "
            f"across {len(image_keys)} camera(s) in {episode_elapsed:.2f}s",
            flush=True,
        )
        return frame_dicts

    def shutdown(self) -> None:
        """Send poison pill and wait for the daemon to exit cleanly."""
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(struct.pack(">4I", 0, 0, 0, 0))
                self._proc.stdin.flush()
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
        except Exception:
            pass
        FrameAnonymizer._instance = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_chunk(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Send one chunk of frames over the binary IPC channel."""
        arr = np.stack(frames, axis=0)   # [N, H, W, C] uint8
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
                    "[FrameAnonymizer] Daemon closed stdout unexpectedly. "
                    "Check stderr for TF/model errors."
                )
            view[pos : pos + len(chunk)] = chunk
            pos += len(chunk)
        return bytes(buf)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
