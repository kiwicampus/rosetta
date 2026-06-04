#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone anonymization preview — no GUI, no LeRobot dataset.

Reads a rosbag, decodes the camera image streams listed in a contract,
runs them through the real ``FrameAnonymizer`` (which spawns the YOLO
anonymization daemon), and writes per-camera before/after MP4s plus a few
sample PNGs so you can eyeball that faces and plates are blurred.

This reuses the production decode + anonymization path; it only swaps out
the LeRobot writer for plain video output.

Run INSIDE the dev container (ROS sourced, rosetta on PYTHONPATH):

    python3 anonymize_bag_preview.py \
        --bag /path/to/rosbag_dir \
        --contract ../contracts/4cams.yaml \
        --out /tmp/anon_preview \
        --max-frames 200

Notes
-----
* ``--bag`` is a rosbag2 directory (contains *.mcap / *.db3 + metadata.yaml).
* Camera streams are auto-discovered from the contract (any stream with an
  ``image:`` block). Defaults to contracts/4cams.yaml (the 4-camera robot).
* CPU anonymization: pass ``--no-gpu``.
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

# ROS / rosbag2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# rosetta production code
from rosetta.common import decoders as _decoders  # noqa: F401  (registers decoders)
from rosetta.common.contract_utils import DECODERS, decode_value
from rosetta.common.anonymizer import FrameAnonymizer

try:
    import cv2
except ImportError:  # pragma: no cover
    print("[preview] ERROR: opencv (cv2) not available in this env.", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Contract parsing — find image streams (key, topic, ros type)
# ---------------------------------------------------------------------------

def _load_camera_streams(contract_path: Path):
    """Return list of (key, topic, ros_type) for every image stream."""
    with open(contract_path) as f:
        contract = yaml.safe_load(f)

    # Image streams live under "observations" (rosetta contract). Fall back to
    # other common section names just in case.
    entries = []
    for section in ("observations", "features", "streams"):
        val = contract.get(section)
        if isinstance(val, list):
            entries.extend(val)

    streams = []
    for entry in entries:
        if not isinstance(entry, dict) or "image" not in entry:
            continue
        key = entry.get("key")
        topic = entry.get("topic")
        ros_type = entry.get("type")
        if key and topic and ros_type:
            streams.append((key, topic, ros_type))
    return streams


def _make_spec(topic: str):
    """Minimal stand-in for the contract spec the decoders expect.

    Native resolution (no resize), rgb8 output — ideal for visual review.
    """
    return SimpleNamespace(
        topic=topic,
        image_encoding="rgb8",
        image_resize=None,
        names=None,
    )


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------

def _detect_storage_id(bag_dir: Path) -> str:
    meta = bag_dir / "metadata.yaml"
    if meta.exists():
        try:
            with open(meta) as f:
                m = yaml.safe_load(f)
            sid = (
                m.get("rosbag2_bagfile_information", {}).get("storage_identifier")
            )
            if sid:
                return sid
        except Exception:
            pass
    # Fall back on file extension
    if list(bag_dir.glob("*.mcap")):
        return "mcap"
    if list(bag_dir.glob("*.db3")):
        return "sqlite3"
    return "mcap"


def _decode_camera_streams(bag_dir: Path, streams, max_frames: int, stride: int):
    """Read the bag once, decode each camera topic in order.

    Returns {key: [rgb uint8 HWC arrays]}.
    """
    topics_wanted = {topic: (key, ros_type) for key, topic, ros_type in streams}
    specs = {topic: _make_spec(topic) for topic in topics_wanted}
    counts = {topic: 0 for topic in topics_wanted}
    frames = {key: [] for key, _, _ in streams}

    storage_id = _detect_storage_id(bag_dir)
    print(f"[preview] Opening bag {bag_dir} (storage_id={storage_id})")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    done = {t: False for t in topics_wanted}
    while reader.has_next():
        topic, data, _ts = reader.read_next()
        if topic not in topics_wanted:
            continue
        if done[topic]:
            continue

        key, ros_type = topics_wanted[topic]
        ros_type = type_map.get(topic, ros_type)
        if ros_type not in DECODERS:
            print(f"[preview] WARN: no decoder for {ros_type} ({topic}); skipping")
            done[topic] = True
            continue

        msg = deserialize_message(data, get_message(ros_type))
        try:
            arr = decode_value(ros_type, msg, specs[topic])
        except Exception as e:  # keep going on a bad frame
            print(f"[preview] decode error on {topic}: {e}")
            continue
        if arr is None:
            continue

        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Skip h264 warm-up / dummy frames (uniform image before first keyframe)
        if arr.ndim != 3 or arr.shape[2] != 3 or int(arr.max()) - int(arr.min()) == 0:
            continue

        counts[topic] += 1
        if (counts[topic] - 1) % stride != 0:
            continue

        frames[key].append(arr)
        if len(frames[key]) >= max_frames:
            done[topic] = True
            if all(done.values()):
                break

    for key, _, _ in streams:
        print(f"[preview]   {key}: {len(frames[key])} frames decoded")
    return frames


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_outputs(out_dir: Path, key: str, originals, anonymized, fps: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = key.replace(".", "_").replace("/", "_")
    if not originals:
        return

    h, w = originals[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Side-by-side before|after video
    sbs_path = out_dir / f"{safe}_before_after.mp4"
    writer = cv2.VideoWriter(str(sbs_path), fourcc, fps, (w * 2, h))
    for orig, anon in zip(originals, anonymized):
        a = anon if anon.shape == orig.shape else cv2.resize(anon, (w, h))
        sbs = np.hstack([orig, a])
        writer.write(cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
    writer.release()

    # A few sample PNGs (anonymized)
    n = len(anonymized)
    for i in [0, n // 2, n - 1]:
        if 0 <= i < n:
            png = out_dir / f"{safe}_sample_{i:04d}.png"
            cv2.imwrite(str(png), cv2.cvtColor(anonymized[i], cv2.COLOR_RGB2BGR))

    print(f"[preview]   wrote {sbs_path.name} (+ sample PNGs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Standalone rosbag anonymization preview")
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    _default_contract = (
        Path(__file__).resolve().parent.parent / "contracts" / "4cams.yaml"
    )
    ap.add_argument("--contract", default=str(_default_contract),
                    help="contract yaml (default: contracts/4cams.yaml)")
    ap.add_argument("--out", default="/tmp/anon_preview", help="output directory")
    ap.add_argument("--max-frames", type=int, default=200,
                    help="max frames per camera to process")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth decoded frame")
    ap.add_argument("--fps", type=float, default=10.0, help="output video fps")
    ap.add_argument("--no-gpu", action="store_true", help="force CPU anonymization")
    args = ap.parse_args()

    bag_dir = Path(args.bag).resolve()
    contract_path = Path(args.contract).resolve()
    out_dir = Path(args.out).resolve()

    if not bag_dir.is_dir():
        sys.exit(f"[preview] bag dir not found: {bag_dir}")
    if not contract_path.is_file():
        sys.exit(f"[preview] contract not found: {contract_path}")

    streams = _load_camera_streams(contract_path)
    if not streams:
        sys.exit(f"[preview] no image streams found in {contract_path}")
    print(f"[preview] Camera streams: {[k for k, _, _ in streams]}")

    # Ensure YOLO weights are present (reuse the converter's downloader).
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from bag_to_lerobot import _ensure_anon_weights  # type: ignore
        _ensure_anon_weights()
    except Exception as e:
        print(f"[preview] WARN: could not auto-download weights ({e}). "
              "If the daemon fails, run bag_to_lerobot.py --anonymize once.")

    # 1) Decode
    frames = _decode_camera_streams(bag_dir, streams, args.max_frames, args.stride)
    image_keys = [k for k, _, _ in streams if frames[k]]
    if not image_keys:
        sys.exit("[preview] no frames decoded from any camera")

    # Keep pristine copies for the before/after comparison
    originals = {k: [f.copy() for f in frames[k]] for k in image_keys}

    # 2) Anonymize via the real pipeline path (one frame-dict per tick)
    print("[preview] Starting anonymizer (spawns YOLO daemon) …")
    anonymizer = FrameAnonymizer.get_instance(gpu=not args.no_gpu)

    n_ticks = max(len(frames[k]) for k in image_keys)
    frame_dicts = []
    for i in range(n_ticks):
        d = {}
        for k in image_keys:
            if i < len(frames[k]):
                d[k] = frames[k][i]
        frame_dicts.append(d)

    anonymizer.anonymize_episode(frame_dicts, image_keys)

    # 3) Write before/after per camera
    for k in image_keys:
        anon = [frame_dicts[i][k] for i in range(len(frame_dicts)) if k in frame_dicts[i]]
        _write_outputs(out_dir, k, originals[k], anon, args.fps)

    anonymizer.shutdown()
    print(f"[preview] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
