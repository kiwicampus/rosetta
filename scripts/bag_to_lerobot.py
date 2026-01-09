#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 bag → LeRobot v3.0 exporter.

Overview
--------
This script converts one or more ROS 2 bags into a LeRobot v3 dataset using
the *same* contract-aware processing utilities used for live
inference. That keeps train/serve paths aligned and minimizes skew.

The conversion pipeline:

1) Load a "contract" describing topics/types/QoS/rate and feature shapes.
2) Scan a bag once; decode each contract topic using shared `decode_value`.
3) Select timestamps per a policy (`contract` / `bag` / `header`).
4) Resample each stream at the contract rate and assemble frames.
5) Coerce/resize images with the shared helpers and write to LeRobot.

Dependencies
------------
Only two shared modules (keep it unified with live inference):

- `rosetta.common.contract_utils`:
    `load_contract`, `iter_specs`, `feature_from_spec`
- `rosetta.common.processing_utils`:
    `decode_value`, `resample`, `stamp_from_header_ns`,
    `_nearest_resize_rgb`, `_coerce_to_uint8_rgb`, `zero_pad`

Command-line usage
------------------
Convert a single bag:

    $ python bag_to_lerobot.py \\
        --bag /path/to/my_bag \\
        --contract /path/to/contract.yaml
    
    # Output: out_lerobot/my_bag/

Convert multiple bags from a splits folder (containing split0, split1, ...):

    $ python bag_to_lerobot.py \\
        --bags /path/to/session_folder \\
        --contract /path/to/contract.yaml
    
    # Output: out_lerobot/session_folder/

Custom output root:

    $ python bag_to_lerobot.py \\
        --bags /path/to/session_folder \\
        --contract /path/to/contract.yaml \\
        --out /custom/output/root
    
    # Output: /custom/output/root/session_folder/

Options of note:

- `--timestamp {contract,bag,header}`
    How to pick per-message timestamps before resampling:
    * contract: per-spec `stamp_src` (default)
    * bag:      use the bag receive time
    * header:   prefer `msg.header.stamp` with bag time as fallback

- `--no-videos`
    Store PNG images instead of H.264/MP4 videos.

Outputs
-------
A LeRobot v3 dataset with:

- `videos/<image_key>/chunk-*/file-*.mp4`  (or `images/*/*.png` if `--no-videos`)
- `data/chunk-*/file-*.parquet`
- `meta/info.json`, `meta/tasks.parquet`, `meta/stats.json`
- `meta/episodes/*/*.parquet`

Notes
-----
- Image coercion uses shared helpers to consistently handle grayscale/alpha,
  float ranges, and nearest-neighbor resize.
- Feature dicts are built directly from `feature_from_spec()` so train-time and
  serve-time shapes match exactly.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import tempfile 
import os 
from pathlib import Path
from mcap.reader import make_reader
import bisect
# ---- LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---- Shared core (ONLY these two)
from rosetta.common.contract_utils import (
    load_contract,
    iter_specs,
    feature_from_spec,
    contract_fingerprint,
)
from rosetta.common.contract_utils import (
    decode_value,
    resample,
    stamp_from_header_ns,
    zero_pad as make_zero_pad,  # alias to avoid name clash with dict var
)

# Import decoders to register them
import rosetta.common.decoders  # noqa: F401

# ---------------------------------------------------------------------------


@dataclass
class _Stream:
    """Decoded per-topic stream buffers accumulated from a bag scan.

    Attributes
    ----------
    spec : Any
        The `SpecView` for this stream (observation or action).
    ros_type : str
        Fully-qualified ROS message type string for deserialization.
    ts : list[int]
        Per-message timestamps in nanoseconds (selected by policy).
    val : list[Any]
        Decoded values in contract-native form (e.g., HWC arrays for images).
        For images, this stores the log_time (MCAP log time) for direct decoding.
    log_times : list[int]
        For images: MCAP log times corresponding to each timestamp.
    """

    spec: Any
    ros_type: str
    ts: List[int]
    val: List[Any]
    is_image: bool = False  
    temp_dir: Optional[Path] = None
    log_times: List[int] = None
    
    def __post_init__(self):
        if self.log_times is None:
            self.log_times = [] 


# ---------------------------------------------------------------------------


def _read_yaml(p: Path) -> Dict[str, Any]:
    """Read a YAML file if it exists; return {} on absence/parse failures."""
    if not p.exists():
        print(f"[WARN] {p} does not exist")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    """Build a `{topic: type}` map from a rosbag2 reader."""
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

def _save_image_to_disk(img_array: np.ndarray, temp_dir: Path, idx: int)->str: #Nuevo
    filepath = temp_dir / f"img_{idx:08d}.npy"
    np.save(filepath, img_array)
    return str(filepath)

def _load_image_from_disk(filepath: str) -> np.ndarray:  #Nuevo
    return np.load(filepath)

# Only keeps frames within the window (50 frames before current position)
_frame_cache: Dict[Tuple[Path, str, int], Optional[np.ndarray]] = {}
_current_position: Dict[Tuple[Path, str], int] = {}  # Current frame index in log_times
_last_decoded_ts: Dict[Tuple[Path, str], int] = {}  # Last decoded log_time
CACHE_WINDOW_SIZE = 50

# MCAP readers cache: (bag_dir, topic) -> (file_handle, reader)
_mcap_readers: Dict[Tuple[Path, str], Tuple[Any, Any]] = {}

def _get_or_create_mcap_reader(bag_dir: Path, topic: str):
    """
    Get or create a MCAP reader for the given bag_dir and topic.
    
    Parameters
    ----------
    bag_dir : Path
        Path to the bag directory
    topic : str
        Topic name
        
    Returns
    -------
    tuple[Any, Any] or None
        (file_handle, reader) tuple, or None if MCAP file not found
    """
    reader_key = (bag_dir, topic)
    
    # Return cached reader if available
    if reader_key in _mcap_readers:
        return _mcap_readers[reader_key]
    
    # Create new reader
    try:
        mcap_files = list(bag_dir.glob("*.mcap"))
        if not mcap_files:
            print(f"[WARN] MCAP file not found in {bag_dir}")
            return None
        
        mcap_path = mcap_files[0]
        if not mcap_path.exists():
            print(f"[WARN] MCAP file not found: {mcap_path}")
            return None
        
        f = open(mcap_path, "rb")
        reader = make_reader(f)
        
        # Cache the reader
        _mcap_readers[reader_key] = (f, reader)
        return (f, reader)
        
    except Exception as e:
        print(f"[WARN] Failed to open MCAP file: {e}")
        return None


def _close_mcap_readers_for_bag(bag_dir: Path):
    """
    Close all MCAP readers for a specific bag_dir.
    This should be called after processing an episode is complete.
    
    Parameters
    ----------
    bag_dir : Path
        Path to the bag directory
    """
    keys_to_remove = []
    for (cached_bag_dir, topic), (f, reader) in _mcap_readers.items():
        if cached_bag_dir == bag_dir:
            try:
                f.close()
            except Exception:
                pass
            keys_to_remove.append((cached_bag_dir, topic))
    
    for key in keys_to_remove:
        _mcap_readers.pop(key, None)


def _load_frame_from_mcap(
    bag_dir: Path,
    topic: str,
    target_log_time: int,  # Direct log_time from resampling
    stream: _Stream,
    header_times: List[int],
    log_times: List[int],
    current_frame_idx: int = None  # Current frame index for window management
) -> Optional[np.ndarray]:
    """
    Decode a single frame from MCAP using the log_time directly.
    
    Parameters
    ----------
    bag_dir : Path
        Path to the bag directory
    topic : str
        Topic name for the image
    target_log_time : int
        Target log_time in nanoseconds (from resampling, no need to search)
    stream : _Stream
        Stream object containing spec and ros_type
    header_times : List[int]
        List of header timestamps (for reference)
    log_times : List[int]
        List of log times (for finding start position)
    current_frame_idx : int, optional
        Current frame index for window management (used to purge old frames)
        
    Returns
    -------
    np.ndarray or None
        Decoded image array or None if not found
    """
    if not header_times or not log_times:
        print("[WARN] No timestamps available")
        return None
    
    stream_key = (bag_dir, topic)
    cache_key = (bag_dir, topic, target_log_time)
    
    # Check cache first
    if cache_key in _frame_cache:
        return _frame_cache[cache_key]
    
    # Determine where to start decoding
    # Use last decoded position if available and target is after it
    start_log_time = log_times[0] if log_times else 0
    
    if stream_key in _last_decoded_ts:
        last_decoded_log_time = _last_decoded_ts[stream_key]
        # If target is after last decoded, start from there (more efficient)
        if target_log_time >= last_decoded_log_time:
            start_log_time = last_decoded_log_time
        else:
            # Target is before last decoded, need to reset decoder
            from rosetta.common.decoders import _decoder_state
            if topic in _decoder_state:
                del _decoder_state[topic]
            # Find the frame index in log_times to determine window start
            try:
                target_idx = log_times.index(target_log_time)
                # Start from 50 frames before target (window start)
                window_start_idx = max(0, target_idx - CACHE_WINDOW_SIZE)
                start_log_time = log_times[window_start_idx]
            except ValueError:
                start_log_time = log_times[0] if log_times else 0
    else:
        # First time decoding this stream - start from beginning
        start_log_time = log_times[0] if log_times else 0
    
    # Update current position for window management
    if current_frame_idx is not None:
        _current_position[stream_key] = current_frame_idx
        # Purge frames outside window
        _purge_cache_outside_window(stream_key, current_frame_idx, log_times)
    
    # Get or create MCAP reader (reused across calls)
    reader_result = _get_or_create_mcap_reader(bag_dir, topic)
    if reader_result is None:
        return None
    
    f, reader = reader_result
    
    # Decode sequentially from start_log_time to target_log_time
    # This ensures the H.264 decoder has proper context
    target_val = None
    
    try:
        for schema, channel, message in reader.iter_messages(
            topics=[topic],
            start_time=start_log_time
        ):
            msg = deserialize_message(message.data, get_message(stream.ros_type))
            
            # Get message timestamp and log_time
            msg_log_time = message.log_time
            
            # Decode the frame (needed for H.264 sequential decoding)
            # Check cache first to avoid re-decoding
            frame_cache_key = (bag_dir, topic, msg_log_time)
            if frame_cache_key in _frame_cache:
                val = _frame_cache[frame_cache_key]
            else:
                val = decode_value(stream.ros_type, msg, stream.spec)
                # Cache the decoded frame (window will be managed by purge function)
                if val is not None:
                    _frame_cache[frame_cache_key] = val
            
            # Update last decoded log_time
            _last_decoded_ts[stream_key] = msg_log_time
            
            # Check if this is the target frame (exact match by log_time)
            if msg_log_time == target_log_time:
                if val is not None:
                    target_val = val
                    break  # Found target, stop
            
            # Stop if we've passed the target
            if msg_log_time > target_log_time:
                break
        
        # Cache the result
        if target_val is not None:
            _frame_cache[cache_key] = target_val
        return target_val
        
    except Exception as e:
        print(f"[WARN] Failed to decode frame at log_time {target_log_time} from MCAP: {e}")
        import traceback
        traceback.print_exc()
        return None

def _purge_cache_outside_window(stream_key: Tuple[Path, str], current_idx: int, log_times: List[int]):
    """
    Purge frames from cache that are outside the window (more than 50 frames before current).
    
    Parameters
    ----------
    stream_key : Tuple[Path, str]
        (bag_dir, topic) tuple identifying the stream
    current_idx : int
        Current frame index in log_times
    log_times : List[int]
        List of log times for determining which frames to keep
    """
    if not log_times or current_idx < 0:
        return
    
    bag_dir, topic = stream_key
    window_start_idx = max(0, current_idx - CACHE_WINDOW_SIZE)
    
    # Determine which log_times are within the window
    if window_start_idx < len(log_times):
        window_log_times = set(log_times[window_start_idx:current_idx + 1])
    else:
        window_log_times = set()
    
    # Find all cache keys for this stream and purge those outside window
    keys_to_delete = []
    for cache_key in _frame_cache.keys():
        cache_bag_dir, cache_topic, cache_log_time = cache_key
        if cache_bag_dir == bag_dir and cache_topic == topic:
            if cache_log_time not in window_log_times:
                keys_to_delete.append(cache_key)
    
    # Delete frames outside window
    for key in keys_to_delete:
        frame = _frame_cache.pop(key, None)
        if frame is not None and isinstance(frame, np.ndarray):
            del frame

def _plan_streams(
    specs: Iterable[Any],
    tmap: Dict[str, str],
    temp_root: Path,
) -> Tuple[Dict[str, _Stream], Dict[str, List[str]]]:
    """Plan `_Stream` buffers for contract specs and build a topic dispatch index.

    Parameters
    ----------
    specs : Iterable[Any]
        Iterable of `SpecView` objects derived from the contract.
    tmap : dict[str, str]
        Map from topic name to ROS type in the bag.

    Returns
    -------
    streams : dict[str, _Stream]
        Mapping from contract key to `_Stream` state.
    by_topic : dict[str, list[str]]
        Mapping from topic name to a list of contract keys using it.

    Raises
    ------
    RuntimeError
        If none of the contract topics exist in the bag.
    """
    streams: Dict[str, _Stream] = {}
    by_topic: Dict[str, List[str]] = {}
    for sv in specs:
        if sv.topic not in tmap:
            # Derive a human-readable kind for logging without assuming SpecView internals.
            if hasattr(sv, "is_action") and sv.is_action:
                kind = "action"
            elif str(getattr(sv, "key", "")).startswith("task."):
                kind = "task"
            else:
                kind = "observation"
            print(
                f"[WARN] Missing {kind} '{getattr(sv, 'key', '?')}' topic in bag: {sv.topic}"
            )
            continue
        rt = sv.ros_type or tmap[sv.topic]
        
        # Create unique key for multiple observation.state specs and action specs
        if sv.key == "observation.state":
            unique_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
        elif sv.is_action:
            # For action specs, we need to check if there are multiple specs with the same key
            # This will be handled later in the consolidation logic
            unique_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
        else:
            unique_key = sv.key
            
        is_image = "observation.image" in sv.key

        streams[unique_key] = _Stream(spec=sv, ros_type=rt, ts=[], val=[], is_image=is_image, temp_dir=None)
        by_topic.setdefault(sv.topic, []).append(unique_key)

        
        if not streams:
            raise RuntimeError("No contract topics found in bag.")

    return streams, by_topic


def _detect_image_shapes_from_bag(
    bag_dir: Path,
    specs: List[Any],
) -> Dict[str, Tuple[int, int, int]]:
    """Pre-scan a bag to detect actual image shapes. To avoid errors when checking image size.
    
    Parameters
    ----------
    bag_dir : Path
        Path to the bag directory to scan.
    specs : list
        List of SpecView objects from the contract.
        
    Returns
    -------
    dict[str, tuple[int, int, int]]
        Mapping from feature key to detected (H, W, C) shape.
    """
    detected_shapes: Dict[str, Tuple[int, int, int]] = {}
    
    # Find image specs
    image_specs = [sv for sv in specs if sv.image_resize is not None]
    if not image_specs:
        return detected_shapes
    
    # Build topic -> spec mapping
    topic_to_spec: Dict[str, Any] = {}
    for sv in image_specs:
        topic_to_spec[sv.topic] = sv
    
    try:
        meta = _read_yaml(bag_dir / "metadata.yaml")
        info = meta.get("rosbag2_bagfile_information") or {}
        storage = info.get("storage_identifier") or "mcap"
        
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
        
        tmap = _topic_type_map(reader)
        topics_found = set()
        
        # Scan until we've found one image from each topic
        while reader.has_next() and len(topics_found) < len(topic_to_spec):
            topic, data, _ = reader.read_next()
            
            if topic not in topic_to_spec or topic in topics_found:
                continue
                
            sv = topic_to_spec[topic]
            ros_type = sv.ros_type or tmap.get(topic)
            if not ros_type:
                continue
                
            try:
                msg = deserialize_message(data, get_message(ros_type))
                val = decode_value(ros_type, msg, sv)
                
                if val is not None and isinstance(val, np.ndarray) and len(val.shape) == 3:
                    h, w, c = val.shape
                    detected_shapes[sv.key] = (h, w, c)
                    topics_found.add(topic)
                    print(f"[Auto-detect] {sv.key}: detected shape ({h}, {w}, {c})")
            except Exception as e:
                print(f"[WARN] Could not decode {topic} for shape detection: {e}")
                continue
                
    except Exception as e:
        print(f"[WARN] Could not pre-scan bag for image shapes: {e}")
    
    return detected_shapes


# ---------------------------------------------------------------------------

def export_bags_to_lerobot(
    bag_dirs: List[Path],
    contract_path: Path,
    out_root: Path,
    repo_id: str = "rosbag_v30",
    use_videos: bool = True,
    image_writer_threads: int = 6,
    image_writer_processes: int = 0,
    chunk_size: int = 1000,
    data_mb: int = 100,
    video_mb: int = 500,
    timestamp_source: str = "contract",  # 'contract' | 'receive' | 'header' |
) -> None:

    """Convert bag directories into a LeRobot v3 dataset under `out_root`.

    Parameters
    ----------
    bag_dirs : list[pathlib.Path]
        One or more bag directories (episodes) to convert.
    contract_path : pathlib.Path
        Path to the YAML/JSON contract. Must specify `rate_hz` and specs.
    out_root : pathlib.Path
        Root directory where the LeRobot dataset will be created/updated.
    repo_id : str, default "rosbag_v30"
        Dataset repo_id metadata stored by LeRobot.
    use_videos : bool, default True
        If True, store videos; otherwise store per-frame PNG images.
    image_writer_threads : int, default 4
        Worker threads per process for image writing.
            The optimal number of processes and threads depends on your computer capabilities.
            Lerobot advises to use 4 threads per camera with 0 processes. If the fps is not stable, try to increase or lower
            the number of threads. If it is still not stable, try to use 1 subprocess, or more.
    image_writer_processes : int, default 0
    chunk_size : int, default 1000
        Max number of frames per Parquet/video chunk.
    data_mb : int, default 100
        Target data file size in MB per chunk.
    video_mb : int, default 500
        Target video file size in MB per chunk.
    timestamp_source : {"contract","receive","header"}, default "contract"
        Timestamp selection policy per decoded message:
        - "contract": Use bag time, unless spec.stamp_src == "header" or spec.stamp_src == "foxglove"
        - "receive":  Always use bag receive time.
        - "header":   Prefer header stamp; fall back to bag receive time.

    Raises
    ------
    ValueError
        If contract `rate_hz` is invalid (<= 0).
    RuntimeError
        If a bag contains no usable/decodable messages.

    Notes
    -----
    - Feature shapes/dtypes are built from `feature_from_spec(..., use_videos)`,
      so your exported dataset matches online inputs exactly.
    - Image coercion uses shared utilities for consistent preprocessing.
    """
    # Contract + specs
    contract = load_contract(contract_path)
    fps = int(contract.rate_hz)
    if fps <= 0:
        raise ValueError("Contract rate_hz must be > 0")
    step_ns = int(round(1e9 / fps))
    specs = list(iter_specs(contract))

    # Pre-scan first bag to detect actual image shapes
    detected_shapes: Dict[str, Tuple[int, int, int]] = {}
    if bag_dirs:
        print("[Auto-detect] Scanning first bag to detect image shapes...")
        detected_shapes = _detect_image_shapes_from_bag(bag_dirs[0], specs)
        if detected_shapes:
            print(f"[Auto-detect] Detected shapes for {len(detected_shapes)} image streams")

    # Features (also detect first image key as anchor)
    features: Dict[str, Dict[str, Any]] = {}
    primary_image_key: Optional[str] = None
    state_specs = []  # Track multiple observation.state specs
    action_specs_by_key: Dict[str, List[Any]] = {}  # Track multiple action specs by key
    
    for sv in specs:
        # Handle multiple observation.state specs
        if sv.key == "observation.state":
            state_specs.append(sv)
            # Don't add to features yet - we'll consolidate them
            continue
            
        # Handle action specs
        if sv.is_action:
            if sv.key not in action_specs_by_key:
                action_specs_by_key[sv.key] = []
            action_specs_by_key[sv.key].append(sv)
            # Don't add to features yet - we'll consolidate them
            continue
            
        # Process other specs normally
        k, ft, is_img = feature_from_spec(sv, use_videos)
            
        # Ensure task.* specs are treated as per-frame strings even if the
        # underlying helper doesn't special-case them yet.
        if str(k).startswith(
            "task."
        ):  # TODO: why is this special-cased? Shouldn't this be handled in constract_utils?
            # Normalize to a simple scalar string field.
            features[k] = {"dtype": "string", "shape": [1]}
        else:
            # Special handling for depth images - they now have 3 channels
            if k.endswith(".depth") and ft["shape"][-1] == 1:
                # Update the shape to reflect 3 channels
                ft["shape"] = list(ft["shape"])
                ft["shape"][-1] = 3
            
            # Override shape with detected shape if available (for images)
            if is_img and k in detected_shapes:
                detected_h, detected_w, detected_c = detected_shapes[k]
                contract_shape = tuple(ft["shape"])
                if contract_shape != (detected_h, detected_w, detected_c):
                    print(f"[Auto-detect] {k}: overriding contract shape {contract_shape} -> ({detected_h}, {detected_w}, {detected_c})")
                    ft["shape"] = (detected_h, detected_w, detected_c)
            
            features[k] = ft
        if is_img and primary_image_key is None:
            primary_image_key = sv.key
    
    # Consolidate multiple observation.state specs into a single feature
    if state_specs:
        all_names = []
        total_shape = 0
        for sv in state_specs:
            all_names.extend(sv.names)
            total_shape += len(sv.names)
        
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (total_shape,),
            "names": all_names
        }
    
    # Consolidate multiple action specs with the same key into a single feature
    for action_key, action_specs in action_specs_by_key.items():
        if len(action_specs) > 1:
            # Multiple specs with same key - consolidate them
            all_names = []
            total_shape = 0
            for sv in action_specs:
                all_names.extend(sv.names)
                total_shape += len(sv.names)
            
            features[action_key] = {
                "dtype": "float32",
                "shape": (total_shape,),
                "names": all_names
            }
        else:
            # Single spec - use it as-is
            sv = action_specs[0]
            k, ft, _ = feature_from_spec(sv, use_videos)
            features[k] = ft
            
    # Mark depth videos in features metadata before dataset creation
    for key, feature in features.items():
        if key.endswith(".depth") and feature.get("dtype") == "video":
            if "info" not in feature:
                feature["info"] = {}
            feature["info"]["video.is_depth_map"] = True

    # Dataset
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_root,
        robot_type=contract.robot_type,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,  # keep simple & predictable
        image_writer_threads=image_writer_threads,
        batch_encoding_size=1,
    )
    
    # Persist the contract fingerprint into info.json so training can validate & propagate it
    try:
        fp = contract_fingerprint(contract)
        ds.meta.info["rosetta_fingerprint"] = fp
    except Exception:
        pass  # non-fatal; downstream will just skip the check
    ds.meta.update_chunk_settings(
        chunks_size=chunk_size,
        data_files_size_in_mb=data_mb,
        video_files_size_in_mb=video_mb,
    )
    

    # Precompute zero pads + shapes for fast frame assembly.
    zero_pad_map = {k: make_zero_pad(ft) for k, ft in features.items()}
    write_keys = [
        k
        for k, ft in features.items()
        if ft["dtype"] in ("video", "image", "float32", "float64", "string")
    ]
    shapes = {k: tuple(features[k]["shape"]) for k in write_keys}

    # Episodes
    for epi_idx, bag_dir in enumerate(bag_dirs):
        print(f"[Episode {epi_idx}] {bag_dir}")

        temp_episode_dir = Path(tempfile.mkdtemp(prefix=f"lerobot_images_ep{epi_idx}_")) 
        try:
            meta = _read_yaml(bag_dir / "metadata.yaml")
            info = meta.get("rosbag2_bagfile_information") or {}
            storage = info.get("storage_identifier") or "mcap"
            meta_dur_ns = int((info.get("duration") or {}).get("nanoseconds") or 0)

            # Operator prompt (if present). Accept either old/new keys gracefully.
            prompt = ""
            cd = info.get("custom_data")
            if isinstance(cd, dict):
                prompt = cd.get("prompt", prompt) or prompt   #value in custom data: prompt

            # Reader
            reader = rosbag2_py.SequentialReader()
            reader.open(
                rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
                rosbag2_py.ConverterOptions(
                    input_serialization_format="cdr",
                    output_serialization_format="cdr",
                ),
            )
        except Exception as e:
            print(f"⚠️  Skipping bag {bag_dir} due to error: {e}")
            continue

        tmap = _topic_type_map(reader)
        
        # Plan once - handle multiple observation.state specs and action specs
        streams, by_topic = _plan_streams(specs, tmap, temp_episode_dir)
        
        # Create consolidated observation.state stream if we have multiple state specs
        if state_specs:
            # Find all observation.state streams
            state_streams = [k for k in streams.keys() if k == "observation.state"]
            if len(state_streams) > 1:
                # Create a consolidated stream that will concatenate the data
                # We'll handle this in the frame processing
                pass
        
        # Counters for light diagnostics
        decoded_msgs = 0
        message_counter = 0

        # Extract timestamps only for images, decode non-images immediately
        while reader.has_next():
            topic, data, bag_ns = reader.read_next()
            message_counter += 1
            if topic not in by_topic:
                continue
            for key in by_topic[topic]:
                st = streams[key]
                sv = st.spec
                
                if st.is_image:
                    # For images: store header_time and log_time (bag_ns as approximation)
                    msg = deserialize_message(data, get_message(st.ros_type))
                    
                    # Timestamp selection policy (header_time for resampling)
                    if timestamp_source == "receive":
                        ts_sel = int(bag_ns)
                    elif timestamp_source == "header":
                        ts_sel = stamp_from_header_ns(msg) or int(bag_ns)
                    else:  # 'contract' (per-spec stamp_src)
                        if sv.stamp_src == "foxglove": #for compressed videos
                            time = msg.timestamp
                            ts_sel = int(time.sec) * 1_000_000_000 + int(time.nanosec)
                        else:
                            ts_sel = stamp_from_header_ns(msg) or int(bag_ns)
                    
                    # Store header_time for resampling and log_time (bag_ns) for direct MCAP seeking
                    log_time = int(bag_ns)  # Use bag_ns as approximation of MCAP log_time
                    st.ts.append(ts_sel)  # Header timestamp for resampling
                    st.val.append(log_time)  # Store log_time as value - resampling will select which log_time to use
                    st.log_times.append(log_time)  # store in log_times for direct lookup
                    decoded_msgs += 1
                else:
                    msg = deserialize_message(data, get_message(st.ros_type))
                    
                    # Timestamp selection policy
                    if timestamp_source == "receive":
                        ts_sel = int(bag_ns)
                    elif timestamp_source == "header":
                        ts_sel = stamp_from_header_ns(msg) or int(bag_ns)
                    else:  # 'contract' (per-spec stamp_src)
                        if sv.stamp_src == "foxglove": #for compressed videos
                            time = msg.timestamp
                            ts_sel = int(time.sec) * 1_000_000_000 + int(time.nanosec)
                        else:
                            ts_sel = stamp_from_header_ns(msg) or int(bag_ns)
                    
                    val = decode_value(st.ros_type, msg, sv)
                   
                    if val is not None:
                        st.ts.append(ts_sel)
                        st.val.append(val)
                        decoded_msgs += 1
                    else: 
                        print("not valid", st.ros_type)
        if decoded_msgs == 0:
            raise RuntimeError(f"No usable messages in {bag_dir} (none decoded).")
        
        # Choose anchor + duration
        valid_ts = [
            np.asarray(st.ts, dtype=np.int64) for st in streams.values() if st.ts
        ]
        if not valid_ts:
            raise RuntimeError(f"No usable messages in {bag_dir} (no timestamps).")
        if (
            primary_image_key
            and streams.get(primary_image_key)
            and streams[primary_image_key].ts
        ):
            start_ns = int(
                np.asarray(streams[primary_image_key].ts, dtype=np.int64).min()
            )
        else:
            start_ns = int(min(ts.min() for ts in valid_ts))

        ts_max = int(max(ts.max() for ts in valid_ts))
        observed_dur_ns = max(0, ts_max - start_ns)

        # Prefer observed duration unless bag metadata matches within ~2 ticks.
        if meta_dur_ns > 0 and abs(meta_dur_ns - observed_dur_ns) <= 2 * step_ns:
            dur_ns = meta_dur_ns
            print("Using duration from metadata")
        else:
            dur_ns = observed_dur_ns
            print(
                "Metadata duration disagrees with observed duration. Using observed duration"
            )
        # Ticks
        n_ticks = int(dur_ns // step_ns) + 1
        ticks_ns = start_ns + np.arange(n_ticks, dtype=np.int64) * step_ns        

        # Resample onto ticks
        resampled: Dict[str, List[Any]] = {}
        for key, st in streams.items():
            if not st.ts:
                resampled[key] = [None] * n_ticks
                continue
            ts = np.asarray(st.ts, dtype=np.int64)
            pol = st.spec.resample_policy
            resampled[key] = resample(
                pol, ts, st.val, ticks_ns, step_ns, st.spec.asof_tol_ms
            )
            
        safe_window = 10
        active_images = {}

        print("Resampling done")
        # Write frames

        for i in range(n_ticks):
            frame: Dict[str, Any] = {}
            

            # Handle consolidated observation.state by concatenating multiple state streams first
            if "observation.state" in features and state_specs:
                # Concatenate all observation.state values from different topics
                state_values = []
                for sv in state_specs:
                    unique_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
                    stream_val = resampled.get(unique_key, [None] * n_ticks)[i]
                    if stream_val is not None:
                        val_array = np.asarray(stream_val, dtype=np.float32).reshape(-1)
                        state_values.append(val_array)
                    else:
                        # Use zero padding for missing topics to maintain expected shape
                        expected_size = len(sv.names)
                        zero_pad = np.zeros((expected_size,), dtype=np.float32)
                        state_values.append(zero_pad)
                
                if state_values:
                    # Concatenate all state values
                    concatenated_state = np.concatenate(state_values)
                    frame["observation.state"] = concatenated_state
                else:
                    # Use zero padding if no state values available
                    frame["observation.state"] = zero_pad_map["observation.state"]
            
            # Handle consolidated action specs by concatenating multiple action streams
            for action_key, action_specs in action_specs_by_key.items():
                if action_key not in features:
                    continue

                action_values = []
                for sv in action_specs:
                    unique_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
                    stream_val = resampled.get(unique_key, [None] * n_ticks)[i]
                    if stream_val is not None:
                        val_array = np.asarray(stream_val, dtype=np.float32).reshape(-1)
                        action_values.append(val_array)

                if action_values:
                    concatenated_action = np.concatenate(action_values)
                    frame[action_key] = concatenated_action
                else:
                    frame[action_key] = zero_pad_map[action_key]

                        
            # Process all other features
            for name in write_keys:
                # Skip observation.state as it's handled above
                if name == "observation.state":
                    continue
                
                # Skip consolidated actions as they're handled above
                if name in action_specs_by_key: 
                    continue
                    
                ft = features[name]
                dtype = ft["dtype"]
                val = resampled.get(name, [None] * n_ticks)[i]

                if val is None:
                    frame[name] = zero_pad_map[name]
                    continue

                if dtype in ("video", "image"): 
                    # val contains the log_time selected by resampling
                    resampled_log_time = val
                    
                    st = streams[name]
                    # Decode from MCAP using the resampled log_time directly 
                    # Find the index of this log_time in st.log_times for window management
                    try:
                        frame_idx = st.log_times.index(int(resampled_log_time))
                    except ValueError:
                        frame_idx = None
                    
                    arr = _load_frame_from_mcap(
                        bag_dir=bag_dir,
                        topic=st.spec.topic,
                        target_log_time=int(resampled_log_time),
                        stream=st,
                        header_times=st.ts,
                        log_times=st.log_times,
                        current_frame_idx=frame_idx
                    )
                    if arr is None:
                        # Decode failed, use zero padding
                        frame[name] = zero_pad_map[name]
                        continue
                
                    # Ensure deterministic storage; lerobot loaders will map back to float [0,1]
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    frame[name] = arr
                    

                elif dtype in ("float32", "float64"):
                    tgt_dt = np.float32 if dtype == "float32" else np.float64
                    arr = np.asarray(val, dtype=tgt_dt).reshape(-1)
                    exp = int(ft["shape"][0])
                    if arr.shape[0] != exp:
                        fixed = np.zeros((exp,), dtype=tgt_dt)
                        fixed[: min(exp, arr.shape[0])] = arr[: min(exp, arr.shape[0])]
                        arr = fixed
                    frame[name] = arr
                    
                elif dtype == "string":
                    frame[name] = str(val)

                else:
                    # Fallback – should not happen with current features
                    frame[name] = val

            # Episode-level operator prompt from bag metadata (kept for policy compatibility).
            # This is`` distinct from any per-frame task.* fields coming from ROS topics.
            # LeRobot requires 'task' field in every frame, so always set it (empty string if no prompt).
            frame["task"] = prompt if prompt else ""
            ds.add_frame(frame)

        ds.save_episode()
        print(
            f"  → saved {n_ticks} frames @ {int(round(fps))} FPS  | decoded_msgs={decoded_msgs}"
        )
        
        # Close MCAP readers for this episode to free resources
        _close_mcap_readers_for_bag(bag_dir)

    
    print(f"\n[OK] Dataset root: {ds.root.resolve()}")
    if use_videos:
        print("  - videos/<image_key>/chunk-*/file-*.mp4")
    else:
        print("  - images/*/*.png")
    print("  - data/chunk-*/file-*.parquet")
    print(
        "  - meta/info.json, meta/tasks.parquet, meta/stats.json, meta/episodes/*/*.parquet"
    )


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line args for bag → LeRobot conversion."""
    ap = argparse.ArgumentParser("ROS2 bag → LeRobot v3 (using rosetta.common.*)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bag", help="Path to a single bag directory (episode)")
    g.add_argument("--bags", help="Path to folder containing split0, split1, ... subfolders")
    ap.add_argument("--contract", required=True, help="Path to YAML contract")
    ap.add_argument("--out", default="out_lerobot", help="Output root directory (default: out_lerobot)")
    ap.add_argument("--repo-id", default="rosbag_v30", help="repo_id metadata")
    ap.add_argument(
        "--no-videos", action="store_true", help="Store images instead of videos"
    )
    ap.add_argument("--image-threads", type=int, default=6, help="Image writer threads")
    ap.add_argument(
        "--image-processes", type=int, default=0, help="Image writer processes"
    )
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--data-mb", type=int, default=100)
    ap.add_argument("--video-mb", type=int, default=500)
    ap.add_argument(
        "--timestamp",
        choices=("contract", "bag", "header"),
        default="contract",
        help=(
            "Which time base to use when resampling: "
            "'contract' (per-spec), 'bag' (receive), or 'header' (message header)."
        ),
    )
    return ap.parse_args()


def _discover_split_folders(parent_dir: Path) -> List[Path]:
    """Discover split folders (split0, split1, ...) in the given directory.
    
    Parameters
    ----------
    parent_dir : Path
        Parent directory containing split* subfolders.
        
    Returns
    -------
    list[Path]
        Sorted list of split folder paths.
        
    Raises
    ------
    ValueError
        If no split folders are found.
    """
    import re
    
    split_pattern = re.compile(r'^split(\d+)$')
    splits = []
    
    for child in parent_dir.iterdir():
        if child.is_dir():
            match = split_pattern.match(child.name)
            if match:
                split_num = int(match.group(1))
                splits.append((split_num, child))
    
    if not splits:
        raise ValueError(f"No split folders (split0, split1, ...) found in {parent_dir}")
    
    # Sort by split number
    splits.sort(key=lambda x: x[0])
    sorted_paths = [p for _, p in splits]
    
    print(f"[Splits] Found {len(sorted_paths)} splits in {parent_dir}:")
    for p in sorted_paths:
        print(f"  - {p.name}")
    
    return sorted_paths


def main() -> None:
    """CLI entry point for batch conversion of ROS 2 bags to LeRobot."""
    args = parse_args()
    
    if args.bag:
        input_path = Path(args.bag)
        bag_dirs = [input_path]
    else:
        # Discover split folders from the provided directory
        input_path = Path(args.bags)
        if not input_path.exists():
            raise FileNotFoundError(f"Splits folder not found: {input_path}")
        bag_dirs = _discover_split_folders(input_path)
    
    # Build output path: out_root/input_folder_name
    out_root = Path(args.out)
    input_folder_name = input_path.name
    final_out_path = out_root / input_folder_name
    
    print(f"[Output] Dataset will be saved to: {final_out_path}")
    
    export_bags_to_lerobot(
        bag_dirs=bag_dirs,
        contract_path=Path(args.contract),
        out_root=final_out_path,
        repo_id=args.repo_id,
        use_videos=not args.no_videos,
        image_writer_threads=args.image_threads,
        image_writer_processes=args.image_processes,
        chunk_size=args.chunk_size,
        data_mb=args.data_mb,
        video_mb=args.video_mb,
        timestamp_source=args.timestamp,
    )


if __name__ == "__main__":
    main()
