#!/usr/bin/env python3
# Copyright 2025 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
ROS2 bag → LeRobot dataset porting script.

Converts rosbag recordings to LeRobot datasets using contract-driven decoding.
Uses the same decoders and resampling as live inference for consistency.

Usage:
    # Port all bags
    python -m rosetta.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_dataset \\
        --contract /path/to/contract.yaml

    # Port a single shard (for SLURM parallel processing)
    python -m rosetta.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_dataset \\
        --contract /path/to/contract.yaml \\
        --num-shards 100 \\
        --shard-index 0

    # Push to HuggingFace Hub
    python -m rosetta.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_org/my_dataset \\
        --contract /path/to/contract.yaml \\
        --push-to-hub
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
import time
from typing import Any

from PIL import Image as PILImage

from lerobot.configs.video import VALID_VIDEO_CODECS
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import SUPPORTED_IMAGE_FORMATS
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import yaml

from .common import decoders as _decoders  # noqa: F401, E402
from .common import encoders as _encoders  # noqa: F401, E402

from .common.contract import load_contract, ObservationStreamSpec, StreamSpec
from .common.contract_utils import (
    build_feature,
    get_namespaced_names,
    iter_specs,
    StreamBuffer,
    zeros_for_spec,
)
from .common.converters import decode_value, DTYPES, get_decoder_dtype
from .bag_video_encoder import BagCameraPlan, PlannedBagVideoEncoder
from .common.ros2_utils import get_message_timestamp_ns
from .common.stamp_guard import StampMonotonicityGuard
from .h264_passthrough import MSG_TYPE as COMPRESSED_VIDEO, StartGate

# Bag metadata keys
BAG_METADATA_KEY = 'rosbag2_bagfile_information'
BAG_CUSTOM_DATA_KEY = 'custom_data'
BAG_PROMPT_KEY = 'lerobot.operator_prompt'

COMPRESSED_IMAGE = 'sensor_msgs/msg/CompressedImage'


def _resolve_msg_format(raw_format: str) -> str | None:
    """Extract a lerobot-recognized codec name from a ROS CompressedImage format string."""
    lowered = raw_format.lower()
    for fmt in SUPPORTED_IMAGE_FORMATS:
        if fmt in lowered:
            return fmt
    return None


# ---------- Bag discovery ----------


def find_bag_dirs(raw_dir: Path) -> list[Path]:
    """Find all bag directories (contain *.mcap files)."""
    bag_dirs = sorted({p.parent for p in raw_dir.rglob('*.mcap')})
    if not bag_dirs:
        raise RuntimeError(f'No bag directories found in {raw_dir}')
    return bag_dirs


# ---------- Internal helpers ----------


def _read_bag_metadata(bag_dir: Path) -> dict[str, Any]:
    """Read bag metadata.yaml."""
    meta_path = bag_dir / 'metadata.yaml'
    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        return yaml.safe_load(f) or {}


def _read_prompt(meta: dict[str, Any]) -> str | None:
    """Read prompt from metadata custom_data. Returns None if not found."""
    info = meta.get(BAG_METADATA_KEY, {})
    custom_data = info.get(BAG_CUSTOM_DATA_KEY, {})
    if isinstance(custom_data, dict):
        return custom_data.get(BAG_PROMPT_KEY) or None
    return None


def _peek_compressed_image_sizes(
    bag_dir: Path,
    specs: list[ObservationStreamSpec],
) -> dict[str, tuple[int, int]]:
    """Return actual (height, width) for each CompressedImage spec in bag_dir.

    Stops as soon as every topic has been seen once.

    If a topic is not found or its payload can't be parsed, it is omitted from
    the result so the caller falls back to scheduling a resize for it.
    """
    topic_to_key: dict[str, str] = {spec.topic: spec.key for spec in specs}
    remaining: set[str] = set(topic_to_key)
    sizes: dict[str, tuple[int, int]] = {}

    meta = _read_bag_metadata(bag_dir)
    info = meta.get(BAG_METADATA_KEY, {})
    storage_id = info.get('storage_identifier', 'mcap')
    bag_files = list(bag_dir.glob(f'*.{storage_id}')) or list(bag_dir.glob('*.mcap'))
    uri = str(bag_files[0]) if bag_files else str(bag_dir)
    if bag_files:
        storage_id = bag_files[0].suffix.lstrip('.')

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    compressed_image_msg_type = get_message('sensor_msgs/msg/CompressedImage')

    while reader.has_next() and remaining:
        topic, data, _ = reader.read_next()
        if topic not in remaining:
            continue
        try:
            msg = deserialize_message(data, compressed_image_msg_type)
            raw = bytes(msg.data)
            if not raw:
                continue
            pil_img = PILImage.open(io.BytesIO(raw))
            sizes[topic_to_key[topic]] = (pil_img.height, pil_img.width)
        except Exception:
            pass
        remaining.discard(topic)

    return sizes


def _first_storage_file(bag_dir: Path) -> Path:
    """Return the bag's storage file, preferring the identifier metadata.yaml declares."""
    info = _read_bag_metadata(bag_dir).get(BAG_METADATA_KEY, {})
    storage_id = info.get('storage_identifier', 'mcap')
    files = list(bag_dir.glob(f'*.{storage_id}')) or list(bag_dir.glob('*.mcap'))
    if not files:
        raise RuntimeError(f'No storage file found in {bag_dir}')
    return files[0]


def _get_topic_types(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    """Get topic -> type mapping from bag."""
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def _is_differentiated(spec: StreamSpec) -> bool:
    """True when a spec's values come from the derivative pass, not a buffer."""
    return bool(getattr(spec, 'differentiate', False))


def _build_buffers(
    specs: list[StreamSpec],
    topic_types: dict[str, str],
) -> dict[str, tuple[StreamSpec, StreamBuffer]]:
    """
    Create StreamBuffers keyed by topic.

    Returns
    -------
        Topic-keyed dict: topic -> (spec, buffer), preserving insertion order.

    """
    buffers: dict[str, tuple[StreamSpec, StreamBuffer]] = {}

    for spec in specs:
        if spec.topic not in topic_types:
            logging.warning('Topic %s not in bag, skipping %s', spec.topic, spec.key)
            continue

        # Derivative specs are handled by _precompute_derivatives, not StreamBuffer
        if _is_differentiated(spec):
            continue

        if isinstance(spec, ObservationStreamSpec):
            buffer = StreamBuffer.from_spec(spec)
        else:
            step_ns = int(1e9 / spec.fps) if spec.fps > 0 else int(1e9 / 30)
            buffer = StreamBuffer(policy='hold', step_ns=step_ns, tol_ns=0)

        buffers[spec.topic] = (spec, buffer)

    if not buffers:
        raise RuntimeError('No contract topics found in bag')

    return buffers


def _build_features(specs: list[StreamSpec]) -> dict[str, dict[str, Any]]:
    """
    Build LeRobot feature definitions from contract specs.

    Specs sharing the same key are aggregated (names concatenated for vectors).
    """
    # Group specs by output key
    by_key: dict[str, list[StreamSpec]] = {}
    for spec in specs:
        by_key.setdefault(spec.key, []).append(spec)

    features = {}
    for key, key_specs in by_key.items():
        first = key_specs[0]
        dtype = DTYPES[first.msg_type]

        if dtype in ('video', 'image'):
            # Images: no aggregation
            features[key] = build_feature(first)
        elif dtype == 'string':
            # Strings: no aggregation
            features[key] = build_feature(first)
        else:
            # Numeric: aggregate names from all specs
            all_names = []
            for spec in key_specs:
                all_names.extend(spec.names if spec.names else get_namespaced_names(spec))
            n = len(all_names) or 1
            features[key] = {
                'dtype': dtype,
                'shape': (n,),
                'names': all_names if all_names else None,
            }

    return features


def _get_bag_time_bounds_ns(reader: rosbag2_py.SequentialReader) -> tuple[int, int]:
    """Get time bounds from bag metadata."""
    metadata = reader.get_metadata()
    start_time = metadata.starting_time
    duration = metadata.duration
    # rosbag2_py returns Time/Duration objects with .nanoseconds property
    start_ns = start_time.nanoseconds
    duration_ns = duration.nanoseconds
    return start_ns, start_ns + duration_ns


def _nearest_idx(timestamps: np.ndarray, t: int) -> int:
    """Return index of the timestamp in `timestamps` nearest to `t`."""
    idx = np.searchsorted(timestamps, t)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    return idx - 1 if abs(timestamps[idx - 1] - t) <= abs(timestamps[idx] - t) else idx


def _precompute_derivatives(
    uri: str,
    storage_id: str,
    deriv_specs: list[StreamSpec],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """First pass: collect full-rate position data and compute velocity via np.gradient.

    np.gradient(positions, median_dt, axis=0)
    where median_dt = median of positive inter-sample intervals in seconds.

    Returns {topic: (timestamps_ns, velocities)} where velocities has shape (N, D).
    """
    topic_to_spec = {s.topic: s for s in deriv_specs}
    topic_history: dict[str, list[tuple[int, np.ndarray]]] = {s.topic: [] for s in deriv_specs}

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    while reader.has_next():
        topic, data, bag_ns = reader.read_next()
        if topic not in topic_history:
            continue
        spec = topic_to_spec[topic]
        msg = deserialize_message(data, get_message(spec.msg_type))
        ts, _ = get_message_timestamp_ns(msg, spec, bag_ns)
        val = decode_value(msg, spec)
        if val is not None:
            topic_history[topic].append((ts, np.asarray(val, dtype=np.float64).flatten()))

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for topic, history in topic_history.items():
        if len(history) < 2:
            logging.warning('Not enough samples for derivative on topic %s (%d)', topic, len(history))
            continue
        timestamps = np.array([h[0] for h in history], dtype=np.float64)
        positions = np.stack([h[1] for h in history])  # (N, D)

        dts_s = np.diff(timestamps) / 1e9
        pos_dts = dts_s[dts_s > 0]
        if pos_dts.size == 0:
            continue
        median_dt = float(np.median(pos_dts))
        velocities = np.gradient(positions, median_dt, axis=0)  # central differences

        result[topic] = (timestamps, velocities)

    return result


# Map LeRobot dtype strings to numpy dtypes
DTYPE_MAP = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
    'bool': bool,
}


def _derivative_value(
    spec: StreamSpec,
    tick_ns: int,
    derivatives: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    np_dtype: Any,
) -> np.ndarray:
    """Pre-computed velocity for `spec` at `tick_ns` (zeros when unavailable)."""
    entry = derivatives.get(spec.topic) if derivatives else None
    if entry is None:
        return np.zeros(max(len(spec.names), 1), dtype=np_dtype)
    ts_arr, vel_arr = entry
    return vel_arr[_nearest_idx(ts_arr, tick_ns), :len(spec.names)].astype(np_dtype)


def _sample_frame(
    tick_ns: int,
    buffers: dict[str, tuple[StreamSpec, StreamBuffer]],
    specs: list[StreamSpec],
    derivatives: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """
    Sample a single frame from buffers at the given tick time.

    Specs sharing the same key are aggregated into one vector, in the order
    `specs` declares them -- the order `_build_features` names the columns in.
    Differentiated specs hold no StreamBuffer (their values come from the
    pre-computed `derivatives`) but still own their declared column slot.
    """
    # Contract order. A regular spec contributes once its topic is in the bag;
    # a differentiated spec always does, or the columns after it would shift.
    by_key: dict[str, list[StreamSpec]] = {}
    for spec in specs:
        if spec.topic in buffers or _is_differentiated(spec):
            by_key.setdefault(spec.key, []).append(spec)

    def sample(spec: StreamSpec) -> Any:
        entry = buffers.get(spec.topic)
        return entry[1].sample(tick_ns) if entry is not None else None

    frame: dict[str, Any] = {}

    for key, key_specs in by_key.items():
        # The first contributing spec classifies the key type
        first_spec = key_specs[0]

        if isinstance(first_spec, ObservationStreamSpec) and first_spec.is_image:
            # Image: single value (no aggregation)
            val = sample(first_spec)
            if val is None:
                frame[key] = zeros_for_spec(first_spec)
            elif isinstance(val, (bytes, bytearray)):
                frame[key] = val
            elif isinstance(val, (int, np.integer)):
                # plan_video buffers log times, not pixels; pass them through.
                frame[key] = int(val)
            else:
                frame[key] = np.asarray(val, dtype=np.uint8)
        elif isinstance(first_spec, ObservationStreamSpec) and first_spec.dtype == 'string':
            # String: pass through
            val = sample(first_spec)
            frame[key] = str(val) if val is not None else ''
        elif isinstance(first_spec, ObservationStreamSpec) and first_spec.dtype in (
            'bool',
            'int32',
            'int64',
        ):
            # Scalar types: single value
            val = sample(first_spec)
            np_dtype = DTYPE_MAP[first_spec.dtype]  # already validated above
            if val is None:
                frame[key] = np.zeros(1, dtype=np_dtype)
            else:
                frame[key] = np.asarray(val, dtype=np_dtype).flatten()
        else:
            # Vector: concatenate all specs with this key
            # Determine dtype from spec or decoder registry
            if isinstance(first_spec, ObservationStreamSpec):
                dtype_str = first_spec.dtype
            else:
                # ActionStreamSpec: get dtype from decoder registry
                dtype_str = get_decoder_dtype(first_spec.msg_type)

            if dtype_str not in DTYPE_MAP:
                raise ValueError(
                    f"Unsupported dtype '{dtype_str}' for key '{key}'. Add to DTYPE_MAP."
                )
            np_dtype = DTYPE_MAP[dtype_str]

            values = []
            for spec in key_specs:
                if _is_differentiated(spec):
                    values.append(_derivative_value(spec, tick_ns, derivatives, np_dtype))
                    continue
                val = sample(spec)
                if val is None:
                    val = np.zeros(max(len(spec.names), 1), dtype=np_dtype)
                else:
                    val = np.asarray(val, dtype=np_dtype).flatten()
                values.append(val)

            frame[key] = np.concatenate(values) if len(values) > 1 else values[0]

    return frame


def _stream_frames_from_bag(
    bag_dir: Path,
    specs: list[StreamSpec],
    prompt: str = '',
    plan_video: bool = False,
    max_stamp_backward_jump_s: float = 0.0,
    h264_gates: dict[str, StartGate] | None = None,
):
    """
    Stream LeRobot frames from a bag file.

    Uses StreamBuffer for resampling (identical to live inference).
    Specs sharing the same key are aggregated into single tensors.
    Header stamped streams are guarded against backward stamp jumps
    (see StampMonotonicityGuard); a negative tolerance disables that.
    h264_gates, when given, receives the StartGate of each H.264 key.
    """
    fps = specs[0].fps
    step_ns = int(1e9 / fps)

    meta = _read_bag_metadata(bag_dir)
    info = meta.get(BAG_METADATA_KEY, {})
    storage_id = info.get('storage_identifier', 'mcap')

    # Open storage file directly when available (avoids metadata.yaml format issues)
    bag_files = list(bag_dir.glob(f'*.{storage_id}')) or list(bag_dir.glob('*.mcap'))
    uri = str(bag_files[0]) if bag_files else str(bag_dir)
    if bag_files:
        storage_id = bag_files[0].suffix.lstrip('.')

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    topic_types = _get_topic_types(reader)
    buffers = _build_buffers(specs, topic_types)

    # Pre-compute velocity for differentiate=true specs (observations and actions)
    deriv_specs = [s for s in specs if _is_differentiated(s) and s.topic in topic_types]
    derivatives = _precompute_derivatives(uri, storage_id, deriv_specs) if deriv_specs else {}

    start_ns, end_ns = _get_bag_time_bounds_ns(reader)
    n_frames = max(1, int((end_ns - start_ns) // step_ns) + 1)

    current_tick_idx = 0
    current_tick_ns = start_ns
    stamp_guard = StampMonotonicityGuard(max_stamp_backward_jump_s)
    header_warned: set[str] = set()
    filled_topics: set[str] = set()
    required_topics: set[str] = set(buffers.keys())
    key_formats: dict[str, str] = {}  # output key -> format string (e.g. 'jpeg', 'png')
    # H.264 topics count from their first IDR: nothing before it decodes.
    gates: dict[str, StartGate] = {} if h264_gates is None else h264_gates

    while reader.has_next():
        topic, data, bag_ns = reader.read_next()

        all_warm = required_topics.issubset(filled_topics)
        while current_tick_idx < n_frames and bag_ns >= current_tick_ns:
            if all_warm: # Only emit frames after all required topics have been filled at least once
                frame = _sample_frame(current_tick_ns, buffers, specs, derivatives)
                frame['task'] = prompt
                yield frame, key_formats
            current_tick_idx += 1
            current_tick_ns = start_ns + current_tick_idx * step_ns

        if topic in buffers:
            spec, buffer = buffers[topic]
            msg = deserialize_message(data, get_message(spec.msg_type))

            ts, used_fallback = get_message_timestamp_ns(msg, spec, bag_ns)
            if spec.stamp_src == 'header' and used_fallback and spec.key not in header_warned:
                logging.warning(
                    "Header stamp unavailable for '%s' in %s, using bag receive time",
                    spec.key,
                    bag_dir.name,
                )
                header_warned.add(spec.key)
            if spec.stamp_src == 'header':
                stamp_guard.check(topic, spec.key, ts)
            if isinstance(spec, ObservationStreamSpec) and spec.is_image:
                if spec.msg_type == COMPRESSED_VIDEO:
                    if not plan_video:
                        raise RuntimeError(f'{spec.key}: H.264 topics need the planned video path')
                    gate = gates.setdefault(spec.key, StartGate(spec.key))
                    if gate.accept(int(bag_ns), bytes(msg.data)):
                        buffer.push(ts, int(bag_ns))
                        filled_topics.add(topic)
                elif spec.msg_type == COMPRESSED_IMAGE:
                    # Push raw bytes — lerobot sniffs the format from the bytes
                    # unless we resolve a codec name from msg.format below.
                    raw = bytes(msg.data)
                    if raw:
                        if spec.key not in key_formats and hasattr(msg, 'format') and msg.format:
                            resolved = _resolve_msg_format(msg.format)
                            if resolved is not None:
                                key_formats[spec.key] = resolved
                        # Buffer the log time, not the bytes: the resampled column is
                        # then the plan the encoder consumes, and no JPEG is held.
                        buffer.push(ts, int(bag_ns) if plan_video else raw)
                        filled_topics.add(topic)
                else:
                    # Uncompressed image (sensor_msgs/msg/Image) — decode to numpy array (resize in decoder)
                    val = decode_value(msg, spec)
                    if val is not None:
                        buffer.push(ts, val)
                        filled_topics.add(topic)
            else:
                val = decode_value(msg, spec)
                if val is not None:
                    buffer.push(ts, val)
                    filled_topics.add(topic)


    # Emit remaining frames 
    while current_tick_idx < n_frames:
        frame = _sample_frame(current_tick_ns, buffers, specs, derivatives)
        frame['task'] = prompt
        yield frame, key_formats

        current_tick_idx += 1
        current_tick_ns = start_ns + current_tick_idx * step_ns


# ---------- Main porting function ----------


def port_bags(
    raw_dir: Path,
    repo_id: str,
    contract_path: Path,
    root: Path | None = None,
    prompt: str | None = None,
    push_to_hub: bool = False,
    num_shards: int | None = None,
    shard_index: int | None = None,
    encoding_kwargs: dict | None = None,
    batch_encoding_size: int = 1,
    image_writer_threads: int = 8,
    fast_video: bool = True,
    max_stamp_backward_jump_s: float = 0.0,
):
    """
    Port ROS2 bags to LeRobot dataset format.

    Args:
        raw_dir: Directory containing bag subdirectories.
        repo_id: HuggingFace repository ID (e.g., "my_org/my_dataset").
        contract_path: Path to Rosetta contract YAML.
        root: Output directory for dataset. Defaults to ~/.cache/huggingface/lerobot.
        prompt: Prompt string. If None, reads the prompt from each bag's
            metadata.yaml custom_data. Raises if no prompt can be found for a bag.
        push_to_hub: Whether to upload to HuggingFace Hub after porting.
        num_shards: Total number of shards for parallel processing.
        shard_index: Index of this shard (0 to num_shards-1).
        encoding_kwargs: Keyword arguments forwarded to ``encode_video_frames``
            (e.g. vcodec, pix_fmt, g, crf, fast_decode).
        batch_encoding_size: Number of episodes per encoding batch. Defaults to 1 for immediate encoding.
        image_writer_threads: Number of image-writer threads for parallel frame writes.
            Set to 0 to disable the thread pool.
        max_stamp_backward_jump_s: Largest tolerated backward stamp jump on a
            header stamped stream, in seconds. 0 rejects any regression while
            allowing repeated stamps; a negative value disables the check.
    """
    contract = load_contract(contract_path)
    specs = list(iter_specs(contract))
    features = _build_features(specs)

    all_bag_dirs = find_bag_dirs(raw_dir)
    total_bags = len(all_bag_dirs)
    logging.info('Found %d bags in %s', total_bags, raw_dir)

    # Select shard subset if sharding
    if num_shards is not None:
        if shard_index is None:
            raise ValueError('shard_index required when num_shards is specified')
        if shard_index >= num_shards:
            raise ValueError(f'shard_index ({shard_index}) >= num_shards ({num_shards})')

        bag_dirs = all_bag_dirs[shard_index::num_shards]
        logging.info('Shard %d/%d: processing %d bags', shard_index, num_shards, len(bag_dirs))
    else:
        bag_dirs = all_bag_dirs

    if not bag_dirs:
        logging.warning('No bags to process in this shard')
        return

    # LeRobot uses root directly as dataset path, so append repo_id
    dataset_root = root / repo_id if root else None
    _encoding_kwargs = dict(encoding_kwargs or {})
    vcodec = _encoding_kwargs.pop("vcodec", "libsvtav1")
    bitrate = _encoding_kwargs.pop("bitrate", None)
    image_specs_for_plan = [
        spec
        for spec in specs
        if isinstance(spec, ObservationStreamSpec)
        and spec.is_image
        and spec.msg_type in (COMPRESSED_IMAGE, COMPRESSED_VIDEO)
        and spec.image_resize
    ]
    if not fast_video and any(s.msg_type == COMPRESSED_VIDEO for s in image_specs_for_plan):
        raise ValueError('H.264 topics are copied by the planned video path; drop --no-fast-video')
    video_encoder = None
    if fast_video and image_specs_for_plan:
        video_encoder = PlannedBagVideoEncoder(
            fps=int(contract.fps),
            vcodec=vcodec,
            pix_fmt=_encoding_kwargs.get('pix_fmt', 'yuv420p'),
            codec_options={
                k: str(v)
                for k, v in (
                    ('g', _encoding_kwargs.get('g', 2)),
                    ('crf', _encoding_kwargs.get('crf')),
                    ('b', bitrate),
                )
                if v is not None
            },
        )
        logging.info(
            'fast video: MCAP -> MP4 directly for %d camera(s), no intermediate JPEGs',
            len(image_specs_for_plan),
        )

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=contract.robot_type,
        fps=contract.fps,
        features=features,
        vcodec=vcodec,
        streaming_encoder=video_encoder,
        encoding_kwargs=_encoding_kwargs or None,
        batch_encoding_size=batch_encoding_size,
        image_writer_threads=image_writer_threads,
    )
    # Build per-camera resize map for CompressedImage keys only.
    # sensor_msgs/msg/Image keys are already resized by the decoder; bytes keys are passthrough
    # and need encode_video_frames to apply the resize.
    # Peek at the first bag to get actual frame dimensions — skip the ffmpeg resize
    # filter entirely for cameras that already publish at the contract target size.
    compressed_image_specs = [
        spec for spec in specs
        if isinstance(spec, ObservationStreamSpec)
        and spec.is_image
        and spec.image_resize
        and spec.msg_type == COMPRESSED_IMAGE
    ]
    actual_sizes = (
        _peek_compressed_image_sizes(bag_dirs[0], compressed_image_specs)
        if compressed_image_specs
        else {}
    )
    per_key_kwargs: dict[str, dict[str, Any]] = {
        spec.key: {'target_size': tuple(spec.image_resize)}
        for spec in compressed_image_specs
        if actual_sizes.get(spec.key) != tuple(spec.image_resize)
    }
    extra_encode_kwargs: dict[str, Any] = {}
    if bitrate is not None:
        extra_encode_kwargs['bitrate'] = bitrate
    if extra_encode_kwargs:
        for spec in specs:
            if isinstance(spec, ObservationStreamSpec) and spec.is_image:
                per_key_kwargs.setdefault(spec.key, {}).update(extra_encode_kwargs)
    if per_key_kwargs:
        lerobot_dataset.writer.per_key_encoding_kwargs = per_key_kwargs

    start_time = time.time()
    num_episodes = len(bag_dirs)
    successful = 0
    failed: list[tuple[Path, str]] = []

    try:
        for episode_index, bag_dir in enumerate(bag_dirs):
            elapsed_time = time.time() - start_time
            d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)

            logging.info(
                f'{episode_index} / {num_episodes} episodes processed '
                f'(after {d} days, {h} hours, {m} minutes, {s:.3f} seconds)'
            )

            try:
                if prompt is not None:
                    episode_prompt = prompt
                else:
                    episode_prompt = _read_prompt(_read_bag_metadata(bag_dir))
                    if episode_prompt is None:
                        raise RuntimeError(
                            f"No prompt defined for {bag_dir.name}. "
                            f"Add prompt to custom_data in metadata.yaml or pass --prompt."
                        )

                if video_encoder is not None:
                    rows: list[dict] = []
                    h264_gates: dict[str, StartGate] = {}
                    for frame, _fmts in _stream_frames_from_bag(
                        bag_dir,
                        specs,
                        prompt=episode_prompt,
                        plan_video=True,
                        max_stamp_backward_jump_s=max_stamp_backward_jump_s,
                        h264_gates=h264_gates,
                    ):
                        rows.append(frame)
                    # An H.264 camera starts on an IDR: start on the first row by which every one
                    # has received an IDR; the others start from their previous IDR, behind.
                    video_keys_h264 = [
                        s.key for s in image_specs_for_plan if s.msg_type == COMPRESSED_VIDEO
                    ]
                    if rows and video_keys_h264:
                        missing = [k for k in video_keys_h264 if k not in h264_gates]
                        if missing:
                            raise RuntimeError(f'{missing}: topic not in {bag_dir.name}')
                        first_row = max(
                            h264_gates[k].first_start_row([r[k] for r in rows])
                            for k in video_keys_h264
                        )
                        if first_row:
                            logging.info(
                                '  starting on row %d, the first by which every H.264 camera '
                                'has received an IDR', first_row
                            )
                            del rows[:first_row]
                    frame_count = len(rows)
                    if frame_count == 0:
                        raise RuntimeError(f'No frames produced for {bag_dir.name}')

                    plans = []
                    for spec in image_specs_for_plan:
                        times = [r.get(spec.key) for r in rows]
                        # A row with no message holds None, or zeros when a stamp ran ahead.
                        empty = sum(not isinstance(t, (int, np.integer)) for t in times)
                        if empty:
                            raise RuntimeError(
                                f'{spec.key}: {empty} of {frame_count} '
                                f'rows have no message in {bag_dir.name}'
                            )
                        height, width = (
                            int(spec.image_resize[0]), int(spec.image_resize[1])
                        )
                        plans.append(
                            BagCameraPlan(
                                video_key=spec.key,
                                topic=spec.topic,
                                log_times=np.asarray(times, dtype=np.int64),
                                width=width,
                                height=height,
                                passthrough=spec.msg_type == COMPRESSED_VIDEO,
                            )
                        )

                    mcap_path = _first_storage_file(bag_dir)
                    video_encoder.set_episode_plan(mcap_path, plans, frame_count)
                    video_encoder.start_episode(
                        [p.video_key for p in plans], Path(lerobot_dataset.root)
                    )
                    logging.info('  video encode started: %s', video_encoder.last_plan_summary)

                    video_keys = {p.video_key for p in plans}
                    for row in rows:
                        lerobot_dataset.add_frame_bytes(
                            {k: v for k, v in row.items() if k not in video_keys}
                        )
                    rows.clear()
                else:
                    frame_count = 0
                    for frame, key_formats in _stream_frames_from_bag(
                        bag_dir,
                        specs,
                        prompt=episode_prompt,
                        max_stamp_backward_jump_s=max_stamp_backward_jump_s,
                    ):
                        lerobot_dataset.add_frame_bytes(frame, format=key_formats or None)
                        frame_count += 1

                lerobot_dataset.save_episode()
                successful += 1
                logging.info('  -> %d frames from %s', frame_count, bag_dir.name)

            except Exception as e:
                failed.append((bag_dir, str(e)))
                logging.error('  -> FAILED %s: %s', bag_dir.name, e)
                continue

        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)
        logging.info(
            f'\nCompleted: {successful}/{num_episodes} episodes '
            f'({len(failed)} failed) in {d}d {h}h {m}m {s:.1f}s'
        )

        if failed:
            logging.warning('Failed bags:')
            for bag_dir, error in failed:
                logging.warning('  - %s: %s', bag_dir.name, error)

        if successful == 0:
            raise RuntimeError(f'All {num_episodes} bags failed to convert')

        lerobot_dataset.finalize()
    finally:  # Ensure image writer is stopped even if an exception occurs
        writer = getattr(lerobot_dataset, 'writer', None)
        if writer is not None:
            writer.stop_image_writer()
        elif hasattr(lerobot_dataset, 'stop_image_writer'):
            lerobot_dataset.stop_image_writer()

    if push_to_hub:
        lerobot_dataset.push_to_hub(
            tags=['rosetta', 'rosbag'],
            private=False,
        )


# ---------- CLI ----------


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

    parser = argparse.ArgumentParser(
        description="Port ROS2 bags to LeRobot dataset"
    )

    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="Directory containing bag subdirectories"
    )
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help="HuggingFace repository ID (e.g., my_org/my_dataset). Defaults to raw-dir name."
    )
    parser.add_argument(
        "--contract", type=Path, required=True,
        help="Rosetta contract YAML path"
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Parent directory for datasets. Dataset saved to root/repo-id. (default: ~/.cache/huggingface/lerobot)"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Upload to HuggingFace Hub after porting"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Prompt for all episodes. If omitted, reads the prompt from each bag's metadata.yaml custom_data."
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Total number of shards for parallel processing"
    )
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="Index of this shard (0 to num-shards-1)"
    )
    parser.add_argument(
        "--vcodec", type=str, default="libsvtav1",
        choices=sorted(VALID_VIDEO_CODECS),
        help=(
            "Video codec for encoding (default: libsvtav1). 'h264' for faster software "
            "encoding, 'h264_nvmpi' on Jetson for hardware encoding."
        )
    )
    parser.add_argument(
        "--bitrate", type=str, default=None,
        help=(
            "Target bitrate for hardware encoders, FFmpeg style (e.g. '8M'). Used by "
            "h264_nvmpi / hevc_nvmpi, which have no CRF mode. Ignored by software codecs."
        )
    )
    parser.add_argument(
        "--pix-fmt", type=str, default=None,
        help="Pixel format (default: yuv420p)."
    )
    parser.add_argument(
        "--g", type=int, default=None,
        help="GOP size / keyframe interval (default: 2)."
    )
    parser.add_argument(
        "--crf", type=int, default=None,
        help="Constant rate factor / quality (default: 30)."
    )
    parser.add_argument(
        "--fast-decode", type=int, default=None,
        help="Fast-decode tuning flag (default: 0, codec-dependent)."
    )
    parser.add_argument(
        "--no-fast-video", action="store_true",
        help=(
            "Disable the planned MCAP->MP4 encoder and fall back to writing intermediate "
            "JPEGs for encode_video_frames. Slower; kept as an escape hatch."
        )
    )
    parser.add_argument(
        "--image-writer-threads", type=int, default=8,
        help="Number of image-writer threads for parallel frame writes (default: 8). Set to 0 to disable."
    )
    parser.add_argument(
        "--max-stamp-backward-jump-s", type=float, default=0.0,
        help=(
            "Largest tolerated backward stamp jump on a header stamped stream, in seconds "
            "(default: 0, any regression fails the bag). Negative disables the check."
        )
    )

    args = parser.parse_args()

    repo_id = args.repo_id or args.raw_dir.name

    encoding_kwargs = {}
    if args.vcodec is not None:
        encoding_kwargs["vcodec"] = args.vcodec
    if args.pix_fmt is not None:
        encoding_kwargs["pix_fmt"] = args.pix_fmt
    if args.g is not None:
        encoding_kwargs["g"] = args.g
    if args.crf is not None:
        encoding_kwargs["crf"] = args.crf
    if args.fast_decode is not None:
        encoding_kwargs["fast_decode"] = args.fast_decode
    if args.bitrate is not None:
        encoding_kwargs["bitrate"] = args.bitrate

    try:
        port_bags(
            raw_dir=args.raw_dir,
            repo_id=repo_id,
            contract_path=args.contract,
            root=args.root,
            prompt=args.prompt,
            push_to_hub=args.push_to_hub,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            encoding_kwargs=encoding_kwargs or None,
            image_writer_threads=args.image_writer_threads,
            fast_video=not args.no_fast_video,
            max_stamp_backward_jump_s=args.max_stamp_backward_jump_s,
        )
    except KeyboardInterrupt:
        logging.info('\nInterrupted by user')
    except Exception as e:
        logging.error('Error: %s', e)
        raise


if __name__ == '__main__':
    main()
