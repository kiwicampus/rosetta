"""An H.264 camera track: its decodable start and its episode MP4, copied without transcoding.
A P frame only decodes after the one before it, so repeated rows become skip frames and no
packet a later one references is ever dropped; only IDRs are decoded, for statistics."""

from __future__ import annotations

import bisect
import contextlib
import io
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator, List, Optional, Sequence, Set, Tuple

import av
import numpy as np

from .common import h264

MSG_TYPE = "foxglove_msgs/msg/CompressedVideo"

_REPEAT = -1


class StartGate:
    """Warm-up of one H.264 topic during resampling: messages count from its first IDR."""

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._keyframe_times: List[int] = []

    def accept(self, log_time: int, payload: bytes) -> bool:
        """Whether this message may be buffered, i.e. an IDR has arrived up to it."""
        if h264.is_keyframe(payload):
            self._keyframe_times.append(int(log_time))
        return bool(self._keyframe_times)

    def first_start_row(self, row_log_times: Sequence[int]) -> int:
        """First row whose message is an IDR, or arrived in the same tick as one.

        A row without a message log time (None or zeros) is never a start."""
        times = sorted(self._keyframe_times)
        prev: Optional[int] = None
        for r, held in enumerate(row_log_times):
            if not isinstance(held, (int, np.integer)):
                prev = None
                continue
            low = held - 1 if prev is None else prev
            if bisect.bisect_right(times, held) > bisect.bisect_right(times, low):
                return r
            prev = held
        raise ValueError(f"{self._name}: no IDR arrives after the episode starts")


@dataclass
class EpisodeVideo:
    """What write_episode_video produced; stats_images are downsampled CHW uint8 frames."""

    rows: int
    packets_copied: int
    skip_frames: int
    max_lag: int
    stats_images: List[np.ndarray] = field(default_factory=list)


def write_episode_video(
    mcap_path: str,
    topic: str,
    row_log_times: Sequence[int],
    out_path: str,
    fps: int,
    size: Tuple[int, int],
    stats_rows: Sequence[int] = (),
) -> EpisodeVideo:
    """Write one MP4 sample per row from hold's message per row, holding one packet at a time.
    size is (height, width) and must match the stream; statistics decode only the IDR nearest each
    stats row. Raises ValueError or h264.H264Error on input it cannot copy."""
    log_times, keyframes, fmt = _scan(mcap_path, topic)
    if fmt.lower() != "h264":
        raise ValueError(f"format {fmt!r} on {topic}; only h264 is copied through")
    index_of = {t: i for i, t in enumerate(log_times)}
    shared = {t for t, n in Counter(log_times).items() if n > 1}
    hold = []
    for t in row_log_times:
        if int(t) not in index_of or int(t) in shared:
            raise ValueError(f"log time {int(t)} does not name exactly one message on {topic}")
        hold.append(index_of[int(t)])

    plan = _plan_packet_rows(hold, keyframes)
    idr_rows = [r for r, p in enumerate(plan) if p != _REPEAT and keyframes[p]]
    stats_at = _nearest_rows(idr_rows, stats_rows)
    video = EpisodeVideo(
        rows=len(plan),
        packets_copied=len({p for p in plan if p != _REPEAT}),
        skip_frames=plan.count(_REPEAT),
        max_lag=max((h - p for h, p in zip(hold, plan) if p != _REPEAT), default=0),
    )
    reader = contextlib.closing(_iter_payloads(mcap_path, topic))
    with _Mp4Writer(out_path, fps) as writer, reader as payloads:
        state = h264.StreamState()
        index, payload = -1, b""
        for row, wanted in enumerate(plan):
            if wanted == _REPEAT:
                writer.write(state.skip_frame(), keyframe=False)
                continue
            while index < wanted:
                index, payload = next(payloads)
            writer.write(payload, keyframe=state.parse_access_unit(payload).is_idr)
            if row in stats_at:
                video.stats_images.append(_stats_image(payload, size))
    return video


def _plan_packet_rows(hold: Sequence[int], keyframes: Sequence[bool]) -> List[int]:
    """Per row, the packet to write or _REPEAT. Advances one packet per row, falling behind where
    hold skipped one, and drops packets only by jumping forward onto an IDR."""
    if not len(hold):
        return []
    first = int(hold[0])
    start = next((k for k in range(first, -1, -1) if keyframes[k]), None)
    if start is None:
        raise ValueError(f"row 0 resamples message {first}, and no IDR precedes it")
    out = [start]
    last = start
    for previous, target in zip(hold, hold[1:]):
        if target < previous:
            raise ValueError(f"hold plan goes backwards: message {target} after {previous}")
        if target == last:
            out.append(_REPEAT)
            continue
        idr = next((k for k in range(target, last + 1, -1) if keyframes[k]), None)
        last = idr if idr is not None else last + 1
        out.append(last)
    return out


def _nearest_rows(candidates: Sequence[int], wanted: Sequence[int]) -> Set[int]:
    """For each wanted row, the nearest candidate row (candidates sorted)."""
    chosen: Set[int] = set()
    for w in wanted:
        k = bisect.bisect_left(candidates, w)
        near = candidates[max(0, k - 1) : k + 1]
        if near:
            chosen.add(min(near, key=lambda c: abs(c - w)))
    return chosen


def _messages(mcap_path: str, topic: str) -> Iterator[Tuple[int, object]]:
    from mcap.reader import make_reader
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    msg_type = get_message(MSG_TYPE)
    with open(mcap_path, "rb") as fh:
        for _, _, message in make_reader(fh).iter_messages(topics=[topic]):
            yield int(message.log_time), deserialize_message(message.data, msg_type)


def _scan(mcap_path: str, topic: str) -> Tuple[List[int], List[bool], str]:
    """First pass: every message's log time and IDR flag, and the stream format."""
    log_times: List[int] = []
    keyframes: List[bool] = []
    fmt = ""
    for log_time, msg in _messages(mcap_path, topic):
        fmt = fmt or str(msg.format)
        log_times.append(log_time)
        keyframes.append(h264.is_keyframe(bytes(msg.data)))
    return log_times, keyframes, fmt


def _iter_payloads(mcap_path: str, topic: str) -> Iterator[Tuple[int, bytes]]:
    """Second pass: (message index, Annex B payload), one at a time."""
    for index, (_, msg) in enumerate(_messages(mcap_path, topic)):
        yield index, bytes(msg.data)


def _stats_image(idr: bytes, size: Tuple[int, int]) -> np.ndarray:
    """Decode one IDR on its own and downsample it as lerobot's stats sampling does."""
    from lerobot.datasets.compute_stats import auto_downsample_height_width

    codec = av.CodecContext.create("h264", "r")
    frames = list(codec.decode(av.Packet(idr))) + list(codec.decode(None))
    if len(frames) != 1:
        raise ValueError(f"an IDR access unit decoded to {len(frames)} frames")
    rgb = frames[0].to_ndarray(format="rgb24")
    if rgb.shape[:2] != tuple(size):
        raise ValueError(
            f"stream is {rgb.shape[1]}x{rgb.shape[0]} but the contract asks for "
            f"{size[1]}x{size[0]}; passthrough cannot resize"
        )
    return np.ascontiguousarray(auto_downsample_height_width(rgb.transpose(2, 0, 1)))


def _annex_b(nals) -> bytes:
    return b"".join(b"\x00\x00\x00\x01" + nal for nal in nals)


class _Mp4Writer:
    """MP4 muxer templated on the first packet, so avcC holds the stream's own SPS and PPS."""

    def __init__(self, path: str, fps: int) -> None:
        self._path = path
        self._time_base = Fraction(1, fps)
        self._container: Optional[av.container.OutputContainer] = None
        self._stream = None
        self._row = 0

    def __enter__(self) -> "_Mp4Writer":
        return self

    def __exit__(self, *exc) -> None:
        if self._container is not None:
            self._container.close()

    def write(self, data: bytes, keyframe: bool) -> None:
        if self._container is None:
            if not keyframe:
                raise ValueError("the first packet of an episode must be an IDR")
            # A template stream never opens an encoder. Its avcC comes from this IDR's SPS and
            # PPS, set explicitly so trailing zero bytes after a NAL unit stay out of it.
            nals = list(h264.iter_nal_units(data))
            with av.open(io.BytesIO(_annex_b(nals)), format="h264") as source:
                self._container = av.open(self._path, "w")
                self._stream = self._container.add_stream_from_template(source.streams.video[0])
            self._stream.codec_context.extradata = _annex_b(
                n for n in nals if h264.nal_type(n) in (h264.NAL_SPS, h264.NAL_PPS)
            )
        packet = av.Packet(data)
        packet.stream = self._stream
        packet.pts = packet.dts = self._row
        packet.duration = 1
        packet.time_base = self._time_base
        packet.is_keyframe = keyframe
        self._container.mux(packet)
        self._row += 1
