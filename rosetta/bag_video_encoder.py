#!/usr/bin/env python3
"""Plan-then-fetch video encoder for MCAP bags, mirroring the HDF5 fast path.

``port_bags`` otherwise writes every camera frame to disk as a JPEG so that
``encode_video_frames`` can read it back, and encodes one camera per process. The HDF5
converter avoids both because an HDF5 file is addressable; ``rosbag2_py.SequentialReader``
is not -- one cursor, forward only -- which is why the bag path looked stuck.

It is not: the ``mcap`` library exposes the file's own chunk and message index, so a reader
can filter by topic and seek by log time. That is enough to plan ``row -> log_time`` in one
cheap pass and then fetch frames in parallel.

Two properties of the plan keep the workers simple: log times never decrease within a
segment, so one seek plus a forward walk suffices; and ``hold`` resampling makes consecutive
rows share a log time, so a frame is decoded once and emitted as often as the plan repeats it.

Implements the ``streaming_encoder`` interface ``DatasetWriter`` expects, the same hook the
HDF5 encoder uses. H.264 cameras (``BagCameraPlan.passthrough``) are copied by
``h264_passthrough`` instead of being decoded and encoded.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
import shutil
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import av
import cv2
import numpy as np
from mcap.reader import make_reader

from . import h264_passthrough

logger = logging.getLogger(__name__)

_JPEG_SOI = b"\xff\xd8\xff"

_DEFAULT_TARGET_UNITS = 12

# Capped low: h264_nvmpi drops output packets once the aggregate rate frames are pushed at
# gets high enough, and NVENC is already saturated around 5 sessions. Swept on this robot.
_DEFAULT_MAX_CONCURRENT = 5

# Decode scales on threads instead of processes: it has no hardware limit, and more processes
# would raise the push rate that costs packets.
_DEFAULT_DECODE_THREADS = 2

# Below this a segment costs more in fork, encoder setup and concat than it saves. Matters on
# short episodes, where cost-proportional shares would otherwise cut a few hundred rows a dozen
# ways; long ones are unaffected.
_MIN_SEGMENT_ROWS = 500

# The drop is a load race, so a repeat usually clears it; three failures are not contention.
_ENCODE_ATTEMPTS = 3

_FRAGMENTED_MP4_OPTIONS = {"movflags": "frag_keyframe+empty_moov"}


class _DroppedPackets(RuntimeError):
    """The encoder returned fewer packets than it was fed. Retryable; see _encode_segment."""


@dataclass
class BagCameraPlan:
    """Everything a worker needs to encode one camera for one episode.

    Attributes:
        video_key: LeRobot feature key, e.g. ``observation.images.head_rgbd``.
        topic: MCAP topic carrying this camera's CompressedImage or CompressedVideo messages.
        log_times: Per output row, the log time of the message to use. Non-decreasing;
            repeats where ``hold`` resampling reuses a frame.
        width: Output width after the contract's resize.
        height: Output height after the contract's resize.
        passthrough: The topic carries H.264 that is copied into the MP4, not re-encoded.
    """

    video_key: str
    topic: str
    log_times: np.ndarray
    width: int
    height: int
    passthrough: bool = False

    @property
    def num_rows(self) -> int:
        return int(len(self.log_times))

    def cost(self) -> float:
        """Relative encode cost; decode and encode both scale with pixels."""
        return self.num_rows * self.width * self.height


@dataclass
class _Segment:
    video_key: str
    index: int
    start: int
    stop: int
    stats_rows: Tuple[int, ...] = field(default_factory=tuple)


def plan_segments(
    plans: Sequence[BagCameraPlan],
    target_units: int,
    stats_rows: Sequence[int] = (),
) -> List[_Segment]:
    """Split each camera into contiguous row ranges of roughly equal cost.

    One unit per camera idles the machine as soon as resolutions differ, since the wall clock
    is then set by the most expensive camera alone. ``stats_rows`` are routed to whichever
    segment contains them so the parent never re-reads a frame.
    """
    total_cost = sum(p.cost() for p in plans) or 1.0
    wanted = set(int(r) for r in stats_rows)
    segments: List[_Segment] = []

    for plan in plans:
        n = plan.num_rows
        share = max(1, round(target_units * plan.cost() / total_cost))
        # Never split into pieces so small that process overhead dominates.
        share = max(1, min(share, max(1, n // _MIN_SEGMENT_ROWS)))
        base, extra = divmod(n, share)
        start = 0
        for i in range(share):
            stop = start + base + (1 if i < extra else 0)
            if stop > start:
                rows = tuple(sorted(r for r in wanted if start <= r < stop))
                segments.append(_Segment(plan.video_key, i, start, stop, rows))
            start = stop

    # Most expensive first, so no worker ends up holding an oversized final segment.
    segments.sort(key=lambda s: s.stop - s.start, reverse=True)
    return segments


def _jpeg_from_payload(payload: bytes) -> Optional[bytes]:
    """Return the JPEG bytes embedded in a CDR-serialized CompressedImage."""
    offset = payload.find(_JPEG_SOI)
    return payload[offset:] if offset >= 0 else None


def _encode_segment(
    mcap_path: str,
    plan: BagCameraPlan,
    segment: _Segment,
    out_path: str,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    codec_options: Dict[str, str],
    decode_threads: int = 1,
) -> Tuple[str, int, List[np.ndarray]]:
    """Wrapper that keeps worker failures readable in the parent.

    A PyAV exception cannot survive the pool's pickling -- it surfaces as
    ``BrokenProcessPool`` two levels from the cause -- so it is re-raised flattened.
    """
    # The loss is load-dependent, not content-dependent, so re-encoding the same rows normally
    # succeeds. Only that failure retries; anything else is a defect and must surface at once.
    last: Optional[Exception] = None
    for attempt in range(_ENCODE_ATTEMPTS):
        try:
            return _encode_segment_impl(
                mcap_path, plan, segment, out_path, fps, vcodec, pix_fmt, codec_options,
                decode_threads,
            )
        except _DroppedPackets as exc:
            last = exc
            Path(out_path).unlink(missing_ok=True)
            logger.warning(
                "%s segment %d: %s; retrying (attempt %d/%d)",
                plan.video_key,
                segment.index,
                exc,
                attempt + 2,
                _ENCODE_ATTEMPTS,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            raise RuntimeError(
                f"{plan.video_key} segment {segment.index} "
                f"[rows {segment.start}:{segment.stop}, {plan.width}x{plan.height}]: "
                f"{type(exc).__name__}: {exc}"
            ) from None

    raise RuntimeError(
        f"{plan.video_key} segment {segment.index} "
        f"[rows {segment.start}:{segment.stop}, {plan.width}x{plan.height}]: "
        f"{vcodec} dropped packets on all {_ENCODE_ATTEMPTS} attempts: {last}"
    ) from None


def _encode_segment_impl(
    mcap_path: str,
    plan: BagCameraPlan,
    segment: _Segment,
    out_path: str,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    codec_options: Dict[str, str],
    decode_threads: int,
) -> Tuple[str, int, List[np.ndarray]]:
    """Encode rows ``[segment.start, segment.stop)`` of one camera. Runs in a worker process.

    Returns ``(out_path, frames_written, stats_images)``. ``stats_images`` holds the
    channel-first, spatially decimated RGB arrays for this segment's sampled rows, in the
    representation ``compute_stats.sample_images`` builds, so the parent can compute the same
    statistics without decoding anything again.
    """
    from lerobot.datasets.compute_stats import auto_downsample_height_width

    av.logging.set_level(av.logging.ERROR)

    wanted_times = [int(t) for t in plan.log_times[segment.start : segment.stop]]
    stats_wanted = set(segment.stats_rows)
    stats_images: List[np.ndarray] = []
    frames_written = 0
    # Packets that reached the muxer, counted rather than assumed equal to rows fed: a lost
    # frame would otherwise surface much later as a short video.
    packets_muxed = 0
    # Kept so a shortfall can name the missing rows, not just a count.
    returned_pts: List[int] = []

    # `hold` makes consecutive rows share a log time, so deduplicating gives one decode per
    # message, reused across the rows that share it. stats_repeat counts the sampled rows.
    unique_times: List[int] = []
    stats_repeat: List[int] = []
    for row_offset, want in enumerate(wanted_times):
        if not unique_times or unique_times[-1] != want:
            unique_times.append(want)
            stats_repeat.append(0)
        if segment.start + row_offset in stats_wanted:
            stats_repeat[-1] += 1

    def decode(payload: bytes, want_stats: bool):
        """Decode one message into an encoder-ready frame. Runs on a decode thread."""
        jpeg = _jpeg_from_payload(payload)
        if jpeg is None:
            raise ValueError(f"{plan.video_key}: no JPEG marker in payload")
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"{plan.video_key}: failed to decode JPEG")
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        if bgr.shape[1] != plan.width or bgr.shape[0] != plan.height:
            frame = frame.reformat(width=plan.width, height=plan.height, format=pix_fmt)
        elif frame.format.name != pix_fmt:
            frame = frame.reformat(format=pix_fmt)
        if not want_stats:
            return frame, None
        # cv2's decode, not libav's: libav applies limited-range math to full-range yuvj420p.
        # Source resolution, because auto_downsample_height_width decimates by an integer
        # factor of the width it is given -- resizing first would change the sampling grid.
        rgb = bgr[:, :, ::-1].transpose(2, 0, 1)
        return frame, np.ascontiguousarray(auto_downsample_height_width(rgb))

    with open(mcap_path, "rb") as fh, av.open(
        out_path, "w", options=_FRAGMENTED_MP4_OPTIONS
    ) as container:
        stream = container.add_stream(vcodec, rate=fps, options=codec_options)
        stream.pix_fmt = pix_fmt
        stream.width = plan.width
        stream.height = plan.height

        reader = make_reader(fh)
        # One seek, then forward only: log times within a segment never decrease.
        messages = reader.iter_messages(topics=[plan.topic], start_time=unique_times[0])

        # cv2.imdecode and swscale release the GIL, so decode genuinely overlaps the encoder
        # this thread drives -- which is how decode scales past the process count.
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=decode_threads, thread_name_prefix="bagdec"
        )
        # Bounded so decode cannot run far ahead: ~1.4 MB per queued frame at 1280x720.
        max_inflight = max(2, 2 * decode_threads)
        inflight: Deque[concurrent.futures.Future] = deque()
        submitted = 0

        def submit_next() -> None:
            """Advance the bag cursor to the next needed message and queue its decode."""
            nonlocal submitted
            target = unique_times[submitted]
            for _, _, message in messages:
                if message.log_time < target:
                    continue
                if message.log_time != target:
                    raise ValueError(
                        f"{plan.video_key}: expected a message at log_time {target} "
                        f"(segment {segment.index}) but the next one is {message.log_time}"
                    )
                inflight.append(pool.submit(decode, message.data, stats_repeat[submitted] > 0))
                submitted += 1
                return
            raise ValueError(
                f"{plan.video_key}: bag ended before log_time {target} "
                f"(segment {segment.index})"
            )

        try:
            while submitted < len(unique_times) and len(inflight) < max_inflight:
                submit_next()

            current_time: Optional[int] = None
            current_frame: Optional[av.VideoFrame] = None
            consumed = 0

            for want in wanted_times:
                if current_time != want:
                    current_frame, stats_rgb = inflight.popleft().result()
                    current_time = want
                    if submitted < len(unique_times):
                        submit_next()
                    if stats_rgb is not None:
                        stats_images.extend([stats_rgb] * stats_repeat[consumed])
                    consumed += 1

                # Explicit PTS per row: `hold` re-encodes the same frame object, and reusing
                # its timestamp makes the muxer reject it as non-monotonic dts.
                current_frame.pts = frames_written
                current_frame.time_base = Fraction(1, fps)
                packet = stream.encode(current_frame)
                if packet:
                    container.mux(packet)
                    packets_muxed += len(packet)
                    returned_pts.extend(int(pk.pts) for pk in packet if pk.pts is not None)
                frames_written += 1
        finally:
            for future in inflight:
                future.cancel()
            pool.shutdown(wait=True)

        # One drain is all there is: a second call raises EOFError, and repeated flushing does
        # not recover the lost packets -- hence the retry in _encode_segment.
        for packet in stream.encode(None):
            container.mux(packet)
            packets_muxed += 1
            if packet.pts is not None:
                returned_pts.append(int(packet.pts))

    if packets_muxed != frames_written:
        # Which rows were lost, not just how many: the position is what distinguishes causes.
        # Ticks per frame come from the packets, since deriving them from fps gets it wrong.
        ordered = sorted(returned_pts)
        steps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        ticks_per_frame = min(steps) if steps else 1
        arrived = set(pts // ticks_per_frame for pts in ordered)
        missing = sorted(set(range(frames_written)) - arrived)
        logger.warning(
            "%s segment %d: %s returned %d packets for %d frames; missing rows %s of 0..%d",
            plan.video_key, segment.index, vcodec, packets_muxed, frames_written,
            missing[:12], frames_written - 1,
        )
        raise _DroppedPackets(
            f"{plan.video_key} segment {segment.index}: fed {frames_written} frames but the "
            f"muxer received {packets_muxed} packets ({vcodec} dropped "
            f"{frames_written - packets_muxed})"
        )

    return out_path, frames_written, stats_images


def _remux_h264(
    mcap_path: str, plan: BagCameraPlan, segment: _Segment, out_path: str, fps: int
) -> Tuple[str, int, List[np.ndarray]]:
    """Copy one camera's H.264 packets into an MP4 along the plan. Runs in a worker process."""
    try:
        video = h264_passthrough.write_episode_video(
            mcap_path, plan.topic, plan.log_times, out_path, fps,
            (plan.height, plan.width), segment.stats_rows,
        )
    except Exception as exc:  # noqa: BLE001, flattened for the pool as in _encode_segment
        raise RuntimeError(f"{plan.video_key} passthrough: {type(exc).__name__}: {exc}") from None
    logger.info(
        "%s: copied %d packets into %d rows, %d skip frames, max lag %d frames",
        plan.video_key, video.packets_copied, video.rows, video.skip_frames, video.max_lag,
    )
    return out_path, video.rows, video.stats_images


def _count_packets(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for p in container.demux(stream) if p.dts is not None)


def _first_pts(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return next(p.pts for p in container.demux(stream) if p.pts is not None)


class PlannedBagVideoEncoder:
    """Encodes episode videos straight from an MCAP bag, in parallel, without intermediate files.

    Usage mirrors :class:`PlannedHdf5VideoEncoder`::

        enc.set_episode_plan(mcap_path, plans, num_rows)
        enc.start_episode([p.video_key for p in plans], dataset_root)
        ...                                  # caller adds non-video features as usual
        results = enc.finish_episode()       # {video_key: (path, stats)}
    """

    def __init__(
        self,
        fps: int,
        vcodec: str = "h264",
        pix_fmt: str = "yuv420p",
        codec_options: Optional[Dict[str, str]] = None,
        target_units: int = _DEFAULT_TARGET_UNITS,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        decode_threads: int = _DEFAULT_DECODE_THREADS,
    ) -> None:
        self._fps = int(fps)
        self._vcodec = vcodec
        self._pix_fmt = pix_fmt
        # Fragmented MP4 has no edit list, so B-frame reorder delay would offset every pts.
        self._codec_options = {**(codec_options or {}), "bf": "0"}
        self._target_units = int(target_units)
        self._max_concurrent = max(1, int(max_concurrent))
        self._decode_threads = max(1, int(decode_threads))

        self._mcap_path: Optional[str] = None
        self._plans: List[BagCameraPlan] = []
        self._num_rows = 0
        self._episode_active = False
        self._temp_dir: Optional[Path] = None
        self._pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._futures: Dict[concurrent.futures.Future, _Segment] = {}
        self.last_plan_summary = ""

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def set_episode_plan(
        self, mcap_path: Path | str, plans: Sequence[BagCameraPlan], num_rows: int
    ) -> None:
        """Register what the next episode should encode. Call before :meth:`start_episode`."""
        if self._episode_active:
            raise RuntimeError("cannot change the plan while an episode is active")
        for plan in plans:
            if plan.num_rows != num_rows:
                raise ValueError(
                    f"{plan.video_key}: plan has {plan.num_rows} rows, episode has {num_rows}"
                )
            if plan.num_rows and np.any(np.diff(plan.log_times) < 0):
                raise ValueError(
                    f"{plan.video_key}: log_times must be non-decreasing; the worker seeks "
                    "once and walks forward"
                )
        self._mcap_path = str(mcap_path)
        self._plans = list(plans)
        self._num_rows = int(num_rows)

    def start_episode(self, video_keys: List[str], temp_dir: Path) -> None:
        """Submit every segment to a process pool and return without waiting."""
        if self._episode_active:
            raise RuntimeError("episode already active")
        if self._mcap_path is None or not self._plans:
            raise RuntimeError("set_episode_plan must be called first")
        missing = set(video_keys) - {p.video_key for p in self._plans}
        if missing:
            raise ValueError(f"no plan for video keys: {sorted(missing)}")

        self._temp_dir = Path(tempfile.mkdtemp(dir=temp_dir, prefix="bagvid_"))
        from lerobot.datasets.compute_stats import sample_indices

        # lerobot's own sampling, so the stats match the non-streaming path row for row.
        stats_rows = sample_indices(self._num_rows) if self._num_rows >= 2 else []
        encoded = [p for p in self._plans if not p.passthrough]
        segments = plan_segments(encoded, self._target_units, stats_rows)
        # A copy is I/O bound and its packet plan spans the episode, so it is never split.
        segments += [
            _Segment(p.video_key, 0, 0, p.num_rows, tuple(stats_rows))
            for p in self._plans
            if p.passthrough
        ]
        by_key = {p.video_key: p for p in self._plans}

        # fork, not spawn: spawn re-imports __main__ and drags torch and lerobot into each worker.
        ctx = mp.get_context("fork")
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(self._max_concurrent, len(segments)), mp_context=ctx
        )
        self._futures = {}
        for segment in segments:
            out = self._temp_dir / f"{segment.video_key}_{segment.index:03d}.mp4"
            plan = by_key[segment.video_key]
            if plan.passthrough:
                future = self._pool.submit(
                    _remux_h264, self._mcap_path, plan, segment, str(out), self._fps
                )
                self._futures[future] = segment
                continue
            future = self._pool.submit(
                _encode_segment,
                self._mcap_path,
                by_key[segment.video_key],
                segment,
                str(out),
                self._fps,
                self._vcodec,
                self._pix_fmt,
                self._codec_options,
                self._decode_threads,
            )
            self._futures[future] = segment

        per_key: Dict[str, int] = {}
        for segment in segments:
            per_key[segment.video_key] = per_key.get(segment.video_key, 0) + 1
        self.last_plan_summary = (
            f"{len(segments)} segments, {len(self._futures)} submitted, "
            + ", ".join(f"{k.split('.')[-1]}x{v}" for k, v in sorted(per_key.items()))
        )
        self._episode_active = True

    def finish_episode(self) -> Dict[str, Tuple[Path, Optional[dict]]]:
        """Wait for every segment, concatenate per camera, and return paths plus statistics."""
        if not self._episode_active:
            raise RuntimeError("no active episode")
        from lerobot.datasets.video_utils import concatenate_video_files

        pieces: Dict[str, Dict[int, Path]] = {p.video_key: {} for p in self._plans}
        written: Dict[str, int] = {p.video_key: 0 for p in self._plans}
        # Keyed by segment index, not by completion order: get_feature_stats is a streaming
        # estimator, so the same samples in a different order shift mean and std.
        samples: Dict[str, Dict[int, List[np.ndarray]]] = {
            p.video_key: {} for p in self._plans
        }

        try:
            for future in concurrent.futures.as_completed(list(self._futures)):
                segment = self._futures[future]
                out_path, frames, stats_images = future.result()
                expected = segment.stop - segment.start
                if frames != expected:
                    raise RuntimeError(
                        f"{segment.video_key} segment {segment.index}: encoder was fed "
                        f"{expected} rows but wrote {frames}"
                    )
                pieces[segment.video_key][segment.index] = Path(out_path)
                written[segment.video_key] += frames
                samples[segment.video_key][segment.index] = stats_images
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

        results: Dict[str, Tuple[Path, Optional[dict]]] = {}
        for plan in self._plans:
            ordered = [pieces[plan.video_key][i] for i in sorted(pieces[plan.video_key])]
            if not ordered:
                raise RuntimeError(f"{plan.video_key}: no segments produced")
            # One directory per camera: DatasetWriter removes ep_path.parent after moving the
            # file, which on a shared parent would take the finals still pending.
            final_dir = self._temp_dir / f"final_{plan.video_key}"
            final_dir.mkdir(parents=True, exist_ok=True)
            final = final_dir / f"{plan.video_key}.mp4"
            # Always concatenate: the remux gives the last packet an explicit duration.
            concatenate_video_files(ordered, final, overwrite=True)

            stored = _count_packets(final)
            if stored != plan.num_rows:
                raise RuntimeError(
                    f"{plan.video_key}: encoded video holds {stored} frames but the episode "
                    f"has {plan.num_rows} rows"
                )
            start = _first_pts(final)
            if start != 0:
                raise RuntimeError(f"{plan.video_key}: encoded video starts at pts {start}, not 0")
            ordered_samples = [
                image
                for index in sorted(samples[plan.video_key])
                for image in samples[plan.video_key][index]
            ]
            results[plan.video_key] = (final, self._stats_from_samples(ordered_samples))

        self._episode_active = False
        return results

    @staticmethod
    def _stats_from_samples(images: List[np.ndarray]) -> Optional[dict]:
        """Per-channel statistics over the sampled CHW uint8 frames, in raw 0-255 units.

        Deliberately not normalized: ``DatasetWriter`` divides streaming-encoder stats by
        255 and reshapes them itself, so dividing here too makes every value 255x too
        small. Same contract as ``PlannedHdf5VideoEncoder._stats_from_samples``.
        """
        if len(images) < 2:
            return None
        from lerobot.datasets.compute_stats import get_feature_stats

        return get_feature_stats(np.stack(images), axis=(0, 2, 3), keepdims=True)

    def cancel_episode(self) -> None:
        if not self._episode_active:
            return
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self._cleanup_temp()
        self._episode_active = False

    def close(self) -> None:
        self.cancel_episode()
        self._cleanup_temp()

    def _cleanup_temp(self) -> None:
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
