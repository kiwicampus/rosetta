#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge: foxglove_msgs/CompressedVideo (e.g. H.264 chunks) -> sensor_msgs/Image (rgb8).

Decode path follows rosbag_curator/tooling/decoder.py (PyAV packet decode, optional max
dimension resize, dummy RGB on any failure). Subscribes to a configurable input topic
and republishes on ``/camera/color/image_raw`` by default for contracts that use
sensor_msgs/Image (e.g. policy_bridge with turtlebot.yaml-style topics).

Optional NVDEC (``h264_cuvid``, etc.) via PyAV when ``use_hardware_decode`` is true,
with automatic fallback to software decode. Shallow subscription QoS (default depth 2)
and optional stale-packet skipping reduce backlog latency.

Requires: PyAV (``av``), OpenCV (``cv2``), ``foxglove_msgs``.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import av
import cv2
import numpy as np
import rclpy
from foxglove_msgs.msg import CompressedVideo
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# Logical format string (from CompressedVideo.format) -> PyAV hardware decoder name
_DEFAULT_HW_CODEC: dict[str, str] = {
    "h264": "h264_cuvid",
    "avc": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "h265": "hevc_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
}


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _reliability_from_param(value: str) -> ReliabilityPolicy:
    s = (value or "best_effort").strip().lower()
    if s in ("reliable", "reliability_reliable", "rel"):
        return ReliabilityPolicy.RELIABLE
    return ReliabilityPolicy.BEST_EFFORT


def _create_dummy_rgb(last_h: int, last_w: int) -> Tuple[np.ndarray, int, int]:
    if not last_h or not last_w:
        h, w = 480, 640
    else:
        h, w = last_h, last_w
    return np.zeros((h, w, 3), dtype=np.uint8), h, w


def _packet_from_msg_data(data) -> av.packet.Packet:
    if isinstance(data, (bytes, bytearray)):
        return av.packet.Packet(data)
    return av.packet.Packet(memoryview(data))


def _decode_packet_rgb(
    codec_ctx: av.codec.context.CodecContext,
    packet: av.packet.Packet,
    last_hw: Tuple[int, int],
    max_dimension: int,
) -> Tuple[np.ndarray, int, int, bool]:
    """
    Decode one PyAV packet; returns (rgb uint8 HWC, new_h, new_w, decode_ok).
    On failure returns dummy, last_hw (or defaults), and decode_ok=False.
    """
    last_h, last_w = last_hw
    try:
        frames = codec_ctx.decode(packet)
    except Exception:
        img, h, w = _create_dummy_rgb(last_h, last_w)
        return img, h, w, False

    if not frames:
        img, h, w = _create_dummy_rgb(last_h, last_w)
        return img, h, w, False

    try:
        frame = frames[0]
        img_rgb = frame.to_ndarray(format="rgb24")
    except Exception:
        img, h, w = _create_dummy_rgb(last_h, last_w)
        return img, h, w, False

    h, w = img_rgb.shape[:2]
    if max_dimension > 0 and (h > max_dimension or w > max_dimension):
        if h > w:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        else:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        last_h, last_w = new_h, new_w
    else:
        last_h, last_w = h, w

    return img_rgb, last_h, last_w, True


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class FoxgloveImageBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("foxglove_image_bridge")

        self.declare_parameter("input_topic", "/camera/color/foxglove")
        self.declare_parameter("output_topic", "/camera/color/image_raw")
        self.declare_parameter("frame_id", "")
        self.declare_parameter("max_decode_dimension", 640)
        self.declare_parameter("qos_depth", 2)
        self.declare_parameter("publisher_qos_depth", 10)
        self.declare_parameter("use_hardware_decode", True)
        self.declare_parameter("hw_codec_name", "")
        self.declare_parameter("hw_device", "")
        self.declare_parameter("skip_if_older_than_ms", 800)
        self.declare_parameter(
            "stale_lag_clock_trust_max_ms",
            120_000.0,
        )
        self.declare_parameter("metrics_period_s", 5.0)
        self.declare_parameter("metrics_sample_cap", 500)
        self.declare_parameter("subscription_reliability", "best_effort")
        self.declare_parameter("publisher_reliability", "best_effort")

        in_topic = str(self.get_parameter("input_topic").value)
        out_topic = str(self.get_parameter("output_topic").value)
        self._frame_id_override = str(self.get_parameter("frame_id").value or "")
        self._max_dim = int(self.get_parameter("max_decode_dimension").value)
        sub_depth = max(1, int(self.get_parameter("qos_depth").value))
        pub_depth = max(1, int(self.get_parameter("publisher_qos_depth").value))
        self._use_hw = _as_bool(self.get_parameter("use_hardware_decode").value)
        self._hw_codec_override = str(self.get_parameter("hw_codec_name").value or "").strip()
        self._hw_device = str(self.get_parameter("hw_device").value or "").strip()
        self._skip_stale_ms = int(self.get_parameter("skip_if_older_than_ms").value)
        self._stale_lag_trust_max_ms = float(
            self.get_parameter("stale_lag_clock_trust_max_ms").value
        )
        self._metrics_period = float(self.get_parameter("metrics_period_s").value)
        self._metrics_cap = max(10, int(self.get_parameter("metrics_sample_cap").value))

        self._lock = threading.Lock()
        self._codec_ctx: Optional[av.codec.context.CodecContext] = None
        self._logical_format: Optional[str] = None
        self._decoder_label: str = ""
        self._hw_disabled_for_format: bool = False
        self._last_hw: Tuple[int, int] = (0, 0)

        # Metrics (per window)
        self._frames_in = 0
        self._frames_out = 0
        self._frames_dropped = 0
        self._decode_ms_samples: Deque[float] = deque(maxlen=self._metrics_cap)
        self._e2e_ms_samples: Deque[float] = deque(maxlen=self._metrics_cap)
        self._drops_since_last_metrics = 0
        self._last_metrics_mono = time.monotonic()
        self._idle_metrics_windows = 0
        self._in_topic = in_topic
        self._clock_mismatch_warned = False

        sub_rel_str = str(self.get_parameter("subscription_reliability").value)
        pub_rel_str = str(self.get_parameter("publisher_reliability").value)
        sub_rel = _reliability_from_param(sub_rel_str)
        pub_rel = _reliability_from_param(pub_rel_str)
        sub_qos = QoSProfile(
            depth=sub_depth,
            reliability=sub_rel,
            history=HistoryPolicy.KEEP_LAST,
        )
        pub_qos = QoSProfile(
            depth=pub_depth,
            reliability=pub_rel,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._pub = self.create_publisher(Image, out_topic, pub_qos)
        self.create_subscription(CompressedVideo, in_topic, self._on_compressed, sub_qos)
        self.get_logger().info(
            f"FoxgloveImageBridge: {in_topic} (CompressedVideo) -> {out_topic} (Image rgb8) "
            f"(sub: depth={sub_depth} rel={sub_rel_str}, "
            f"pub: depth={pub_depth} rel={pub_rel_str}, use_hw={self._use_hw})"
        )

        if self._metrics_period > 0.0:
            self.create_timer(self._metrics_period, self._emit_metrics)

    def _logical_format_name(self, format_name: str) -> str:
        return (format_name or "h264").strip().lower() or "h264"

    def _create_codec_ctx(self, codec_id: str) -> av.codec.context.CodecContext:
        ctx = av.codec.CodecContext.create(codec_id, "r")
        if self._hw_device and codec_id.endswith("_cuvid"):
            dev = self._hw_device
            if dev.lower().startswith("cuda:"):
                dev = dev.split(":", 1)[1]
            try:
                opts = dict(getattr(ctx, "options", None) or {})
                opts["hwaccel_device"] = dev
                ctx.options = opts
            except (TypeError, AttributeError, ValueError):
                pass
        return ctx

    def _ensure_codec(self, format_name: str) -> None:
        logical = self._logical_format_name(format_name)
        if self._codec_ctx is not None and self._logical_format == logical:
            return

        old_ctx = self._codec_ctx
        self._codec_ctx = None
        if old_ctx is not None:
            try:
                close = getattr(old_ctx, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

        self._logical_format = logical
        self._decoder_label = ""
        self._hw_disabled_for_format = False

        tried_hw = False
        if self._use_hw:
            hw_name = self._hw_codec_override or _DEFAULT_HW_CODEC.get(logical)
            if hw_name:
                tried_hw = True
                try:
                    self._codec_ctx = self._create_codec_ctx(hw_name)
                    self._decoder_label = f"hw:{hw_name}"
                    self.get_logger().info(
                        f"PyAV decoder initialized ({self._decoder_label}) for logical format '{logical}'"
                    )
                    return
                except Exception as e:
                    self.get_logger().warning(
                        f"Hardware decoder '{hw_name}' failed ({e!r}); falling back to software '{logical}'"
                    )

        try:
            self._codec_ctx = self._create_codec_ctx(logical)
            self._decoder_label = f"sw:{logical}" + (" (hw requested, failed)" if tried_hw else "")
            self.get_logger().info(
                f"PyAV decoder initialized ({self._decoder_label}) for format '{logical}'"
            )
        except Exception as e:
            self._codec_ctx = None
            self._decoder_label = ""
            self.get_logger().error(f"Failed to create codec context for '{logical}': {e}")

    def _fallback_to_software_unlocked(self, logical: str, reason: str) -> None:
        if self._hw_disabled_for_format:
            return
        self._hw_disabled_for_format = True
        self.get_logger().warning(
            f"Disabling hardware decode for '{logical}' ({reason}); switching to software decoder"
        )
        ctx = self._codec_ctx
        self._codec_ctx = None
        self._decoder_label = ""
        if ctx is not None:
            try:
                close = getattr(ctx, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        try:
            self._codec_ctx = self._create_codec_ctx(logical)
            self._decoder_label = f"sw:{logical}"
            self.get_logger().info(f"PyAV software decoder active ({self._decoder_label})")
        except Exception as e:
            self._codec_ctx = None
            self._decoder_label = ""
            self.get_logger().error(f"Software fallback failed for '{logical}': {e}")

    def _on_compressed(self, msg: CompressedVideo) -> None:
        self._frames_in += 1
        fmt = getattr(msg, "format", None) or "h264"
        logical = self._logical_format_name(str(fmt))

        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = _stamp_to_ns(msg.timestamp)
        lag_ms = (now_ns - stamp_ns) / 1e6

        packet = _packet_from_msg_data(msg.data)
        is_kf = bool(getattr(packet, "is_keyframe", False))

        clocks_comparable = abs(lag_ms) <= self._stale_lag_trust_max_ms
        if (
            not self._clock_mismatch_warned
            and self._skip_stale_ms >= 0
            and not clocks_comparable
        ):
            self._clock_mismatch_warned = True
            self.get_logger().warning(
                f"Stamp vs node clock skew {lag_ms:.0f} ms exceeds "
                f"stale_lag_clock_trust_max_ms={self._stale_lag_trust_max_ms:.0f}; "
                "treating as wall/sim mismatch — stale drops disabled. "
                "When playing rosbags, set use_sim_time:=true on this node and publish /clock."
            )

        if (
            self._skip_stale_ms >= 0
            and clocks_comparable
            and lag_ms > float(self._skip_stale_ms)
            and not is_kf
        ):
            self._frames_dropped += 1
            self._drops_since_last_metrics += 1
            if self._drops_since_last_metrics >= 50:
                self.get_logger().warning(
                    f"Dropped {self._drops_since_last_metrics} stale non-keyframe packets "
                    f"(lag > {self._skip_stale_ms} ms; latest lag {lag_ms:.1f} ms)"
                )
                self._drops_since_last_metrics = 0
            return

        t_decode0 = time.perf_counter()
        with self._lock:
            self._ensure_codec(str(fmt))
            if self._codec_ctx is None:
                img, lh, lw = _create_dummy_rgb(self._last_hw[0], self._last_hw[1])
                self._last_hw = (lh, lw)
            else:
                assert self._codec_ctx is not None
                img, lh, lw, ok = _decode_packet_rgb(
                    self._codec_ctx,
                    packet,
                    self._last_hw,
                    self._max_dim,
                )
                if (
                    not ok
                    and self._use_hw
                    and not self._hw_disabled_for_format
                    and self._decoder_label.startswith("hw:")
                ):
                    self._fallback_to_software_unlocked(
                        logical, "hardware decode returned no frame"
                    )
                    if self._codec_ctx is not None:
                        packet2 = _packet_from_msg_data(msg.data)
                        img, lh, lw, ok = _decode_packet_rgb(
                            self._codec_ctx,
                            packet2,
                            self._last_hw,
                            self._max_dim,
                        )
                self._last_hw = (lh, lw)

        decode_ms = (time.perf_counter() - t_decode0) * 1000.0
        e2e_ms = (self.get_clock().now().nanoseconds - stamp_ns) / 1e6

        self._decode_ms_samples.append(decode_ms)
        self._e2e_ms_samples.append(e2e_ms)

        out = Image()
        out.header.stamp = msg.timestamp
        fid = self._frame_id_override or getattr(msg, "frame_id", "") or ""
        out.header.frame_id = fid
        out.height, out.width = int(img.shape[0]), int(img.shape[1])
        out.encoding = "rgb8"
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = img.tobytes()
        self._pub.publish(out)
        self._frames_out += 1

    def _emit_metrics(self) -> None:
        now_m = time.monotonic()
        dt = now_m - self._last_metrics_mono
        self._last_metrics_mono = now_m

        d_sorted = sorted(self._decode_ms_samples)
        e_sorted = sorted(self._e2e_ms_samples)
        d_avg = float(np.mean(d_sorted)) if d_sorted else float("nan")
        d_p50 = _percentile(d_sorted, 50.0)
        d_p95 = _percentile(d_sorted, 95.0)
        e_avg = float(np.mean(e_sorted)) if e_sorted else float("nan")
        e_p50 = _percentile(e_sorted, 50.0)
        e_p95 = _percentile(e_sorted, 95.0)

        self.get_logger().info(
            f"FoxgloveImageBridge metrics ({dt:.1f}s): "
            f"decoder={self._decoder_label or 'none'} "
            f"in={self._frames_in} out={self._frames_out} dropped={self._frames_dropped} "
            f"decode_ms avg={d_avg:.2f} p50={d_p50:.2f} p95={d_p95:.2f} "
            f"e2e_ms avg={e_avg:.2f} p50={e_p50:.2f} p95={e_p95:.2f}"
        )

        if self._frames_in == 0 and self._frames_out == 0:
            self._idle_metrics_windows += 1
            if self._idle_metrics_windows % 3 == 0:
                self.get_logger().warning(
                    f"No CompressedVideo on '{self._in_topic}' (in=0). "
                    "RViz will show nothing until this topic publishes. Check: "
                    "(1) ros2 bag play is running and topic names match the launch remaps; "
                    "(2) same ROS_DOMAIN_ID as the publisher; "
                    f"(3) QoS: run `ros2 topic info -v {self._in_topic}` and set "
                    "subscription_reliability / publisher_reliability to match the publisher "
                    "(rosbag often uses reliable; live sensors often best_effort)."
                )
        else:
            self._idle_metrics_windows = 0

        self._frames_in = 0
        self._frames_out = 0
        self._frames_dropped = 0
        self._decode_ms_samples.clear()
        self._e2e_ms_samples.clear()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = FoxgloveImageBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
