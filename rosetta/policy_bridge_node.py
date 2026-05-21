#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolicyBridge: contract-true live policy inference.

"""

from __future__ import annotations

import csv
import json
import math
import os
import threading
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from std_srvs.srv import Trigger
from rosidl_runtime_py.utilities import get_message
from rcl_interfaces.msg import SetParametersResult
import torch


from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from rosetta.common.contract_utils import (
    load_contract,
    iter_specs,
    SpecView,
    feature_from_spec,
    zero_pad,
    qos_profile_from_dict,    contract_fingerprint,
    decode_value,
    StreamBuffer,
    observation_stored_footprint,
    stamp_from_header_ns,
    encode_value,
)

from rosetta_interfaces.action import RunPolicy

_ACTION_NAME = "run_policy"
_FEEDBACK_PERIOD_S = 0.5


@dataclass(slots=True)
class _SubState:
    spec: SpecView
    msg_type: Any
    buf: StreamBuffer
    stamp_src: str  # 'receive' or 'header'


@dataclass(slots=True)
class _RuntimeParams:
    use_chunks: bool
    actions_per_chunk: int
    chunk_size_threshold: float
    use_header_time: bool
    use_autocast: bool
    max_queue_actions: int = 512


def _device_from_param(requested: Optional[str] = None) -> torch.device:
    r = (requested or "auto").lower().strip()

    def mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if r == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Explicit CUDA (supports 'cuda' and 'cuda:N')
    if r.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(r)  # 'cuda' or 'cuda:N'

    # Explicit MPS (or 'metal' alias)
    if r in {"mps", "metal"}:
        if not mps_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    # Anything else: try to parse ('cpu', 'xpu', etc.), otherwise fallback
    try:
        return torch.device(r)
    except (TypeError, ValueError, RuntimeError):
        # Invalid device requested, fallback to CPU
        return torch.device("cpu")


class PolicyBridge(Node):
    """Contract-true live inference node with persistent timers and action control."""

    def __init__(self) -> None:
        super().__init__("policy_bridge")

        # ---------------- Parameters ----------------
        self.declare_parameter("contract_path", "")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("policy_device", "auto")
        self.declare_parameter("use_chunks", True)
        self.declare_parameter("actions_per_chunk", 25)
        self.declare_parameter("chunk_size_threshold", 0.5)
        self.declare_parameter("max_queue_actions", 512)
        self.declare_parameter("use_header_time", True)
        self.declare_parameter("use_autocast", False)
        self.declare_parameter("bridge_debug", False)
        # std_msgs/Float32MultiArray when use_chunks is false; empty string disables.
        self.declare_parameter("predicted_action_topic", "predicted_action")
        self.declare_parameter("debug_chunk_topic", "debug/predicted_chunk")
        self.declare_parameter("debug_exec_topic", "debug/executed_action")
        self.declare_parameter("debug_chunk_log_dir", "")
        # Log latest reference command next to executed actions in executed.csv (TwistStamped).
        self.declare_parameter(
            "reference_cmd_topic",
            "/motion_control/speed_controller/reference_cmd",
        )
        # Odometry pose sampled when each chunk is generated (for executed.csv anchors).
        self.declare_parameter("chunk_log_anchor_odom_topic", "/odometry/local")
        # nav_msgs/Path publisher for RViz visualization of predicted trajectory.
        self.declare_parameter("waypoint_path_topic", "predicted_waypoints")
        # GPS viz: Path in this frame. Twist viz: uses anchor odometry parent frame when available.
        # Waypoints viz (nav_msgs/Odometry actions or mode=waypoints): Path in waypoint_path_frame.
        self.declare_parameter("waypoint_path_frame", "map")
        self.declare_parameter("gps_ref_topic", "/gps/filtered")
        # auto = infer (NavSatFix→gps, Twist*→integrate v/w, Odometry→plot x,y rows); gps | twist | waypoints.
        self.declare_parameter("waypoint_path_mode", "auto")

        self._params = self._read_params()
        self._bridge_debug = bool(self.get_parameter("bridge_debug").value)
        self._debug_chunk_log_dir = str(
            self.get_parameter("debug_chunk_log_dir").value or ""
        ).strip()
        self.add_on_set_parameters_callback(self._on_params)

        # ---------------- Contract ----------------
        contract_path = str(self.get_parameter("contract_path").value or "")
        if not contract_path:
            raise RuntimeError("policy_bridge: 'contract_path' is required")
        self._contract = load_contract(Path(contract_path))

        print(f"Loaded contract from {contract_path} with {len(self._contract.observations or [])} observations and {len(self._contract.actions or [])} actions.")

        self._obs_qos_by_key: Dict[str, Optional[Dict[str, Any]]] = {
            o.key: o.qos for o in (self._contract.observations or [])
        }
        self._act_qos_by_key: Dict[str, Optional[Dict[str, Any]]] = {
            a.key: a.publish_qos for a in (self._contract.actions or [])
        }

        # ---------------- Policy load ----------------
        policy_path = str(self.get_parameter("policy_path").value or "")
        if not policy_path:
            raise RuntimeError("policy_bridge: 'policy_path' is required")

        # Check if policy_path is a Hugging Face repo ID (contains '/')
        is_hf_repo = '/' in policy_path and not os.path.exists(policy_path)

        print("Using policy path:", policy_path)
        cfg_type = ""  # Default value
        if is_hf_repo:
            # For Hugging Face repos, we'll let from_pretrained handle the download
            # and get the config type from the loaded policy
            self.get_logger().info(f"Detected Hugging Face repo: {policy_path}")
        else:
            # For local paths, try to read config.json
            cfg_json = os.path.join(policy_path, "config.json")
            try:
                if os.path.exists(cfg_json):
                    with open(cfg_json, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        cfg_type = str(cfg.get("type", "")).lower()
            except (OSError, json.JSONDecodeError, KeyError) as e:
                self.get_logger().warning(
                    f"Could not read policy config.json: {e!r}"
                )

        req = str(self.get_parameter("policy_device").value)
        self.device = _device_from_param(req)
        r = req.strip().lower()
        r = "mps" if r == "metal" else r
        if r not in {"auto", ""} and str(torch.device(r)) != str(self.device):
            self.get_logger().warning(f"policy_device='{req}' requested; using '{self.device}' instead.")
        self.get_logger().info(f"Using device: {self.device}")

        if is_hf_repo:
            # For Hugging Face repos, we need to load the policy first to get the config type
            # We'll use a temporary approach: try common policy types
            policy_types_to_try = ["act", "diffusion", "pi0", "pi05", "smolvla"]
            policy_loaded = False
            
            for policy_type in policy_types_to_try:
                try:
                    self.get_logger().info(f"Trying to load as {policy_type} policy...")
                    PolicyCls = get_policy_class(policy_type)
                    self.policy = PolicyCls.from_pretrained(policy_path)
                    self.policy.to(self.device)
                    self.policy.eval()
                    policy_loaded = True
                    cfg_type = policy_type
                    self.get_logger().info(f"Successfully loaded {policy_type} policy from {policy_path}")
                    break
                except Exception as e:
                    self.get_logger().debug(f"Failed to load as {policy_type}: {e}")
                    continue
            
            if not policy_loaded:
                raise RuntimeError(f"Could not load policy from {policy_path} with any known policy type")
        else:
            # For local paths, use the config type we read earlier
            if not cfg_type:
                raise RuntimeError(f"Could not determine policy type from {policy_path}")
            PolicyCls = get_policy_class(cfg_type)
            self.policy = PolicyCls.from_pretrained(policy_path)
            self.policy.to(self.device)
            self.policy.eval()

        # Load dataset stats from the policy artifact if present
        ds_stats = None
        for cand in ("dataset_stats.json", "stats.json", "meta/stats.json"):
            p = Path(policy_path) / cand
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        ds_stats = json.load(f)
                    self.get_logger().info(f"Loaded dataset stats from {p}")
                    break
                except Exception as e:
                    self.get_logger().warning(f"Failed to read {p}: {e!r}")

        # Validate contract fingerprint if available
        try:
            current_fp = contract_fingerprint(self._contract)
            self.get_logger().info(f"Contract fingerprint: {current_fp}")
            
            # Check if policy has a stored fingerprint
            policy_fp_path = Path(policy_path) / "contract_fingerprint.txt"
            if policy_fp_path.exists():
                with policy_fp_path.open("r") as f:
                    stored_fp = f.read().strip()
                if stored_fp != current_fp:
                    self.get_logger().warning(
                        f"Contract fingerprint mismatch! Policy: {stored_fp}, Current: {current_fp}"
                    )
                else:
                    self.get_logger().info("Contract fingerprint matches policy")
        except Exception as e:
            self.get_logger().warning(f"Contract fingerprint validation failed: {e!r}")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=policy_path,
            dataset_stats=ds_stats,  # <-- critical for parity
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)}},
            postprocessor_overrides={
                "device_processor": {"device": str(self.device)}},
        )

        # ---------------- Specs & rate ----------------
        self._specs: List[SpecView] = list(iter_specs(self._contract))
        self._obs_specs = [s for s in self._specs if not s.is_action]
        self._act_specs = [s for s in self._specs if s.is_action]
        #TODO: process _task_specs

        # Handle multiple action specs with the same key (consolidate them)
        self._action_specs_by_key: Dict[str, List[SpecView]] = {}
        for spec in self._act_specs:
            if spec.key not in self._action_specs_by_key:
                self._action_specs_by_key[spec.key] = []
            self._action_specs_by_key[spec.key].append(spec)
        
        # For now, we only support one action key (but multiple specs with that key)
        if len(self._action_specs_by_key) != 1:
            raise ValueError(
                f"This bridge expects exactly one action key in the contract, got {list(self._action_specs_by_key.keys())}. "
                f"Multiple action keys (e.g., action.arm, action.gripper) are not yet supported."
            )
        
        # Get the action key and specs
        self._action_key = list(self._action_specs_by_key.keys())[0]
        self._action_specs = self._action_specs_by_key[self._action_key]
        
        # For backward compatibility, keep the first spec as the "primary" one
        self._act_spec = self._action_specs[0]

        # Action type detection: use ros_type from contract publish.type
        self._action_is_gps = self._act_spec.ros_type == "sensor_msgs/msg/NavSatFix"
        self._action_is_twist = self._act_spec.ros_type == "geometry_msgs/msg/TwistStamped"
        # Local waypoint actions (e.g. pose.pose.position x,y in odom) — no unicycle integration.
        self._action_is_odom_xy = self._act_spec.ros_type == "nav_msgs/msg/Odometry"
        
        self._viz_is_gps = self._action_is_gps
        self._viz_is_twist = self._action_is_twist
        self._viz_is_waypoints_xy = self._action_is_odom_xy

        # GPS reference lock/origin (GPS viz mode only; updated by _gps_ref_cb)
        self._gps_ref_lock = threading.Lock()
        self._gps_ref: Optional[Tuple[float, float]] = None  # (lat0, lon0)
        # Last predicted chunk — kept so the path can be republished on every producer tick.
        self._last_chunk_np: Optional[np.ndarray] = None

        self.fps = int(self._contract.rate_hz)
        if self.fps <= 0:
            raise ValueError("Contract rate_hz must be >= 1")
        self.step_ns = int(round(1e9 / self.fps))
        self.step_sec = 1.0 / self.fps

        self._cbg = ReentrantCallbackGroup()
        self._obs_zero, self._subs, self._ros_sub_handles = {}, {}, []
        self._state_specs = [s for s in self._obs_specs if s.key == "observation.state"]

        for s in self._obs_specs:
            k, meta, _ = feature_from_spec(s, use_videos=False)
            
            # Create unique key for multiple observation.state specs (mirror bag_to_lerobot logic)
            if s.key == "observation.state" and len(self._state_specs) > 1:
                dict_key = f"{s.key}_{s.topic.replace('/', '_')}"
            else:
                dict_key = s.key
                
            self._obs_zero[dict_key] = zero_pad(meta)

            msg_cls = get_message(s.ros_type)
            sub = self.create_subscription(
                msg_cls, s.topic, lambda m, sv=s: self._obs_cb(m, sv),
                qos_profile_from_dict(self._obs_qos_by_key.get(s.key)),
                callback_group=self._cbg,
            )
            self._ros_sub_handles.append(sub)

            tol_ns = int(max(0, s.asof_tol_ms)) * 1_000_000
            self._subs[dict_key] = _SubState(
                spec=s,
                msg_type=msg_cls,
                buf=StreamBuffer(policy=s.resample_policy, step_ns=self.step_ns, tol_ns=tol_ns),
                stamp_src=s.stamp_src,
            )

        self.get_logger().info(
            f"Subscribed to {len(self._subs)} observation streams.")

        # Latest reference command (ground truth) for CSV logging — same layout as primary action spec names.
        self._ref_lock = threading.Lock()
        self._ref_vec: Optional[np.ndarray] = None
        ref_topic = str(self.get_parameter("reference_cmd_topic").value or "").strip()
        self._reference_cmd_topic = ref_topic if ref_topic else ""
        if self._reference_cmd_topic:
            ref_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(
                TwistStamped,
                self._reference_cmd_topic,
                self._reference_cmd_cb,
                ref_qos,
                callback_group=self._cbg,
            )
            self.get_logger().info(
                f"Subscribed to reference cmd for CSV: {self._reference_cmd_topic}"
            )

        self._odom_lock = threading.Lock()
        self._last_odom_xyyaw: Optional[Tuple[float, float, float]] = None
        # Parent frame of the odometry pose (e.g. "odom"); must match Path.header when integrating v/w in world XY.
        self._odom_frame_id: Optional[str] = None
        anchor_topic = str(
            self.get_parameter("chunk_log_anchor_odom_topic").value or ""
        ).strip()
        if anchor_topic and (self._debug_chunk_log_dir or self._viz_is_twist):
            odom_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(
                Odometry,
                anchor_topic,
                self._anchor_odom_cb,
                odom_qos,
                callback_group=self._cbg,
            )
            self.get_logger().info(
                f"Subscribed to odometry for chunk CSV anchors: {anchor_topic}"
            )

        # ---------------- Publishers ----------------
        self._act_pubs: Dict[str, Any] = {}
        for spec in self._action_specs:
            act_qos_dict = self._act_qos_by_key.get(spec.key)
            pub_qos = qos_profile_from_dict(act_qos_dict) or QoSProfile(depth=10)
            pub = self.create_publisher(
                get_message(spec.ros_type), spec.topic, pub_qos
            )
            self._act_pubs[spec.topic] = pub
            self.get_logger().info(f"Created publisher for {spec.topic} ({spec.ros_type})")
        
        # For backward compatibility, keep the primary publisher reference
        self._act_pub = self._act_pubs[self._act_spec.topic]

        pred_topic = str(
            self.get_parameter("predicted_action_topic").value or ""
        ).strip()
        self._predicted_action_pub: Optional[Any] = None
        if pred_topic:
            self._predicted_action_pub = self.create_publisher(
                Float32MultiArray,
                pred_topic,
                QoSProfile(depth=10),
            )
            self.get_logger().info(
                f"Predicted action debug publisher (use_chunks=false): '{pred_topic}' "
                "(std_msgs/Float32MultiArray)"
            )

        dbg_chunk_topic = str(
            self.get_parameter("debug_chunk_topic").value or ""
        ).strip()
        dbg_exec_topic = str(
            self.get_parameter("debug_exec_topic").value or ""
        ).strip()
        self._debug_chunk_pub: Optional[Any] = None
        self._debug_exec_pub: Optional[Any] = None
        if dbg_chunk_topic:
            self._debug_chunk_pub = self.create_publisher(
                Float32MultiArray,
                dbg_chunk_topic,
                QoSProfile(depth=10),
            )
            self.get_logger().info(
                f"Debug predicted-chunk publisher: '{dbg_chunk_topic}'"
            )
        if dbg_exec_topic:
            self._debug_exec_pub = self.create_publisher(
                Float32MultiArray,
                dbg_exec_topic,
                QoSProfile(depth=10),
            )
            self.get_logger().info(
                f"Debug executed-action publisher: '{dbg_exec_topic}'"
            )

        wp_path_topic = str(self.get_parameter("waypoint_path_topic").value or "").strip()
        self._waypoint_path_frame = str(self.get_parameter("waypoint_path_frame").value or "map")
        gps_ref_topic = str(self.get_parameter("gps_ref_topic").value or "").strip()

        self._waypoint_path_pub: Optional[Any] = None
        if wp_path_topic and (
            self._viz_is_gps or self._viz_is_twist or self._viz_is_waypoints_xy
        ):
            self._waypoint_path_pub = self.create_publisher(
                NavPath, wp_path_topic, QoSProfile(depth=10)
            )
            if self._viz_is_gps and gps_ref_topic:
                gps_qos = QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                )
                self.create_subscription(
                    NavSatFix, gps_ref_topic, self._gps_ref_cb,
                    gps_qos, callback_group=self._cbg,
                )
                self.get_logger().info(
                    f"GPS reference subscription: '{gps_ref_topic}'"
                )

        self._cancel_srv = self.create_service(
            Trigger, f"{_ACTION_NAME}/cancel", self._cancel_service_cb,
            callback_group=self._cbg
        )

        # ---------------- Action server ----------------
        self._active_handle: Optional[Any] = None
        self._running_event = threading.Event()
        self._stop_requested = threading.Event()
        self._done_event = threading.Event()
        self._finishing = threading.Event()
        self._prompt = ""
        self._pub_count = 0
        self._terminal: Optional[Tuple[str, str]] = None

        self._action_server = ActionServer(
            self,
            RunPolicy,
            _ACTION_NAME,
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cbg,
        )

        # ---------------- Deadline state ----------------
        self._max_duration_s = float(
            getattr(self._contract, "max_duration_s", 1000.0))
        self._deadline_active = False
        self._deadline_end_ns: Optional[int] = None


        # ---------------- Safety behavior ----------------
        self._safety_behavior = getattr(
            self._act_spec, "safety_behavior", "zeros"
        ).lower()
        if self._safety_behavior not in ("zeros", "hold"):
            self.get_logger().warning(
                f"Unknown safety_behavior '{self._safety_behavior}', defaulting to 'zeros'"
            )
            self._safety_behavior = "zeros"
        self.get_logger().info(
            "Safety: zeros on stop."
            if self._safety_behavior == "zeros"
            else "Safety: hold last action on stop."
        )

        # ---------------- Timing strategy ----------------
        raw_action_spec = self._contract.actions[0] if self._contract.actions else None
        strategy = getattr(raw_action_spec, "publish_strategy", None) or {}
        self._publish_mode = strategy.get("mode", "nearest")
        # Be a bit lenient on timing jitter (network + scheduling)
        self._publish_tolerance_ns = int(self.step_ns)

        # ---------------- Async producer/executor ----------------
        # Queue entries: (sched_time_ns, action_vec, chunk_id, step_index)
        self._queue: Deque[Tuple[int, np.ndarray, int, int]] = deque(
            maxlen=self._params.max_queue_actions
        )
        self._queue_lock = threading.Lock()
        self._last_action: Optional[np.ndarray] = None
        self._producer_buffer: List[Tuple[int, np.ndarray, int, int]] = []
        self._chunk_counter: int = 0
        self._chunk_anchors: Dict[int, Tuple[float, float, float]] = {}
        self._run_start_ns: Optional[int] = None
        self._exec_csv_header_written: bool = False

        self._cbg_timers = ReentrantCallbackGroup()
        self._producer_timer = self.create_timer(
            self.step_sec, self._producer_tick, callback_group=self._cbg_timers
        )
        self._executor_timer = self.create_timer(
            self.step_sec, self._executor_tick, callback_group=self._cbg_timers
        )
        self._feedback_timer = self.create_timer(
            _FEEDBACK_PERIOD_S, self._feedback_tick, callback_group=self._cbg_timers
        )
        self._deadline_timer = self.create_timer(
            0.2, self._deadline_tick, callback_group=self._cbg_timers
        )  # poll 5 Hz

        self.get_logger().info(
            f"PolicyBridge ready at {self.fps:.1f} Hz on device={self.device}."
        )
        if self._bridge_debug:
            self.get_logger().info(
                "[bridge_debug] enabled: extra logs for observation zero-padding causes "
                "(set bridge_debug:=false to silence)"
            )

    # ---------------- Parameter handling ----------------
    def _read_params(self) -> _RuntimeParams:
        return _RuntimeParams(
            use_chunks=bool(self.get_parameter("use_chunks").value),
            actions_per_chunk=self.get_parameter("actions_per_chunk").value, #TODO: this should come from the model/policy config. This nomenclature is confusing and not consistent with LeRobot. 
            chunk_size_threshold=float(
                self.get_parameter("chunk_size_threshold").value or 0.5 #TODO: also inconsistent with LeRobot naming.
            ),
            use_header_time=bool(self.get_parameter("use_header_time").value),
            use_autocast=bool(self.get_parameter("use_autocast").value),
            max_queue_actions=self.get_parameter("max_queue_actions").value,
        )

    def _on_params(self, _params: List[Parameter]) -> SetParametersResult:
        for p in _params:
            if p.name == "bridge_debug":
                self._bridge_debug = bool(p.value)
        new_params = self._read_params()
        if self._queue.maxlen != new_params.max_queue_actions:
            with self._queue_lock:
                self._queue = deque(self._queue, maxlen=new_params.max_queue_actions)
        self._params = new_params
        return SetParametersResult(successful=True)

    def _next_exec_tick_ns(self, now_ns: int) -> int:
        return ((now_ns + self.step_ns - 1) // self.step_ns) * self.step_ns

    @staticmethod
    def _float32_multiarray_1d(
        vec: np.ndarray, dim_label: str = "action_dim"
    ) -> Float32MultiArray:
        v = np.asarray(vec, dtype=np.float32).ravel()
        n = int(v.size)
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label=dim_label, size=n, stride=n)
        ]
        msg.layout.data_offset = 0
        msg.data = v.tolist()
        return msg

    def _run_log_dir(self) -> Optional[Path]:
        if not self._debug_chunk_log_dir or self._run_start_ns is None:
            return None
        p = Path(self._debug_chunk_log_dir) / str(self._run_start_ns)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _anchor_odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        fid = (msg.header.frame_id or "").strip()
        with self._odom_lock:
            self._last_odom_xyyaw = (float(p.x), float(p.y), float(yaw))
            if fid:
                self._odom_frame_id = fid

    def _gps_ref_cb(self, msg: NavSatFix) -> None:
        with self._gps_ref_lock:
            self._gps_ref = (float(msg.latitude), float(msg.longitude))

    def _publish_waypoint_path(self, chunk_np: np.ndarray) -> None:
        if self._waypoint_path_pub is None:
            return

        now = self.get_clock().now().to_msg()
        path_msg = NavPath()
        path_msg.header.stamp = now
        if self._viz_is_twist:
            # Integrated (x,y) are in the odometry parent frame (e.g. odom), not base_link or map.
            with self._odom_lock:
                odom_fid = self._odom_frame_id
            path_msg.header.frame_id = (
                (odom_fid or "").strip()
                or str(self._waypoint_path_frame or "").strip()
                or "odom"
            )
        else:
            path_msg.header.frame_id = self._waypoint_path_frame

        def _make_pose(x: float, y: float, yaw: float = 0.0) -> PoseStamped:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            return ps

        if self._viz_is_gps:
            # GPS mode: flat-earth lat/lon → local XY (east=+X, north=+Y)
            # TODO: support lookahead_n > 1 (chunk_np cols > 2, rows contain N lat/lon pairs)
            with self._gps_ref_lock:
                if self._gps_ref is None:
                    return
                lat0, lon0 = self._gps_ref
            METERS_PER_DEG = 111319.9
            cos_lat0 = math.cos(math.radians(lat0))
            path_msg.poses.append(_make_pose(0.0, 0.0))  # robot origin
            for i in range(chunk_np.shape[0]):
                lat = float(chunk_np[i, 0])  # col 0 = latitude
                lon = float(chunk_np[i, 1])  # col 1 = longitude
                x = (lon - lon0) * cos_lat0 * METERS_PER_DEG  # east
                y = (lat - lat0) * METERS_PER_DEG              # north
                path_msg.poses.append(_make_pose(x, y))

        elif self._viz_is_twist:
            # Twist mode: integrate unicycle kinematics from current odometry
            # TODO: support lookahead_n > 1 (chunk_np cols > 2, rows contain N [v, w] pairs)
            with self._odom_lock:
                if self._last_odom_xyyaw is None:
                    return
                x0, y0, yaw0 = self._last_odom_xyyaw
            dt = self.step_sec
            x, y, th = x0, y0, yaw0
            path_msg.poses.append(_make_pose(x, y, th))
            for i in range(chunk_np.shape[0]):
                v = float(chunk_np[i, 0])  # col 0 = linear.x
                w = float(chunk_np[i, 1])  # col 1 = angular.z
                if not math.isfinite(v): v = 0.0
                if not math.isfinite(w): w = 0.0
                x += v * math.cos(th) * dt
                y += v * math.sin(th) * dt
                th += w * dt
                path_msg.poses.append(_make_pose(x, y, th))

        elif self._viz_is_waypoints_xy:
            # Flat (x, y[, yaw]) per horizon step in waypoint_path_frame (e.g. odom) — not integrated v/w.
            path_msg.header.frame_id = (
                str(self._waypoint_path_frame or "").strip() or "odom"
            )
            for i in range(chunk_np.shape[0]):
                x = float(chunk_np[i, 0])
                y = float(chunk_np[i, 1])
                yaw = 0.0
                path_msg.poses.append(_make_pose(x, y, yaw))
        else:
            return

        self._waypoint_path_pub.publish(path_msg)

    def _snapshot_chunk_anchor_odom(self) -> Tuple[float, float, float]:
        with self._odom_lock:
            if self._last_odom_xyyaw is None:
                return (float("nan"), float("nan"), float("nan"))
            return self._last_odom_xyyaw

    def _reference_row_for_csv(self, n_d: int, chunk_id: int = -1) -> list[float]:
        """Return reference values for CSV logging, or [] to omit the ref columns entirely.

        For odom-position actions: actual displacement from chunk anchor at execution time,
        i.e. ref_di = current_odom_i - anchor_i  (local relative displacement).
        For twist actions: latest TwistStamped from the reference_cmd_topic (unchanged).
        """
        if n_d <= 0:
            return []

        if self._action_is_odom_xy:
            with self._odom_lock:
                odom = self._last_odom_xyyaw
            if odom is None:
                return [float("nan")] * n_d
            anchor = self._chunk_anchors.get(chunk_id)
            if anchor is None:
                return [float("nan")] * n_d
            rel = [odom[0] - anchor[0], odom[1] - anchor[1]]
            row = [float("nan")] * n_d
            for i, v in enumerate(rel[:n_d]):
                row[i] = float(v)
            return row

        # Twist mode: latest TwistStamped reference command (existing behaviour, unchanged).
        if not self._reference_cmd_topic:
            return []
        with self._ref_lock:
            if self._ref_vec is None:
                return [float("nan")] * n_d
            r = np.asarray(self._ref_vec, dtype=np.float64).ravel()
        row = [float("nan")] * n_d
        m = min(n_d, int(r.size))
        for i in range(m):
            row[i] = float(r[i])
        return row

    def _append_executed_csv(
        self,
        chunk_id: int,
        step_index: int,
        exec_time_ns: int,
        action_vec: np.ndarray,
    ) -> None:
        d = self._run_log_dir()
        if d is None:
            return
        path = d / "executed.csv"
        vec = np.asarray(action_vec, dtype=np.float32).ravel()
        n_d = int(vec.size)
        ref_suffix = self._reference_row_for_csv(n_d, chunk_id=chunk_id)
        write_header = (not path.exists()) or (not self._exec_csv_header_written)
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                cols = (
                    ["chunk_id", "step_index", "exec_time_ns"]
                    + ["anchor_x", "anchor_y", "anchor_yaw"]
                    + [f"d{i}" for i in range(n_d)]
                )
                if ref_suffix:
                    cols += [f"ref_d{i}" for i in range(n_d)]
                w.writerow(cols)
                self._exec_csv_header_written = True
            anchor = self._chunk_anchors.get(
                chunk_id, (float("nan"), float("nan"), float("nan"))
            )
            row = (
                [chunk_id, step_index, exec_time_ns]
                + [anchor[0], anchor[1], anchor[2]]
                + vec.tolist()
            )
            if ref_suffix:
                row += ref_suffix
            w.writerow(row)

    def _publish_debug_predicted_chunk(self, chunk_id: int, arr: np.ndarray) -> None:
        if self._debug_chunk_pub is None:
            return
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        horizon, dim = int(a.shape[0]), int(a.shape[1])
        flat = a.reshape(-1)
        n = int(flat.size)
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(
                label=f"chunk_id:{chunk_id} horizon:{horizon} dim:{dim}",
                size=n,
                stride=n,
            )
        ]
        msg.layout.data_offset = 0
        msg.data = flat.tolist()
        self._debug_chunk_pub.publish(msg)

    def _save_predicted_chunk_npy(self, chunk_id: int, arr: np.ndarray) -> None:
        d = self._run_log_dir()
        if d is None:
            return
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        np.save(d / f"chunk_{chunk_id:06d}.npy", a)

    def _publish_executed_debug(
        self,
        action_vec: np.ndarray,
        chunk_id: int,
        step_index: int,
    ) -> None:
        exec_ns = self.get_clock().now().nanoseconds
        if self._debug_exec_pub is not None:
            self._debug_exec_pub.publish(
                self._float32_multiarray_1d(
                    action_vec,
                    dim_label=f"chunk_id:{chunk_id} step:{step_index}",
                )
            )
        self._append_executed_csv(chunk_id, step_index, exec_ns, action_vec)

    # ---------------- Timers (persistent) ----------------
    def _feedback_tick(self) -> None:
        if (
            self._active_handle is None
            or not self._running_event.is_set()
            or self._finishing.is_set()
        ):
            return
        rem_s: Optional[int] = None
        if self._deadline_end_ns is not None:
            now_ns = self.get_clock().now().nanoseconds
            rem_s = max(0, (self._deadline_end_ns - now_ns) // 1_000_000_000)
        try:
            fb = RunPolicy.Feedback()
            if hasattr(fb, "published_actions"):
                fb.published_actions = int(self._pub_count)
            if hasattr(fb, "queue_depth"):
                with self._queue_lock:
                    fb.queue_depth = int(len(self._queue))
            if hasattr(fb, "status"):
                fb.status = "executing"
            if rem_s is not None and hasattr(fb, "seconds_remaining"):
                fb.seconds_remaining = int(rem_s)
            self._active_handle.publish_feedback(fb)
        except (RuntimeError, AttributeError) as e:
            self.get_logger().warning(f"Feedback timer publish failed: {e!r}")

    def _deadline_tick(self) -> None:
        if (
            not self._deadline_active
            or self._active_handle is None
            or self._finishing.is_set()
        ):
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._deadline_end_ns is not None and now_ns >= self._deadline_end_ns:
            self.get_logger().warning(
                f"Policy run timed out after {self._max_duration_s:.1f}s."
            )
            self._finish_run(timeout=True)

    # ---------------- Action callbacks ----------------
    def _goal_cb(self, _req) -> GoalResponse:
        if self._active_handle is not None:
            self.get_logger().info("Goal request: REJECT (already running)")
            return GoalResponse.REJECT
        self.get_logger().info("Goal request: ACCEPT")
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
        if self._active_handle is None or goal_handle != self._active_handle:
            return CancelResponse.REJECT
        self.get_logger().info("Action cancel requested")
        self._stop_requested.set()
        return CancelResponse.ACCEPT

    def _cancel_service_cb(self, _req, resp):
        self.get_logger().info("Cancel service called")
        self._stop_requested.set()
        self._done_event.set()  # Wake executor immediately
        resp.success = True
        resp.message = "Policy run cancellation requested"
        return resp

    # ---- Execute lifecycle ---------------------------------------------------
    def _execute_cb(self, goal_handle) -> RunPolicy.Result:
        if self._active_handle is not None:
            goal_handle.abort()
            res = RunPolicy.Result()
            res.success = False
            res.message = "Already running"
            return res

        self._active_handle = goal_handle
        self._stop_requested.clear()
        self._done_event.clear()
        self._finishing.clear()
        self._running_event.set()
        self._pub_count = 0
        self._run_start_ns = self.get_clock().now().nanoseconds
        self._exec_csv_header_written = False
        self._chunk_counter = 0
        self._chunk_anchors.clear()
        with self._ref_lock:
            self._ref_vec = None
        with self._queue_lock:
            self._queue.clear()

        task = getattr(goal_handle.request, "task", None)
        prompt = getattr(goal_handle.request, "prompt", None)
        self._prompt = task or prompt or ""

        if hasattr(self.policy, "reset"):
            try:
                self.policy.reset()
            except (RuntimeError, AttributeError) as e:
                self.get_logger().warning(f"policy.reset() failed: {e!r}")

        self.get_logger().info(
            f"{_ACTION_NAME}: started (task='{self._prompt}')")

        # Arm deadline
        now_ns = self.get_clock().now().nanoseconds
        self._deadline_end_ns = now_ns + int(self._max_duration_s * 1e9)
        self._deadline_active = True

        self._done_event.wait()

        # Send terminal status from execute callback thread (canonical ROS 2 pattern)
        status, msg = (self._terminal or ("aborted", "No active goal"))
        was_action_cancel = goal_handle.is_cancel_requested  # only true for real action cancels

        try:
            if was_action_cancel:
                goal_handle.canceled()            # valid: EXECUTING -> CANCELING -> CANCELED
            elif status == "timeout":
                goal_handle.abort()               # timeouts are typically 'aborted'
            elif status == "canceled":            # cancel via Trigger service → no action cancel
                status, msg = "aborted", "Cancelled via service"
                goal_handle.abort()               # valid from EXECUTING
            elif status == "succeeded":
                goal_handle.succeed()
            else:
                goal_handle.abort()
        except (RuntimeError, AttributeError) as e:
            self.get_logger().warning(f"Result send failed: {e!r}")

        # Build the Result payload to match the status
        ok = status == "succeeded"
        # Cleanup for next goal
        self._active_handle = None
        self._terminal = None

        return self._mk_result(ok, msg)
    def _mk_result(self, success: bool, message: str) -> RunPolicy.Result:
        res = RunPolicy.Result()
        res.success = bool(success)
        res.message = str(message)
        return res

    def _finish_run(self, timeout: bool = False) -> None:
        if self._finishing.is_set():
            return
        self._finishing.set()
        self._running_event.clear()
        self._deadline_active = False

        # Decide outcome only - don't send status from worker thread
        if timeout:
            self._terminal = ("timeout", f"Completed successfully after {self._max_duration_s:.1f}s")
            self.get_logger().info(f"{_ACTION_NAME}: completed (timeout)")
        elif self._stop_requested.is_set():
            self._terminal = ("canceled", "Cancelled")
            self.get_logger().info(f"{_ACTION_NAME}: stopped (cancelled)")
        else:
            self._terminal = ("succeeded", "Policy run ended")
            self.get_logger().info(f"{_ACTION_NAME}: stopped (succeeded)")

        self._publish_safety_command(increment_count=False, log_message="Published safety command on stop.")
        with self._queue_lock:
            self._queue.clear()
        self._prompt = ""
        self._done_event.set()

    def _create_safety_vector(self) -> np.ndarray:
        """Create safety action vector.
        Position mode: hold last valid coordinate (zeros would send robot to training-frame origin).
        Velocity mode: zeros (stop).
        """
        if self._action_is_odom_xy:
            if self._last_action is not None:
                return self._last_action.copy()
        elif self._safety_behavior == "hold" and self._last_action is not None:
            return self._last_action.copy()
        total_size = sum(len(spec.names or []) for spec in self._action_specs)
        if total_size == 0:
            total_size = 2
        return np.zeros((total_size,), dtype=np.float32)

    def _publish_action_vector(
        self,
        action_vec: np.ndarray,
        increment_count: bool = True,
        log_message: str = None,
        error_context: str = "action",
        chunk_id: int = -1,
        step_index: int = -1,
    ) -> None:
        """Publish an action vector with consistent error handling.
        
        Args:
            action_vec: The action vector to publish
            increment_count: Whether to increment the publish counter
            log_message: Optional message to log after successful publish
            error_context: Context string for error messages
            chunk_id: Debug trace id for chunk inference (-1 if unknown / safety)
            step_index: Step within chunk (-1 if unknown / safety)
        """
        published_ok = False
        try:
            # Handle multiple action specs by splitting the action vector
            if len(self._action_specs) > 1:
                published_ok = self._publish_multiple_actions(
                    action_vec, increment_count, log_message, error_context
                )
                if published_ok:
                    self.get_logger().info(f"Published {len(self._action_specs)} action specs")
            else:
                # Single action spec (original behavior)
                msg = encode_value(
                    ros_type=self._act_spec.ros_type,
                    names=self._act_spec.names,
                    action_vec=action_vec,
                    clamp=getattr(self._act_spec, "clamp", None),
                )
                self._act_pub.publish(msg)
                self.get_logger().info(f"Published single action spec")
                self.get_logger().info(f"Action vector shape: {action_vec.shape}")
                self.get_logger().info(f"Action vector: {action_vec}")
                
                if increment_count:
                    self._pub_count += 1
                if log_message:
                    self.get_logger().info(log_message)
                published_ok = True

            if published_ok and (
                self._debug_exec_pub is not None or self._run_log_dir() is not None
            ):
                self._publish_executed_debug(action_vec, chunk_id, step_index)

        except (RuntimeError, ValueError, TypeError) as e:
            action_type = getattr(self._act_spec, "ros_type", "unknown")
            action_names = getattr(self._act_spec, "names", None)
            vec_len = len(action_vec) if action_vec is not None else "unknown"
            names_count = len(action_names) if action_names else 0
            self.get_logger().error(
                f"{error_context} publish error: {e} "
                f"(type={action_type}, names_count={names_count}, vec_len={vec_len})"
            )

    def _publish_multiple_actions(
        self,
        action_vec: np.ndarray,
        increment_count: bool = True,
        log_message: str = None,
        error_context: str = "action",
    ) -> bool:
        """Publish action vector to multiple action specs by splitting based on names.

        Returns True if at least one spec was published successfully.
        """
        start_idx = 0
        published_count = 0
        
        for spec in self._action_specs:
            try:
                # Calculate how many values this spec needs
                spec_len = len(spec.names) if spec.names else 0
                if spec_len == 0:
                    continue
                    
                # Extract the portion of the action vector for this spec
                end_idx = start_idx + spec_len
                if end_idx > len(action_vec):
                    self.get_logger().error(
                        f"Action vector too short for spec {spec.topic}: "
                        f"need {spec_len} values, have {len(action_vec) - start_idx}"
                    )
                    break
                    
                spec_action_vec = action_vec[start_idx:end_idx]
                
                # Encode and publish
                msg = encode_value(
                    ros_type=spec.ros_type,
                    names=spec.names,
                    action_vec=spec_action_vec,
                    clamp=getattr(spec, "clamp", None),
                )
                
                pub = self._act_pubs[spec.topic]
                pub.publish(msg)
                published_count += 1
                start_idx = end_idx
                
            except (RuntimeError, ValueError, TypeError) as e:
                self.get_logger().error(
                    f"{error_context} publish error for {spec.topic}: {e} "
                    f"(type={spec.ros_type}, names_count={len(spec.names) if spec.names else 0})"
                )
        
        if increment_count and published_count > 0:
            self._pub_count += 1
        if log_message and published_count > 0:
            self.get_logger().info(f"{log_message} (published to {published_count} topics)")
        return published_count > 0

    def _publish_safety_command(self, increment_count: bool = True, log_message: str = None) -> None:
        """Publish safety command (zeros or hold last action)."""
        safety_vec = self._create_safety_vector()
        self._publish_action_vector(
            safety_vec,
            increment_count,
            log_message,
            "safety",
            chunk_id=-1,
            step_index=-1,
        )

    def _reference_cmd_cb(self, msg: TwistStamped) -> None:
        try:
            v = decode_value("geometry_msgs/msg/TwistStamped", msg, self._act_spec)
            arr = np.asarray(v, dtype=np.float64).ravel()
            with self._ref_lock:
                self._ref_vec = arr.copy()
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            self.get_logger().warning(f"reference_cmd decode failed: {e!r}")

    # ---------------- Sub callback ----------------
    def _obs_cb(self, msg, spec: SpecView) -> None:
        use_header = (spec.stamp_src ==
                      "header") or self._params.use_header_time
        ts = stamp_from_header_ns(msg) if use_header else None
        ts_ns = int(
            ts) if ts is not None else self.get_clock().now().nanoseconds
        val = decode_value(spec.ros_type, msg, spec)
        if val is None:
            # if self._bridge_debug:
            #     self.get_logger().warning(
            #         f"[bridge_debug] decode_value None → no buffer push: topic={spec.topic} "
            #         f"key={spec.key} type={spec.ros_type} ts_ns={ts_ns}"
            #     )
            return
        # Mirror the subscription key used at construction time
        if spec.key == "observation.state" and len(self._state_specs) > 1:
            dict_key = f"{spec.key}_{spec.topic.replace('/', '_')}"
        else:
            dict_key = spec.key
        buf = self._subs[dict_key].buf
        buf.push(ts_ns, val)
        # StreamBuffer is not a list: it keeps a single latest sample (last_ts, last_val).
        # if self._bridge_debug:
        #     self.get_logger().info(
        #         f"[bridge_debug] stored observation key={dict_key} topic={spec.topic} "
        #         f"stored_samples={buf.stored_sample_count()} "
        #         f"payload={observation_stored_footprint(buf.last_val)} "
        #         f"last_ts={buf.last_ts} policy={buf.policy} tol_ns={buf.tol_ns}"
        #     )

    # ---------------- Producer: timer tick (persistent) ----------------
    def _producer_tick(self) -> None:
        if not self._running_event.is_set() or self._finishing.is_set():
            return
        if self._stop_requested.is_set():
            try:
                self._finish_run(timeout=False)
            except (RuntimeError, AttributeError) as e:
                self.get_logger().error(
                    f"finish on cancel (producer) failed: {e!r}")
            return
        try:
            self._produce_actions()
        except (RuntimeError, ValueError, TypeError) as e:
            self.get_logger().error(f"producer tick failed: {e!r}")

    def _produce_actions(self) -> int:
        """Run policy inference and enqueue actions. Returns number produced."""
        use_chunks = self._params.use_chunks
        k = self._params.actions_per_chunk
        thr = float(self._params.chunk_size_threshold)

        produced = 0
        with self._queue_lock:
            queue_length = len(self._queue)
            need_chunk = use_chunks and (
                queue_length == 0 or (k > 0 and (queue_length / max(1, k)) <= thr)
            )
        if use_chunks and not need_chunk:
            if self._last_chunk_np is not None:
                self._publish_waypoint_path(self._last_chunk_np)
            return 0

        self._producer_buffer.clear()
        use_autocast = self._params.use_autocast and (
            hasattr(torch.amp, "autocast_mode")
            and torch.amp.autocast_mode.is_autocast_available(self.device.type)
        )

        # 1) Choose a sampling time (can be header time) for observations
        sample_t_ns = None
        if self._params.use_header_time:
            ts = self._get_most_recent_image_timestamp()
            # Guard: if header time is too stale relative to node clock, ignore it
            if ts is not None:
                skew = self.get_clock().now().nanoseconds - ts
                if 0 <= skew <= int(500e6):  # <= 500 ms stale is OK
                    sample_t_ns = ts
        if sample_t_ns is None:
            sample_t_ns = self.get_clock().now().nanoseconds

        # if self._bridge_debug:
        #     self.get_logger().info(
        #         f"[bridge_debug] _produce_actions sample_t_ns={sample_t_ns} "
        #         f"now_ns={self.get_clock().now().nanoseconds} "
        #         f"use_header_time={self._params.use_header_time}"
        #     )

        obs_frame = self._sample_obs_frame(sample_t_ns)
        batch = self._prepare(obs_frame)
        batch = self.preprocessor(batch)

        with torch.inference_mode():
            cm = (
                torch.autocast(self.device.type, enabled=use_autocast)
            )
            with cm:
                if use_chunks:
                    try:
                        chunk = self.policy.predict_action_chunk(batch)
                        self.get_logger().info(f"Generated action chunk shape: {chunk.shape}")
                        chunk = chunk.squeeze(0)
                        chunk = self._postprocess_actions(chunk)
                        self.get_logger().info(f"Postprocessed action chunk shape: {chunk.shape}")
                    except Exception as e:
                        self.get_logger().error(f"Error generating actions: {e}")
                        import traceback
                        self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                        return 0

                    self._chunk_counter += 1
                    cid = self._chunk_counter
                    ax, ay, ayaw = self._snapshot_chunk_anchor_odom()
                    self._chunk_anchors[cid] = (ax, ay, ayaw)
                    if self._debug_chunk_log_dir and not (
                        math.isfinite(ax) and math.isfinite(ay) and math.isfinite(ayaw)
                    ):
                        self.get_logger().warning(
                            f"Chunk {cid}: no valid odometry for CSV anchor yet "
                            f"(check chunk_log_anchor_odom_topic); anchor_* will be nan"
                        )
                    chunk_np = chunk.detach().cpu().float().numpy()
                    self._last_chunk_np = chunk_np
                    self._publish_debug_predicted_chunk(cid, chunk_np)
                    self._save_predicted_chunk_npy(cid, chunk_np)
                    self._publish_waypoint_path(chunk_np)

                    # 2) Always schedule publishes on the node clock,
                    #    aligned to the next execution tick and in the future.
                    now_wall = self.get_clock().now().nanoseconds
                    base_t = self._next_exec_tick_ns(now_wall + self.step_ns)
                    for i in range(k):
                        t_i = base_t + i * self.step_ns
                        self._producer_buffer.append(
                            (
                                t_i,
                                np.asarray(
                                    chunk[i].detach().cpu().numpy(),
                                    dtype=np.float32, #TODO: We might want to have the option to use other dtypes
                                ).ravel(),
                                cid,
                                i,
                            )
                        )
                        produced = k
                    if self._producer_buffer:
                        self.get_logger().info(
                            f"[dbg] enqueued {len(self._producer_buffer)} actions "
                            f"sched=[{self._producer_buffer[0][0]}..{self._producer_buffer[-1][0]}]"
                        )
                else:
                    try:
                        a = self.policy.select_action(batch)
                        
                        self.get_logger().info(f"Generated single action shape: {a.shape}")
                        a = self._postprocess_actions(a)
                        self.get_logger().info(f"Postprocessed single action shape: {a.shape}")
                    except Exception as e:
                        self.get_logger().error(f"Error generating single action: {e}")
                        import traceback
                        self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                        return 0
                    act_vec = np.asarray(
                        a[0].detach().cpu().numpy(), dtype=np.float32
                    ).ravel()
                    self._chunk_counter += 1
                    cid = self._chunk_counter
                    ax, ay, ayaw = self._snapshot_chunk_anchor_odom()
                    self._chunk_anchors[cid] = (ax, ay, ayaw)
                    if self._debug_chunk_log_dir and not (
                        math.isfinite(ax) and math.isfinite(ay) and math.isfinite(ayaw)
                    ):
                        self.get_logger().warning(
                            f"Chunk {cid}: no valid odometry for CSV anchor yet "
                            f"(check chunk_log_anchor_odom_topic); anchor_* will be nan"
                        )
                    self._last_chunk_np = act_vec.reshape(1, -1)
                    self._publish_debug_predicted_chunk(cid, act_vec)
                    self._save_predicted_chunk_npy(cid, act_vec)
                    self._publish_waypoint_path(self._last_chunk_np)
                    if self._predicted_action_pub is not None:
                        self._predicted_action_pub.publish(
                            self._float32_multiarray_1d(act_vec)
                        )
                    now_wall = self.get_clock().now().nanoseconds
                    t0 = self._next_exec_tick_ns(now_wall + self.step_ns)
                    self._producer_buffer.append((t0, act_vec, cid, 0))
                    produced = 1

        if self._producer_buffer:
            with self._queue_lock:
                self._queue.extend(self._producer_buffer)

        batch_summary = []
        for bk, v in batch.items():
            if torch.is_tensor(v):
                batch_summary.append(f"{bk}=tensor{tuple(v.shape)}:{v.dtype}")
            else:
                batch_summary.append(f"{bk}={type(v).__name__}")
        self.get_logger().info("[dbg] inference input: " + ", ".join(batch_summary))

        return produced

    # ---------------- Executor: timer tick (persistent) ----------------
    def _executor_tick(self) -> None:
        if not self._running_event.is_set() or self._finishing.is_set():
            return
        self.get_logger().info(f"Executor tick started  with queue length: {len(self._queue)}")
        now_ns = self.get_clock().now().nanoseconds

        # Warmup: widen tolerance x4 and avoid noisy warnings
        tol_ns = self._publish_tolerance_ns

        act_vec: Optional[np.ndarray] = None
        with self._queue_lock:
            # Drop clearly stale actions so we can catch up
            while self._queue and (self._queue[0][0] < now_ns - tol_ns):
                self._queue.popleft()
            
            # Check if queue is empty (either initially or after cleanup)
            if not self._queue:
                self.get_logger().warning(
                    "Executor tick: queue empty, publishing safety command"
                )
                # Publish safety command instead of skipping
                self._publish_safety_command()
                return

            # Find best action after cleanup
            best_idx = -1
            best_abs = None
            for idx, (t_ns, _, _, _) in enumerate(self._queue):
                d = abs(t_ns - now_ns)
                if best_abs is None or d < best_abs:
                    best_abs, best_idx = d, idx

            if best_abs is None or best_abs > tol_ns:
                head_dt_ms = (
                    (self._queue[0][0] - now_ns) /
                    1e6 if self._queue else 0
                )
                self.get_logger().warning(
                    f"No action within ±{tol_ns/1e6:.1f}ms of now "
                    f"(head Δ={head_dt_ms:.1f}ms, size={len(self._queue)})"
                )
                return

            # Remove all actions before the best one
            for _ in range(best_idx):
                self._queue.popleft()
            _t_sel, act_vec, chunk_id, step_index = self._queue.popleft()

        self._last_action = act_vec
        self._publish_action_vector(
            act_vec, chunk_id=chunk_id, step_index=step_index
        )


    def _get_most_recent_image_timestamp(self) -> Optional[int]:
        """Get the timestamp of the most recent primary image observation."""
        # Look for image observations by checking if the spec has image_resize set
        image_keys = []
        for key, sub_state in self._subs.items():
            if hasattr(sub_state.spec, 'image_resize') and sub_state.spec.image_resize is not None:
                image_keys.append(key)
        
        if not image_keys:
            return None
            
        # Get the most recent timestamp from image observations
        most_recent_ts = None
        for key in image_keys:
            latest_ts = getattr(self._subs[key].buf, 'last_ts', None)
            if latest_ts is not None:
                if most_recent_ts is None or latest_ts > most_recent_ts:
                    most_recent_ts = latest_ts
        
        # Optional: log clock skew for debugging
        if most_recent_ts is not None:
            skew_ms = (self.get_clock().now().nanoseconds - most_recent_ts) / 1e6
            self.get_logger().info(f"obs-header skew: {skew_ms:.1f} ms")
        
            # Verificar skew de otros sensores
            for key, sub_state in self._subs.items():
                if key not in image_keys:
                    latest_ts = getattr(sub_state.buf, 'last_ts', None)
                    if latest_ts:
                        sensor_skew = (most_recent_ts - latest_ts) / 1e6
                        # self.get_logger().info(
                        #     f"⏰ [{key}] skew vs image: {sensor_skew:.1f}ms"
                        # )
                    
        return most_recent_ts

    # ---------------- Observation sampling ----------------
    def _sample_obs_frame(self, sample_t_ns: int) -> Dict[str, Any]:
        obs_frame: Dict[str, Any] = {}
        zero_padded_keys = []
        
        # Handle multiple observation.state specs by consolidating them
        if len(self._state_specs) > 1:
            state_parts = []
            for sv in self._state_specs:
                dict_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
                if dict_key in self._subs:
                    v = self._subs[dict_key].buf.sample(sample_t_ns)
                    if v is None:
                        zp = self._obs_zero[dict_key]
                        v = zp.copy() if isinstance(zp, np.ndarray) else zp
                        self.get_logger().warning(
                            f"Observation {dict_key} is None, zero padding"
                        )
                        if self._bridge_debug:
                            st = self._subs[dict_key]
                            self.get_logger().warning(
                                f"[bridge_debug] zero-pad reason [{dict_key}] "
                                f"topic={st.spec.topic} sample_t_ns={sample_t_ns}: "
                                f"{st.buf.sample_miss_reason(sample_t_ns)}"
                            )
                    state_parts.append(v)
                else:
                    # Fallback to zero padding if subscription missing
                    zp = self._obs_zero.get(dict_key, np.zeros((len(sv.names),), dtype=np.float32))
                    state_parts.append(zp.copy() if isinstance(zp, np.ndarray) else zp)
            
            # Concatenate all state parts in contract order
            if state_parts:
                obs_frame["observation.state"] = np.concatenate(state_parts, axis=0)
            else:
                obs_frame["observation.state"] = np.zeros((0,), dtype=np.float32)
        
        # Handle all other observations
        for key, st in self._subs.items():
            # Skip individual state keys if we have multiple state specs (already handled above)
            if key.startswith("observation.state_") and len(self._state_specs) > 1:
                continue
                
            v = st.buf.sample(sample_t_ns)
            if v is None:
                zero_padded_keys.append(key)
                zp = self._obs_zero[key]
                obs_frame[key] = zp.copy() if isinstance(
                    zp, np.ndarray) else zp
                self.get_logger().warning(f"Observation {key} is None, zero padding")
                if self._bridge_debug:
                    self.get_logger().warning(
                        f"[bridge_debug] zero-pad reason [{key}] "
                        f"topic={st.spec.topic} sample_t_ns={sample_t_ns}: "
                        f"{st.buf.sample_miss_reason(sample_t_ns)}"
                    )
            else:
                obs_frame[key] = v

        if zero_padded_keys:
            self.get_logger().warning(
                f"⚠️ Zero-padded {len(zero_padded_keys)} observations: "
                f"{', '.join(zero_padded_keys)}"
            )
            
        obs_frame["task"] = self._prompt
        summary = []
        for k, v in obs_frame.items():
            if k == "task":
                summary.append(f"{k}=str(len={len(v)})")
            elif isinstance(v, np.ndarray):
                summary.append(f"{k}=ndarray{tuple(v.shape)}")
            else:
                summary.append(f"{k}={type(v).__name__}")
        # self.get_logger().info(
        #     f"[dbg] sampled obs for inference at sample_t_ns={sample_t_ns}: " + ", ".join(summary)
        # )
        return obs_frame

    # ---------------- Batch preparation ----------------
    def _prepare(self, obs_frame: Dict[str, Any]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        for k, v in obs_frame.items():
            if v is None:
                continue
            if isinstance(v, str):
                batch[k] = v
                continue
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
                if t.ndim == 3 and t.shape[2] in (1, 3, 4):
                    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
                    if np.issubdtype(v.dtype, np.integer):
                        max_val = float(np.iinfo(v.dtype).max)
                        t = t.to(self.device, dtype=torch.float32) / max_val
                    else:
                        t = t.to(self.device, dtype=torch.float32)
                    batch[k] = t
                    continue
                batch[k] = torch.as_tensor(
                    v, dtype=torch.float32, device=self.device)
                continue
            if torch.is_tensor(v):
                t = v
                if t.ndim == 3 and t.shape[2] in (1, 3, 4):
                    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
                batch[k] = t.to(self.device, dtype=torch.float32)
                continue
            try:
                batch[k] = torch.as_tensor(
                    v, dtype=torch.float32, device=self.device)
            except (ValueError, TypeError, RuntimeError):
                pass
        return batch

    # ---------------- Postprocess wrapper ----------------
    def _postprocess_actions(self, x):
            x = x.to(self.device)
            return self.postprocessor(x)


def main() -> None:
    """Main function to run the policy bridge node."""
    try:
        rclpy.init()
        node = PolicyBridge()
        exe = MultiThreadedExecutor(num_threads=4)
        exe.add_node(node)
        exe.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
