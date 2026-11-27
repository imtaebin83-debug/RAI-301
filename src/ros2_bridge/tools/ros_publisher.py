from __future__ import annotations

import array
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["SimPublisher", "joint_state_layout", "to_ros_time", "as_uint8_array"]

POINT_DTYPE = np.dtype({"names": ["x", "y", "z", "ring"],
                        "formats": ["<f4", "<f4", "<f4", "<u2"],
                        "offsets": [0, 4, 8, 12], "itemsize": 16})


def as_uint8_array(buf) -> "array.array":
    return array.array("B", buf.tobytes())


def yaw_split(quat_wxyz) -> "tuple[np.ndarray, np.ndarray]":
    import mujoco

    q = np.ascontiguousarray(np.asarray(quat_wxyz, float))
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, q)
    rot = rot.reshape(3, 3)
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    neg = np.empty(4)
    res = np.empty(4)
    mujoco.mju_negQuat(neg, q_yaw)
    mujoco.mju_mulQuat(res, neg, q)
    return q_yaw, res


def to_ros_time(sim_time: float):
    from builtin_interfaces.msg import Time

    sec = int(sim_time)
    return Time(sec=sec, nanosec=int(round((sim_time - sec) * 1e9)))


def joint_state_layout(model, prefix: str = "") -> Dict[str, np.ndarray]:
    import mujoco

    names: List[str] = []
    qadr: List[int] = []
    vadr: List[int] = []
    for i in range(model.njnt):
        if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if prefix and not name.startswith(prefix):
            continue
        names.append(name[len(prefix):] if prefix else name)
        qadr.append(int(model.jnt_qposadr[i]))
        vadr.append(int(model.jnt_dofadr[i]))
    return {
        "names": names,
        "qpos": np.array(qadr, np.intp),
        "qvel": np.array(vadr, np.intp),
    }


class _Tick:
    def __init__(self, hz: float) -> None:
        self.period = 1.0 / float(hz)
        self.next = 0.0

    def due(self, now: float) -> bool:
        if now < self.next:
            return False
        self.next += self.period
        if now - self.next > self.period:
            self.next = now + self.period
        return True

    def reset(self, now: float = 0.0) -> None:
        self.next = now


class SimPublisher:
    def __init__(
        self,
        model,
        prefix: str = "",
        node_name: str = "mujoco_sim",
        clock_hz: float = 100.0,
        joint_hz: float = 50.0,
        publish_effort: bool = True,
        camera_hz: float = 20.0,
        lidar_hz: float = 10.0,
        lidar: str = "OS1-32",
        lidar_columns: int = 0,
        lidar_threads: int = 1,
        camera: str = "d435i_color",
        camera_frame: str = "camera_color_optical_frame",
        lidar_frame: str = "lidar_link",
        imu_hz: float = 100.0,
        odom_hz: float = 50.0,
        odom_frame: str = "odom",
        base_frame: str = "pelvis",
        publish_tf: bool = True,
        footprint_frame: Optional[str] = None,
        odom_twist_window: float = 0.0,
        unitree_odom: bool = True,
        unitree_odom_hz: float = 500.0,
        unitree_odom_lf_hz: float = 20.0,
        unitree_dds_domain: int = 0,
        unitree_dds_interface: Optional[str] = None,
        cmd_vel_callback: Optional[Callable[[float, float, float], None]] = None,
        cmd_vel_topic: str = "/cmd_vel",
        cmd_timeout: float = 0.5,
        cmd_limits: Sequence[float] = (0.6, 0.4, 1.0),
        cmd_bias: Sequence[float] = (0.0, 0.0, 0.0),
        extra_joint_prefixes: Optional[Dict[str, str]] = None,
        extra_joint_commanders: Optional[Dict[str, Any]] = None,
        extra_rgb_cameras: Optional[Dict[str, Sequence[Any]]] = None,
        extra_rgb_hz: Optional[float] = None,
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.signals import SignalHandlerOptions
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, PointCloud2
        from std_msgs.msg import Float64, Float64MultiArray

        if not rclpy.ok():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

        self._node: "Node" = Node(node_name)
        # The clock publisher must use wall time or it waits on its own /clock messages.
        self._node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])

        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        self._clock_pub = self._node.create_publisher(Clock, "/clock", reliable)
        self._joint_pub = self._node.create_publisher(JointState, "joint_states", reliable)

        self._Clock, self._JointState = Clock, JointState
        self._Image, self._CameraInfo, self._PointCloud2 = Image, CameraInfo, PointCloud2
        self._Imu, self._Odometry = Imu, Odometry
        self._layout = joint_state_layout(model, prefix)
        self._publish_effort = bool(publish_effort)
        self._ticks = {"clock": _Tick(clock_hz), "joint": _Tick(joint_hz)}

        self._extra_joints = []
        self._pending_joint_targets: Dict[int, float] = {}
        self._pending_gripper: Dict[int, float] = {}
        extra_joint_commanders = extra_joint_commanders or {}
        for namespace, extra_prefix in (extra_joint_prefixes or {}).items():
            ns = namespace.strip("/")
            commander = extra_joint_commanders.get(ns)
            layout = joint_state_layout(model, extra_prefix)
            publisher = self._node.create_publisher(
                JointState, f"/{ns}/joint_states", reliable
            )
            joint_actuators: Dict[str, int] = {}
            for actuator in range(model.nu):
                if model.actuator_trntype[actuator] != 0:  # mjTRN_JOINT
                    continue
                joint = int(model.actuator_trnid[actuator, 0])
                joint_name = model.joint(joint).name or ""
                if joint_name.startswith(extra_prefix):
                    joint_actuators[joint_name[len(extra_prefix):]] = actuator
            def joint_command(msg, mapping=joint_actuators, control=commander) -> None:
                if len(msg.name) != len(msg.position):
                    self._node.get_logger().warning(
                        "ignored Panda joint command with mismatched name/position lengths"
                    )
                    return
                if control is not None:
                    try:
                        control.command_joint_positions(msg.name, msg.position)
                    except (TypeError, ValueError) as exc:
                        self._node.get_logger().warning(f"ignored Panda joint command: {exc}")
                    return
                for name, value in zip(msg.name, msg.position):
                    actuator = mapping.get(name)
                    if actuator is not None and np.isfinite(value):
                        self._pending_joint_targets[actuator] = float(value)

            def gripper_command(msg, control=commander) -> None:
                if np.isfinite(msg.data):
                    if control is not None:
                        control.set_gripper_opening(float(msg.data))

            joint_sub = self._node.create_subscription(
                JointState, f"/{ns}/joint_command", joint_command, reliable
            )
            gripper_sub = self._node.create_subscription(
                Float64, f"/{ns}/gripper_command", gripper_command, reliable
            )
            eef_delta_sub = None
            if commander is not None and callable(getattr(commander, "set_eef_delta", None)):
                def eef_delta_command(msg, controller=commander) -> None:
                    action = np.asarray(msg.data, dtype=float)
                    if action.shape != (7,) or not np.all(np.isfinite(action)):
                        self._node.get_logger().warning(
                            "ignored Panda EEF command: expected 7 finite normalized values"
                        )
                        return
                    controller.set_eef_delta(action[:6], normalized=True)
                    controller.set_gripper_opening(0.04 if action[6] < 0.0 else 0.0)

                eef_delta_sub = self._node.create_subscription(
                    Float64MultiArray,
                    f"/{ns}/eef_delta_command",
                    eef_delta_command,
                    reliable,
                )
            self._extra_joints.append(
                {
                    "namespace": ns,
                    "layout": layout,
                    "publisher": publisher,
                    "joint_sub": joint_sub,
                    "gripper_sub": gripper_sub,
                    "eef_delta_sub": eef_delta_sub,
                }
            )

        # Best-effort QoS avoids blocking physics; RViz subscribers must match.
        self._model, self._prefix = model, prefix
        self._camera_frame, self._lidar_frame = camera_frame, lidar_frame
        self._odom_frame, self._base_frame = odom_frame, base_frame
        from dataclasses import replace as _replace

        from ros2_bridge.tools.g1_sensor_rig import OUSTER

        spec = OUSTER[lidar]
        if lidar_columns:
            spec = _replace(spec, columns=int(lidar_columns))
        self._camera_name, self._lidar_spec = camera, spec
        if lidar_threads:
            self._lidar_threads = max(1, int(lidar_threads))
        else:
            from ros2_bridge.tools.g1_sensor_rig import default_threads

            self._lidar_threads = default_threads()
        self._renderer = None
        self._info_msg = None
        self._beams = None
        self._extra_rgb_cameras = []
        extra_rgb_hz = float(camera_hz if extra_rgb_hz is None else extra_rgb_hz)

        if camera_hz > 0:
            import mujoco

            from ros2_bridge.tools import mj_renderer
            from ros2_bridge.tools.g1_sensor_rig import camera_info

            self._color_pub = self._node.create_publisher(
                Image, "camera/color/image_raw", qos_profile_sensor_data)
            self._depth_pub = self._node.create_publisher(
                Image, "camera/depth/image_raw", qos_profile_sensor_data)
            self._info_pub = self._node.create_publisher(
                CameraInfo, "camera/camera_info", qos_profile_sensor_data)
            self._ticks["camera"] = _Tick(camera_hz)

            # Create framebuffers before the viewer thread to avoid GL_FRAMEBUFFER_UNSUPPORTED.
            name = self._prefix + self._camera_name
            info = camera_info(self._model, name)
            self._cam_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, name
            )
            if self._cam_id < 0:
                raise RuntimeError(f"G1 RGB-D camera missing: {name}")
            self._renderer = mj_renderer.OffscreenRenderer(
                self._model, width=info["width"], height=info["height"]
            )
            msg = self._CameraInfo()
            msg.header.frame_id = self._camera_frame
            msg.width, msg.height = info["width"], info["height"]
            msg.distortion_model = info["distortion_model"]
            msg.d, msg.k, msg.r, msg.p = info["d"], info["k"], info["r"], info["p"]
            self._info_msg = msg

        if extra_rgb_cameras:
            import mujoco

            for topic, camera_config in extra_rgb_cameras.items():
                if len(camera_config) not in (2, 4):
                    raise ValueError(
                        "extra RGB camera config must be (camera, frame) or "
                        "(camera, frame, width, height)"
                    )
                camera_name, frame = str(camera_config[0]), str(camera_config[1])
                width, height = (
                    (256, 256)
                    if len(camera_config) == 2
                    else (int(camera_config[2]), int(camera_config[3]))
                )
                camera_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                )
                if camera_id < 0:
                    raise RuntimeError(f"extra RGB camera missing: {camera_name}")
                clean_topic = topic.strip("/")
                image_pub = self._node.create_publisher(
                    Image, f"/{clean_topic}/image_raw", qos_profile_sensor_data
                )
                info_pub = self._node.create_publisher(
                    CameraInfo, f"/{clean_topic}/camera_info", qos_profile_sensor_data
                )
                fovy = np.deg2rad(float(model.cam_fovy[camera_id]))
                focal = 0.5 * height / np.tan(0.5 * fovy)
                cx, cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
                info = CameraInfo()
                info.header.frame_id = frame
                info.width, info.height = width, height
                info.distortion_model = "plumb_bob"
                info.d = [0.0] * 5
                info.k = [focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0]
                info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                info.p = [focal, 0.0, cx, 0.0, 0.0, focal, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
                self._extra_rgb_cameras.append(
                    {
                        "topic": clean_topic,
                        "camera": camera_id,
                        "frame": frame,
                        "width": width,
                        "height": height,
                        "image_pub": image_pub,
                        "info_pub": info_pub,
                        "info": info,
                    }
                )
            if extra_rgb_hz > 0:
                self._ticks["extra_camera"] = _Tick(extra_rgb_hz)
                from ros2_bridge.tools import mj_renderer

                first = self._extra_rgb_cameras[0]
                if self._renderer is None:
                    self._renderer = mj_renderer.OffscreenRenderer(
                        self._model, width=first["width"], height=first["height"]
                    )
        if lidar_hz > 0:
            self._cloud_pub = self._node.create_publisher(
                PointCloud2, "lidar/points", qos_profile_sensor_data)
            self._ticks["lidar"] = _Tick(lidar_hz)

        # Raw IMUs omit ground-truth orientation.
        self._imu_sources = []
        if imu_hz > 0:
            self._imu_sources = [
                self._make_imu_source("imu", "imu_in_pelvis",
                                      "imu_ang_vel", "imu_lin_acc",
                                      qos_profile_sensor_data),
                self._make_imu_source("camera/imu", "camera_imu_frame",
                                      "d435i_gyro", "d435i_accel",
                                      qos_profile_sensor_data),
                self._make_imu_source("lidar/imu", "imu_link",
                                      "ouster_gyro", "ouster_accel",
                                      qos_profile_sensor_data),
            ]
            self._ticks["imu"] = _Tick(imu_hz)

        self._root_qadr = self._root_dadr = None
        self._tf_broadcaster = None
        # Split TF: odom -> footprint (x, y, yaw) -> pelvis (z, roll, pitch).
        self._footprint_frame = footprint_frame or None
        if self._footprint_frame and odom_hz <= 0:
            raise ValueError("footprint_frame needs odom_hz > 0")
        self._twist_window = max(0.0, float(odom_twist_window))
        self._twist_hist: "deque" = deque()
        if odom_hz > 0:
            import mujoco

            roots = [
                j for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(prefix)
            ]
            if len(roots) != 1:
                raise ValueError(f"expected one robot free joint under {prefix!r}, found {len(roots)}")
            root = roots[0]
            self._root_qadr = int(model.jnt_qposadr[root])
            self._root_dadr = int(model.jnt_dofadr[root])
            self._odom_pub = self._node.create_publisher(Odometry, "odom", reliable)
            if publish_tf:
                from tf2_ros import TransformBroadcaster

                self._tf_broadcaster = TransformBroadcaster(self._node)
            self._ticks["odom"] = _Tick(odom_hz)

        self._unitree_odom = None
        if unitree_odom:
            from ros2_bridge.tools.unitree_odom import UnitreeOdomPublisher

            self._unitree_odom = UnitreeOdomPublisher(
                model,
                prefix,
                high_hz=unitree_odom_hz,
                low_hz=unitree_odom_lf_hz,
                domain_id=unitree_dds_domain,
                network_interface=unitree_dds_interface,
            )

        # Spin subscriptions on the physics thread to avoid policy races.
        self._cmd_callback = cmd_vel_callback
        self._cmd_timeout = max(0.0, float(cmd_timeout))
        limits = np.asarray(cmd_limits, dtype=float)
        if limits.shape != (3,) or np.any(limits <= 0) or not np.all(np.isfinite(limits)):
            raise ValueError("cmd_limits must contain three positive finite values")
        self._cmd_limits = limits
        bias = np.asarray(cmd_bias, dtype=float)
        if bias.shape != (3,) or not np.all(np.isfinite(bias)):
            raise ValueError("cmd_bias must contain three finite values")
        # Apply feed-forward after clamping, including watchdog stops, to cancel idle creep.
        self._cmd_bias = bias
        self._last_cmd_wall: Optional[float] = None
        self._last_cmd = np.zeros(3, dtype=float)
        self._cmd_timed_out = False
        self._cmd_sub = None
        if cmd_vel_callback is not None:
            self._cmd_sub = self._node.create_subscription(
                Twist, cmd_vel_topic, self._on_cmd_vel, reliable)

        self._node.get_logger().info(
            f"/clock {clock_hz:g} Hz | joint_states {joint_hz:g} Hz "
            f"({len(self._layout['names'])} joints) | G1 RGB-D {camera_hz:g} Hz | "
            f"extra RGB {extra_rgb_hz:g} Hz | "
            f"lidar {lidar_hz:g} Hz ({spec.name} {spec.channels}x{spec.columns}, "
            f"{self._lidar_threads} worker(s)) | "
            f"IMU {imu_hz:g} Hz | odom {odom_hz:g} Hz"
            + (f" (twist over {self._twist_window:g} s)" if self._twist_window > 0 else "")
            + (f" via {self._footprint_frame}" if self._footprint_frame else "")
        )
        if self._unitree_odom is not None:
            self._node.get_logger().info(
                "Unitree DDS odometry: "
                f"{self._unitree_odom.HIGH_TOPIC} {self._unitree_odom.high_hz:g} Hz | "
                f"{self._unitree_odom.LOW_TOPIC} {self._unitree_odom.low_hz:g} Hz | "
                f"domain {self._unitree_odom.domain_id}"
            )
            if self._unitree_odom.domain_id == 0:
                self._node.get_logger().warning(
                    "publishing synthetic Unitree odometry on DDS domain 0; "
                    "use --no-unitree-odom or an isolated --unitree-dds-domain "
                    "when this computer is connected to a physical G1 network"
                )
        if self._cmd_sub is not None:
            self._node.get_logger().info(
                f"listening on {cmd_vel_topic}; timeout {self._cmd_timeout:g}s; "
                f"limits vx/vy/wz={tuple(float(v) for v in self._cmd_limits)}"
                + (f", bias {tuple(float(v) for v in self._cmd_bias)}"
                   if np.any(self._cmd_bias) else "")
            )

    def publish_static_transform(
        self,
        parent: str,
        child: str,
        pos: Sequence[float],
        quat_wxyz: Sequence[float],
    ) -> None:
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster

        if not hasattr(self, "_static_tf_broadcaster"):
            self._static_tf_broadcaster = StaticTransformBroadcaster(self._node)
        qw, qx, qy, qz = (float(value) for value in quat_wxyz)
        msg = TransformStamped()
        msg.header.stamp = to_ros_time(0.0)
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = (
            float(value) for value in pos
        )
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        self._static_tf_broadcaster.sendTransform(msg)


    @property
    def node(self):
        return self._node

    @property
    def joint_names(self) -> Sequence[str]:
        return self._layout["names"]


    def update(self, data) -> None:
        self.process_callbacks()
        for actuator, target in self._pending_joint_targets.items():
            if self._model.actuator_ctrllimited[actuator]:
                target = float(np.clip(target, *self._model.actuator_ctrlrange[actuator]))
            data.ctrl[actuator] = target
        for actuator, target in self._pending_gripper.items():
            data.ctrl[actuator] = target
        now = float(data.time)
        if self._ticks["clock"].due(now):
            self.publish_clock(now)
        if self._ticks["joint"].due(now):
            self.publish_joint_states(data, now)
            self.publish_extra_joint_states(data, now)
        if "camera" in self._ticks and self._ticks["camera"].due(now):
            self.publish_camera(data, now)
        if "extra_camera" in self._ticks and self._ticks["extra_camera"].due(now):
            self.publish_extra_rgb_cameras(data, now)
        if "lidar" in self._ticks and self._ticks["lidar"].due(now):
            self.publish_lidar(data, now)
        if "imu" in self._ticks and self._ticks["imu"].due(now):
            self.publish_imus(data, now)
        if "odom" in self._ticks and self._ticks["odom"].due(now):
            self.publish_odometry(data, now)
        if self._unitree_odom is not None:
            self._unitree_odom.update(data, now)

    def _on_cmd_vel(self, msg) -> None:
        raw = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=float)
        if not np.all(np.isfinite(raw)):
            self._node.get_logger().warning("ignored non-finite /cmd_vel")
            return
        cmd = np.clip(raw, -self._cmd_limits, self._cmd_limits) + self._cmd_bias
        self._last_cmd[:] = cmd
        self._last_cmd_wall = time.monotonic()
        self._cmd_timed_out = False
        self._cmd_callback(*(float(v) for v in cmd))

    def process_callbacks(self) -> None:
        import rclpy

        rclpy.spin_once(self._node, timeout_sec=0.0)
        if self._cmd_sub is None or self._cmd_timeout <= 0 or self._last_cmd_wall is None:
            return
        if time.monotonic() - self._last_cmd_wall <= self._cmd_timeout:
            return
        if not self._cmd_timed_out:
            self._last_cmd[:] = self._cmd_bias
            self._cmd_callback(*(float(v) for v in self._cmd_bias))
            self._cmd_timed_out = True
            self._node.get_logger().warning("cmd_vel timed out; commanding a stop")

    def reset(self, sim_time: float = 0.0) -> None:
        for t in self._ticks.values():
            t.reset(sim_time)
        self._twist_hist.clear()


    def publish_clock(self, sim_time: float) -> None:
        msg = self._Clock()
        msg.clock = to_ros_time(sim_time)
        self._clock_pub.publish(msg)

    def publish_joint_states(self, data, sim_time: Optional[float] = None) -> None:
        lay = self._layout
        msg = self._JointState()
        msg.header.stamp = to_ros_time(float(data.time) if sim_time is None else sim_time)
        msg.name = list(lay["names"])
        msg.position = data.qpos[lay["qpos"]].tolist()
        msg.velocity = data.qvel[lay["qvel"]].tolist()
        if self._publish_effort:
            msg.effort = data.qfrc_actuator[lay["qvel"]].tolist()
        self._joint_pub.publish(msg)

    def publish_extra_joint_states(self, data, sim_time: Optional[float] = None) -> None:
        stamp = to_ros_time(float(data.time) if sim_time is None else sim_time)
        for extra in self._extra_joints:
            lay = extra["layout"]
            msg = self._JointState()
            msg.header.stamp = stamp
            msg.name = list(lay["names"])
            msg.position = data.qpos[lay["qpos"]].tolist()
            msg.velocity = data.qvel[lay["qvel"]].tolist()
            if self._publish_effort:
                msg.effort = data.qfrc_actuator[lay["qvel"]].tolist()
            extra["publisher"].publish(msg)

    def publish_camera(self, data, sim_time: Optional[float] = None) -> None:
        import mujoco

        from ros2_bridge.tools.g1_sensor_rig import D435I_COLOR, camera_info

        stamp = to_ros_time(float(data.time) if sim_time is None else sim_time)
        name = self._prefix + self._camera_name

        if self._info_msg is None:
            from ros2_bridge.tools import mj_renderer

            info = camera_info(self._model, name)
            self._cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if self._renderer is None:
                self._renderer = mj_renderer.OffscreenRenderer(
                    self._model, width=info["width"], height=info["height"])
            msg = self._CameraInfo()
            msg.header.frame_id = self._camera_frame
            msg.width, msg.height = info["width"], info["height"]
            msg.distortion_model = info["distortion_model"]
            msg.d, msg.k, msg.r, msg.p = info["d"], info["k"], info["r"], info["p"]
            self._info_msg = msg

        self._renderer.resize(self._info_msg.width, self._info_msg.height)
        rgb, depth = self._renderer.render_rgbd(data, camera=self._cam_id)

        # REP 118 uses NaN for missing float depth readings.
        depth = np.where((depth >= D435I_COLOR.z_min) & (depth <= D435I_COLOR.z_max),
                         depth, np.nan).astype(np.float32)

        color = self._Image()
        color.header.stamp, color.header.frame_id = stamp, self._camera_frame
        color.height, color.width = rgb.shape[0], rgb.shape[1]
        color.encoding, color.is_bigendian = "rgb8", 0
        color.step = rgb.shape[1] * 3
        color.data = as_uint8_array(rgb)

        dep = self._Image()
        dep.header.stamp, dep.header.frame_id = stamp, self._camera_frame
        dep.height, dep.width = depth.shape[0], depth.shape[1]
        dep.encoding, dep.is_bigendian = "32FC1", 0
        dep.step = depth.shape[1] * 4
        dep.data = as_uint8_array(depth)

        self._info_msg.header.stamp = stamp
        self._color_pub.publish(color)
        self._depth_pub.publish(dep)
        self._info_pub.publish(self._info_msg)

    def _publish_extra_rgb(self, extra: Dict[str, Any], rgb: np.ndarray, stamp) -> None:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"extra RGB frame {extra['topic']} must be uint8 HxWx3, got "
                f"{rgb.dtype} {rgb.shape}"
            )
        if rgb.shape[:2] != (extra["height"], extra["width"]):
            raise ValueError(
                f"extra RGB frame {extra['topic']} must be "
                f"{extra['width']}x{extra['height']}, got {rgb.shape[1]}x{rgb.shape[0]}"
            )
        image = self._Image()
        image.header.stamp, image.header.frame_id = stamp, extra["frame"]
        image.height, image.width = rgb.shape[:2]
        image.encoding, image.is_bigendian = "rgb8", 0
        image.step = rgb.shape[1] * 3
        image.data = as_uint8_array(rgb)
        extra["info"].header.stamp = stamp
        extra["image_pub"].publish(image)
        extra["info_pub"].publish(extra["info"])

    def publish_extra_rgb_cameras(
        self, data, sim_time: Optional[float] = None
    ) -> None:
        if not self._extra_rgb_cameras:
            return
        stamp = to_ros_time(float(data.time) if sim_time is None else sim_time)
        if self._renderer is None:
            from ros2_bridge.tools import mj_renderer

            first = self._extra_rgb_cameras[0]
            self._renderer = mj_renderer.OffscreenRenderer(
                self._model, width=first["width"], height=first["height"]
            )
        if getattr(self, "_extra_rgb_option", None) is None:
            import mujoco

            # Clean frames, as in the Panda learning data: no collision geoms
            # (groups 3-5) and no diagnostic sites (grip-site ball, axes, zones).
            option = mujoco.MjvOption()
            option.geomgroup[3:6] = 0
            option.sitegroup[:] = 0
            self._extra_rgb_option = option
        for extra in self._extra_rgb_cameras:
            rgb = self._renderer.render_rgb(
                data,
                width=extra["width"],
                height=extra["height"],
                camera=extra["camera"],
                scene_option=self._extra_rgb_option,
            )
            self._publish_extra_rgb(extra, rgb, stamp)

    def publish_extra_rgb_frames(
        self,
        frames: Dict[str, np.ndarray],
        sim_time: float,
    ) -> None:
        if not frames:
            return
        stamp = to_ros_time(float(sim_time))
        for extra in self._extra_rgb_cameras:
            rgb = frames.get(extra["topic"])
            if rgb is not None:
                self._publish_extra_rgb(extra, rgb, stamp)

    def publish_lidar(self, data, sim_time: Optional[float] = None) -> None:
        from sensor_msgs.msg import PointField

        from ros2_bridge.tools.g1_sensor_rig import beam_directions, scan

        spec = self._lidar_spec
        if self._beams is None:
            self._beams = beam_directions(spec)
        ranges, points = scan(self._model, data, spec, prefix=self._prefix,
                              dirs=self._beams, threads=self._lidar_threads)

        hit = ranges > 0
        rings = np.broadcast_to(np.arange(ranges.shape[0], dtype=np.uint16)[:, None],
                                ranges.shape)
        rec = np.zeros(int(hit.sum()), POINT_DTYPE)
        rec["x"], rec["y"], rec["z"] = points[hit].T
        rec["ring"] = rings[hit]

        msg = self._PointCloud2()
        msg.header.stamp = to_ros_time(float(data.time) if sim_time is None else sim_time)
        msg.header.frame_id = self._lidar_frame
        msg.height, msg.width = 1, rec.size
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=12, datatype=PointField.UINT16, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = POINT_DTYPE.itemsize
        msg.row_step = POINT_DTYPE.itemsize * rec.size
        msg.is_dense = True
        msg.data = as_uint8_array(rec)
        self._cloud_pub.publish(msg)

    def _sensor_slice(self, name: str) -> slice:
        import mujoco

        sid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, self._prefix + name)
        if sid < 0:
            raise KeyError(f"sensor {self._prefix + name!r} is missing; regenerate g1_dex_sensors.xml")
        adr, dim = int(self._model.sensor_adr[sid]), int(self._model.sensor_dim[sid])
        if dim != 3:
            raise ValueError(f"sensor {name!r} has dimension {dim}, expected 3")
        return slice(adr, adr + dim)

    def _make_imu_source(self, topic: str, frame: str, gyro: str, accel: str, qos):
        return {
            "publisher": self._node.create_publisher(self._Imu, topic, qos),
            "frame": frame,
            "gyro": self._sensor_slice(gyro),
            "accel": self._sensor_slice(accel),
        }

    def publish_imus(self, data, sim_time: Optional[float] = None) -> None:
        stamp = to_ros_time(float(data.time) if sim_time is None else sim_time)
        for src in self._imu_sources:
            gyro = data.sensordata[src["gyro"]]
            accel = data.sensordata[src["accel"]]
            msg = self._Imu()
            msg.header.stamp, msg.header.frame_id = stamp, src["frame"]
            msg.orientation_covariance[0] = -1.0
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, gyro)
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, accel)
            src["publisher"].publish(msg)

    def publish_odometry(self, data, sim_time: Optional[float] = None) -> None:
        import mujoco

        now = float(data.time) if sim_time is None else sim_time
        stamp = to_ros_time(now)
        qadr, dadr = self._root_qadr, self._root_dadr
        q = np.asarray(data.qpos[qadr:qadr + 7], dtype=float)
        v = np.asarray(data.qvel[dadr:dadr + 6], dtype=float)
        rot = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rot, np.ascontiguousarray(q[3:7]))
        rot = rot.reshape(3, 3)

        if self._footprint_frame is None:
            child = self._base_frame
            pos, quat = q[:3], q[3:7]
            linear, angular = rot.T @ v[:3], v[3:6]
        else:
            q_yaw, q_res = yaw_split(q[3:7])
            c, sn = float(q_yaw[0] ** 2 - q_yaw[3] ** 2), float(2.0 * q_yaw[0] * q_yaw[3])

            def to_footprint(w):
                return np.array([c * w[0] + sn * w[1], -sn * w[0] + c * w[1], w[2]])

            child = self._footprint_frame
            pos, quat = np.array([q[0], q[1], 0.0]), q_yaw
            linear, angular = to_footprint(v[:3]), to_footprint(rot @ v[3:6])
        linear, angular = self._smooth_twist(now, linear, angular)

        msg = self._Odometry()
        msg.header.stamp, msg.header.frame_id = stamp, self._odom_frame
        msg.child_frame_id = child
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, pos)
        # MuJoCo quaternion order is wxyz; ROS order is xyzw.
        msg.pose.pose.orientation.w, msg.pose.pose.orientation.x = float(quat[0]), float(quat[1])
        msg.pose.pose.orientation.y, msg.pose.pose.orientation.z = float(quat[2]), float(quat[3])
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = map(float, linear)
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z = map(float, angular)
        self._odom_pub.publish(msg)

        if self._tf_broadcaster is None:
            return
        tfs = [self._transform(stamp, self._odom_frame, child, pos, quat)]
        if self._footprint_frame is not None:
            tfs.append(self._transform(stamp, self._footprint_frame, self._base_frame,
                                       (0.0, 0.0, float(q[2])), q_res))
        self._tf_broadcaster.sendTransform(tfs)

    def _smooth_twist(self, now: float, linear, angular):
        if self._twist_window <= 0.0:
            return linear, angular
        self._twist_hist.append((now, np.concatenate([linear, angular])))
        while self._twist_hist and now - self._twist_hist[0][0] > self._twist_window:
            self._twist_hist.popleft()
        mean = np.mean([value for _, value in self._twist_hist], axis=0)
        return mean[:3], mean[3:]

    @staticmethod
    def _transform(stamp, parent: str, child: str, pos, quat_wxyz):
        from geometry_msgs.msg import TransformStamped

        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, parent, child
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            float(value) for value in pos
        )
        tf.transform.rotation.w, tf.transform.rotation.x = float(quat_wxyz[0]), float(quat_wxyz[1])
        tf.transform.rotation.y, tf.transform.rotation.z = float(quat_wxyz[2]), float(quat_wxyz[3])
        return tf


    def shutdown(self) -> None:
        import rclpy

        if self._cmd_callback is not None:
            self._cmd_callback(0.0, 0.0, 0.0)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._unitree_odom is not None:
            self._unitree_odom.close()
            self._unitree_odom = None
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def __enter__(self) -> "SimPublisher":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
