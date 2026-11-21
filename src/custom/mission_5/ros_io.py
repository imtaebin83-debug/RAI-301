"""Mission 5 sensor and action boundary.

The only dynamic inputs are the two permitted RGB cameras, Panda joint states
and the published world-from-robot-base TF.  The robot-only MJCF is used for FK;
no task scene, object pose, success diagnostic or simulator state is loaded.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

ARM_NAMES = tuple(f"robot0_joint{i}" for i in range(1, 8))
FINGER_NAMES = ("gripper0_finger_joint1", "gripper0_finger_joint2")
BASE_FRAME = "panda/robot0_base"
RGB_TOPICS = ("/panda/agentview/image_raw", "/panda/wrist_camera/image_raw")
POSITION_SCALE_M = 0.05
ROTATION_SCALE_RAD = 0.5


def stamp_seconds(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def rgb_array(message) -> np.ndarray:
    """Decode row padding and channel order without preview-only rotations."""
    encoding = message.encoding.lower()
    if encoding not in ("rgb8", "bgr8", "rgba8", "bgra8"):
        raise ValueError(f"Unsupported RGB encoding {message.encoding}")
    channels = 4 if "a8" in encoding else 3
    if message.width <= 0 or message.height <= 0 or message.step < message.width * channels:
        raise ValueError("Invalid RGB dimensions/stride")
    image = np.ndarray((message.height, message.width, channels), dtype=np.uint8,
                       buffer=message.data, strides=(message.step, channels, 1))[..., :3]
    if encoding.startswith("bgr"):
        image = image[..., ::-1]
    result = np.ascontiguousarray(image)
    if result.shape != (256, 256, 3):
        raise ValueError(f"Expected raw 256x256 RGB, received {result.shape}")
    return result


def joint_mapping(message) -> dict[str, float]:
    if len(message.name) != len(message.position):
        raise ValueError("Joint name/position lengths differ")
    values = {name.rsplit("/", 1)[-1]: float(value)
              for name, value in zip(message.name, message.position)}
    missing = [name for name in ARM_NAMES + FINGER_NAMES if name not in values]
    if missing or not all(np.isfinite(values[name]) for name in ARM_NAMES + FINGER_NAMES):
        raise ValueError(f"Missing or nonfinite Panda joints: {missing}")
    return values


def transform_matrix(message) -> np.ndarray:
    transform = message.transform if hasattr(message, "transform") else message
    q, p = transform.rotation, transform.translation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [p.x, p.y, p.z]
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite robot-base transform")
    return matrix


def base_action_to_world(action, world_from_base) -> np.ndarray:
    """Match runtime's base-frame left-multiplied delta at the ROS world boundary.

    Translation and rotation use isotropic scales, so both normalized vectors
    rotate by R_world_base.  Direction-preserving limiting handles a rotated
    diagonal that would exceed the ROS command's per-world-axis [-1,1] range.
    """
    action = np.asarray(action, dtype=np.float64)
    transform = np.asarray(world_from_base, dtype=np.float64)
    if action.shape != (7,) or transform.shape != (4, 4):
        raise ValueError("Expected action[7] and world_from_base[4,4]")
    if not np.isfinite(action).all() or not np.isfinite(transform).all():
        raise ValueError("Nonfinite action or base transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .999:
        raise ValueError("Invalid base rotation")
    result = np.empty(7)
    for start in (0, 3):
        vector = rotation @ np.clip(action[start:start + 3], -1., 1.)
        result[start:start + 3] = vector / max(1., float(np.abs(vector).max()))
    result[6] = -1. if action[6] < 0 else 1.
    return result


class PandaKinematics:
    """Forward kinematics from joint sensors and the public robot description."""
    def __init__(self, model_path=None):
        import mujoco

        source = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path is not None else (
            source / "environment/robots/description/franka_description/libero_panda/panda.xml")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.mujoco = mujoco
        self._addresses = {}
        for name in ARM_NAMES + FINGER_NAMES:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError("Robot description is missing " + name)
            self._addresses[name] = int(self.model.jnt_qposadr[joint])
        self._base = self._body_id("robot0_base")
        self._hand = self._body_id("robot0_right_hand")
        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        if self._site < 0:
            raise ValueError("Robot description lacks grip site")
        self.base_from_eef = np.eye(4)
        self.base_from_hand = np.eye(4)

    def _body_id(self, name):
        identifier = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if identifier < 0:
            raise ValueError("Robot description is missing " + name)
        return identifier

    def _body_transform(self, index):
        matrix = np.eye(4)
        matrix[:3, :3] = self.data.xmat[index].reshape(3, 3)
        matrix[:3, 3] = self.data.xpos[index]
        return matrix

    def state(self, joints: dict[str, float]) -> np.ndarray:
        if not all(name in joints and np.isfinite(joints[name]) for name in self._addresses):
            raise ValueError("FK needs all finite Panda arm and finger positions")
        for name, address in self._addresses.items():
            self.data.qpos[address] = joints[name]
        self.mujoco.mj_forward(self.model, self.data)
        base_inverse = np.linalg.inv(self._body_transform(self._base))
        eef = np.eye(4)
        eef[:3, :3] = self.data.site_xmat[self._site].reshape(3, 3)
        eef[:3, 3] = self.data.site_xpos[self._site]
        self.base_from_eef = base_inverse @ eef
        self.base_from_hand = base_inverse @ self._body_transform(self._hand)
        return np.asarray([*self.base_from_eef[:3, 3],
                           *Rotation.from_matrix(self.base_from_eef[:3, :3]).as_rotvec(),
                           *(joints[name] for name in FINGER_NAMES)], dtype=np.float32)


class PandaROSIO:
    def __init__(self, mission, *, kinematics=None, max_skew=.25, max_age=1.):
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Float64, Float64MultiArray

        self.mission = mission
        self.kinematics = kinematics or PandaKinematics()
        self.max_skew, self.max_age = float(max_skew), float(max_age)
        self._Image, self._JointState = Image, JointState
        self._Float64, self._Action = Float64, Float64MultiArray
        self._stamps, self._received = {}, {}
        self.last_joints = None
        self.world_from_base = None
        self.base_parent = None
        self._eef_pub = mission.create_publisher(Float64MultiArray, "/panda/eef_delta_command", 10)
        self._joint_pub = mission.create_publisher(JointState, "/panda/joint_command", 10)
        self._grip_pub = mission.create_publisher(Float64, "/panda/gripper_command", 10)
        mission._tf()
        for topic in RGB_TOPICS:
            mission.latest(topic, Image, timeout=0.)
        mission.latest("/panda/joint_states", JointState, timeout=0.)

    def _latest(self, topic, kind):
        message = self.mission.latest(topic, kind, timeout=0.)
        if message is not None:
            stamp = stamp_seconds(message)
            if self._stamps.get(topic) != stamp:
                self._stamps[topic] = stamp
                self._received[topic] = time.monotonic()
            if time.monotonic() - self._received.get(topic, 0.) > self.max_age:
                return None
        return message

    def robot_base(self):
        from rclpy.duration import Duration
        from rclpy.time import Time

        errors = []
        for parent in ([self.base_parent] if self.base_parent else ["world", "map"]):
            try:
                message = self.mission._tf().lookup_transform(
                    parent, BASE_FRAME, Time(), timeout=Duration(seconds=.08))
                self.world_from_base = transform_matrix(message)
                self.base_parent = parent
                return self.world_from_base.copy()
            except Exception as error:
                errors.append(str(error))
        raise RuntimeError("Published Panda base TF unavailable: " + "; ".join(errors))

    def observation(self, timeout=3.):
        deadline = time.monotonic() + timeout
        last_error = "waiting for two cameras and Panda joints"
        while time.monotonic() < deadline:
            messages = [self._latest(topic, self._Image) for topic in RGB_TOPICS]
            joints = self._latest("/panda/joint_states", self._JointState)
            if all(message is not None for message in messages) and joints is not None:
                try:
                    times = [stamp_seconds(message) for message in [*messages, joints]]
                    if max(times) - min(times) > self.max_skew:
                        raise ValueError("Panda sensor timestamps are not aligned")
                    mapping = joint_mapping(joints)
                    state = self.kinematics.state(mapping)
                    self.robot_base()
                    images = [rgb_array(message) for message in messages]
                    self.last_joints = mapping
                    return {"time": min(times[:2]), "image": images[0],
                            "wrist_image": images[1], "state": state}
                except Exception as error:
                    last_error = str(error)
            time.sleep(.01)
        raise TimeoutError("Panda observation unavailable: " + last_error)

    def command(self, action):
        world = base_action_to_world(action, self.robot_base())
        self._eef_pub.publish(self._Action(data=world.tolist()))
        return world

    def hold(self):
        """Switch arm to measured-joint hold, preserving measured gripper opening."""
        message = self._latest("/panda/joint_states", self._JointState)
        if message is None:
            return False
        try:
            joints = joint_mapping(message)
            arm = self._JointState()
            arm.name = list(ARM_NAMES)
            arm.position = [joints[name] for name in ARM_NAMES]
            grip = self._Float64(data=float(np.clip(abs(joints[FINGER_NAMES[0]]), 0., .04)))
            for _ in range(3):
                self._joint_pub.publish(arm)
                self._grip_pub.publish(grip)
                time.sleep(.02)
            return True
        except (ValueError, RuntimeError):
            return False
