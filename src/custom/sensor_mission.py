"""Public-sensor helpers for the 11/26 Mission API.

The official runner owns ROS and privileged simulator commands. This node only
caches documented sensors and team-derived /custom/ topics; it never restores
the old mission-status subscription or accesses the runner's private SimLink.
"""
from __future__ import annotations

import math
import threading
import time

from mission_api import Mission as OfficialMission


SENSOR_TOPICS = frozenset({
    "/lidar/points",
    "/camera/color/image_raw", "/camera/depth/image_raw", "/camera/camera_info",
    "/imu", "/camera/imu", "/lidar/imu",
    "/joint_states", "/panda/joint_states",
    "/tf", "/tf_static", "/clock", "/g1/respawned",
    "/panda/agentview/image_raw", "/panda/wrist_camera/image_raw",
})


def check_sensor_topic(topic: str) -> str:
    """Reject non-public topics before creating any ROS subscription."""
    if not isinstance(topic, str):
        raise ValueError("Sensor topic must be an absolute ROS topic name")
    parts = topic.split("/")
    derived = (topic.startswith("/custom/") and len(parts) > 2
               and all(part and part not in (".", "..") for part in parts[1:]))
    if topic not in SENSOR_TOPICS and not derived:
        raise ValueError("Topic is not a permitted sensor or team-derived /custom/ topic: " + topic)
    return topic


class SensorMission(OfficialMission):
    """Official Mission with a bounded, thread-safe public sensor cache."""

    def __init__(self, node_name=None):
        super().__init__(node_name)
        from std_msgs.msg import Empty

        self._sensor_cv = threading.Condition()
        self._sensor_subs = {}
        self._sensor_types = {}
        self._sensor_messages = {}
        self._respawn_count = 0
        # An Empty notification means the documented fixed start (16, 11), 0°.
        # No pose or scenario information comes from this notification.
        self._sensor_types["/g1/respawned"] = Empty
        self._sensor_subs["/g1/respawned"] = self.create_subscription(
            Empty, "/g1/respawned", self._on_public_respawn, 10,
            callback_group=self._api_group)

    def _on_public_respawn(self, message):
        with self._sensor_cv:
            self._respawn_count += 1
            self._sensor_messages["/g1/respawned"] = message
            self._sensor_cv.notify_all()

    @property
    def respawn_count(self):
        """Public notifications observed during this node's lifetime."""
        with self._sensor_cv:
            return self._respawn_count

    @property
    def gap_falls(self):
        """Legacy navigation name: now counts every observed public respawn."""
        return self.respawn_count

    def latest(self, topic, msg_type, timeout=5.0, qos=None):
        """Return cached data, or wait at most timeout for its first arrival.

        Subscriptions are created once and run on the official runner's
        executor. timeout=0 is nonblocking. Message freshness remains the
        caller's responsibility (for example the Panda timestamp checks).
        """
        topic = check_sensor_topic(topic)
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Sensor wait timeout must be finite and nonnegative")
        with self._sensor_cv:
            if topic in self._sensor_types and self._sensor_types[topic] is not msg_type:
                raise TypeError("Sensor topic already subscribed with another message type: " + topic)
            if topic not in self._sensor_subs:
                from rclpy.qos import (DurabilityPolicy, QoSProfile,
                                      ReliabilityPolicy, qos_profile_sensor_data)

                if qos is None:
                    qos = (QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
                           if topic == "/tf_static" else qos_profile_sensor_data)

                def store(message):
                    with self._sensor_cv:
                        self._sensor_messages[topic] = message
                        self._sensor_cv.notify_all()

                self._sensor_types[topic] = msg_type
                self._sensor_subs[topic] = self.create_subscription(
                    msg_type, topic, store, qos, callback_group=self._api_group)
            deadline = time.monotonic() + timeout
            while topic not in self._sensor_messages:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    return None
                self._sensor_cv.wait(timeout=remaining)
            return self._sensor_messages[topic]


# Mission files may keep their existing class MissionN(Mission) declaration.
Mission = SensorMission
