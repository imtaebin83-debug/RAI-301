"""LiDAR odometry anchored once to the public simulator startup pose.

Only LiDAR, IMU and robot-link TF are read while moving.  Simulator status
positions are read again only on a *new respawn event*, as README permits.
KISS-ICP runs on CPU; a small SciPy frame-to-frame ICP is the fallback.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, PointCloud2
from custom.inertial import GyroBuffer, stamp_seconds
from custom.icp_registration import GuardedKissICP, RegistrationRejected


def cloud_xyz(message) -> np.ndarray:
    """Decode XYZ with PointCloud2 offsets/stride/endian and row padding."""
    endian = ">" if message.is_bigendian else "<"
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    fields = {f.name: f for f in message.fields}
    if not all(axis in fields for axis in ("x", "y", "z")):
        return np.empty((0, 3))
    dtype = np.dtype({"names": ["x", "y", "z"],
                      "formats": [endian + types[fields[n].datatype] for n in ("x", "y", "z")],
                      "offsets": [fields[n].offset for n in ("x", "y", "z")],
                      "itemsize": message.point_step})
    records = np.ndarray((message.height, message.width), dtype=dtype,
                         buffer=message.data, strides=(message.row_step, message.point_step))
    xyz = np.column_stack([records[n].reshape(-1) for n in ("x", "y", "z")]).astype(np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    return np.ascontiguousarray(xyz[np.isfinite(xyz).all(axis=1) & (radius > 0.65) & (radius < 24.0)])


def transform_matrix(transform) -> np.ndarray:
    t = transform.transform if hasattr(transform, "transform") else transform
    q = t.rotation
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    out[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return out


def floor_height(level_points: np.ndarray) -> tuple[float, float]:
    """Estimate pelvis height from a nearby gravity-aligned floor plane.

    Input coordinates have pelvis as origin and gravity aligned to +z. The
    plane must be 0.3--1.5 m below the pelvis; deeper pits and most tabletop
    surfaces are thereby excluded. No simulator geometry or truth is read.
    """
    radius = np.linalg.norm(level_points[:, :2], axis=1)
    candidate = level_points[(radius > 0.8) & (radius < 5.0)
                             & (level_points[:, 2] > -1.5)
                             & (level_points[:, 2] < -0.3)]
    if len(candidate) < 35:
        raise RuntimeError("insufficient visible floor for sensor height initialization")
    if len(candidate) > 2400:
        candidate = candidate[::int(math.ceil(len(candidate) / 2400))]
    rng = np.random.default_rng(41)
    best, best_count = None, 0
    # A z histogram supplies an inexpensive near-horizontal proposal in
    # addition to RANSAC proposals when the floor dominates the observation.
    counts, bins = np.histogram(candidate[:, 2], bins=np.linspace(-1.5, -0.3, 49))
    peak = int(np.argmax(counts))
    proposals = [(np.array([0.0, 0.0, 1.0]), -float(0.5 * (bins[peak] + bins[peak + 1])))]
    for _ in range(110):
        p = candidate[rng.choice(len(candidate), size=3, replace=False)]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if normal[2] < 0:
            normal *= -1
        if normal[2] < 0.98:
            continue
        offset = -float(normal @ p[0])
        if 0.3 < offset / normal[2] < 1.5:
            proposals.append((normal, offset))
    for normal, offset in proposals:
        mask = np.abs(candidate @ normal + offset) < 0.035
        count = int(mask.sum())
        if count > best_count:
            best, best_count = mask, count
    if best is None or best_count < 30:
        raise RuntimeError("no supported horizontal floor plane")
    floor = candidate[best]
    if np.ptp(floor[:, 0]) < 0.8 or np.ptp(floor[:, 1]) < 0.8:
        raise RuntimeError("floor plane lacks horizontal extent")
    center = floor.mean(axis=0)
    _, _, vt = np.linalg.svd(floor - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal *= -1
    height = float(-normal @ center / normal[2])
    if normal[2] < 0.98 or not 0.3 < height < 1.5:
        raise RuntimeError("invalid sensor-derived pelvis height")
    return height, best_count / len(candidate)


class _ScipyICP:
    """Conservative CPU fallback, not a replacement for a mapped localizer."""
    def __init__(self):
        self.last_pose = np.eye(4)
        self.last_delta = np.eye(4)
        self.previous = None
        self.quality = 0.0

    def register_frame(self, points, timestamps):
        _, keep = np.unique(np.floor(points / 0.12).astype(np.int32), axis=0, return_index=True)
        points = points[np.sort(keep)]
        if len(points) > 4000:
            points = points[::int(math.ceil(len(points) / 4000))]
        if self.previous is None:
            self.previous = points
            self.quality = 1.0
            return
        tree = cKDTree(self.previous)
        estimate = self.last_delta.copy()
        fit = 0.0
        for _ in range(24):
            moved = points @ estimate[:3, :3].T + estimate[:3, 3]
            distance, index = tree.query(moved, workers=1)
            valid = distance < 0.65
            fit = float(np.mean(valid))
            if valid.sum() < 50:
                self.quality = 0.0
                return
            # Trim the farthest 20% of valid pairs to suppress moving limbs.
            valid &= distance <= np.quantile(distance[valid], 0.8)
            a, b = moved[valid], self.previous[index[valid]]
            ca, cb = a.mean(axis=0), b.mean(axis=0)
            u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
            rot = vt.T @ u.T
            if np.linalg.det(rot) < 0:
                vt[-1] *= -1
                rot = vt.T @ u.T
            delta = np.eye(4)
            delta[:3, :3] = rot
            delta[:3, 3] = cb - rot @ ca
            estimate = delta @ estimate
            if np.linalg.norm(delta[:3, 3]) < 0.0005 and np.linalg.norm(rot - np.eye(3)) < 0.001:
                break
        self.last_pose = self.last_pose @ estimate
        self.last_delta = estimate
        self.previous = points
        self.quality = fit


class Localizer:
    def __init__(self, mission, start: dict):
        self.mission = mission
        self.quality = 0.0
        self.last_error = "waiting for LiDAR"
        self.backend = "uninitialized"
        self._stamp = None
        self._last_received = time.monotonic()
        self._last_status_seq = int(start.get("seq", -1))
        self._world_lidar0 = None
        self._pose = (float(start["pos"][0]), float(start["pos"][1]), math.radians(float(start.get("yaw_deg", 0.0))))
        self._anchor = dict(start)
        self._gyro = GyroBuffer()
        self._gyro_rotation = None
        self._attempt_stamp = None
        self.rejected_scans = 0
        self.metrics = {}
        self._odom = self._new_odometry()
        mission._tf()
        mission.latest("/lidar/points", PointCloud2, timeout=0.0)
        mission.latest("/imu", Imu, timeout=0.0)
        if isinstance(self._odom, GuardedKissICP):
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            self._gyro_sub = mission.create_subscription(
                Imu, "/lidar/imu", self._on_gyro,
                QoSProfile(depth=400, reliability=ReliabilityPolicy.BEST_EFFORT),
                callback_group=mission._api_group)
        self._last_transform = np.eye(4)
        self._map_offset = np.eye(4)
        self._recovery_times = []
        self._recovery_blocked = False
        self.recovery_count = 0
        self.map_matcher = None
        if os.environ.get("HAIC_MAP_CORRECTION", "0") == "1":
            from custom.public_map import PublicWallMatcher
            self.map_matcher = PublicWallMatcher()

    @property
    def stale(self) -> bool:
        return self._world_lidar0 is None or time.monotonic() - self._last_received > 1.0 or self.quality < 0.2

    @property
    def T_world_pelvis(self) -> np.ndarray:
        return self._last_transform.copy()

    @property
    def measurement_stamp(self):
        return self._stamp

    def _on_gyro(self, message):
        w = message.angular_velocity
        self._gyro.add(stamp_seconds(message.header.stamp), [w.x, w.y, w.z], message.header.frame_id)

    def _new_odometry(self):
        try:
            from kiss_icp.config import KISSConfig
            config = KISSConfig()
            config.data.max_range = 24.0
            config.data.min_range = 0.65
            config.data.deskew = False  # simulator publishes instantaneous scans
            config.mapping.voxel_size = 0.15
            config.adaptive_threshold.initial_threshold = 0.5
            config.registration.max_num_threads = 2
            config.registration.max_num_iterations = 40
            self.backend = "guarded-kiss-icp"
            return GuardedKissICP(config)
        except (ImportError, AttributeError) as error:
            self.backend = "scipy-icp"
            self.mission.log(f"[localizer] KISS-ICP unavailable, using CPU SciPy ICP: {error}")
            return _ScipyICP()

    def _pelvis_lidar(self, stamp):
        from rclpy.time import Time
        from rclpy.duration import Duration
        return transform_matrix(self.mission._tf().lookup_transform(
            "pelvis", "lidar_link", Time.from_msg(stamp), timeout=Duration(seconds=0.05)))

    def _scan_rotation(self, stamp):
        if self._stamp is None:
            return None
        if self._gyro_rotation is None:
            from rclpy.time import Time
            from rclpy.duration import Duration
            if not self._gyro.frame:
                raise RegistrationRejected("waiting for LiDAR IMU")
            self._gyro_rotation = transform_matrix(self.mission._tf().lookup_transform(
                "lidar_link", self._gyro.frame, Time(), timeout=Duration(seconds=0.03)))[:3, :3]
        begin = self._stamp[0] + self._stamp[1] * 1e-9
        try:
            return self._gyro.rotation(begin, stamp_seconds(stamp), self._gyro_rotation)
        except ValueError as error:
            raise RegistrationRejected(str(error)) from error

    def _validate_motion(self, pose, stamp):
        if not np.isfinite(pose).all():
            raise RegistrationRejected("nonfinite pelvis transform")
        if self._stamp is None:
            return
        dt = stamp[0] - self._stamp[0] + (stamp[1] - self._stamp[1]) * 1e-9
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        translation = float(np.linalg.norm(pose[:2, 3] - self._last_transform[:2, 3]))
        turn = abs((yaw - self._pose[2] + math.pi) % (2 * math.pi) - math.pi)
        if translation > max(.40, 1.6 * dt) or turn > max(.30, 1.5 * dt):
            raise RegistrationRejected(f"pelvis motion jump: {translation:.3f}m/{math.degrees(turn):.1f}deg dt={dt:.3f}s")

    def _initialize(self, pelvis_lidar, points):
        # At rest, accelerometers measure +g in body coordinates. Average unique
        # IMU samples; no fixed pelvis tilt or ground-truth orientation is used.
        gravity, stamps = [], set()
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline and len(gravity) < 25:
            imu = self.mission.latest("/imu", Imu, timeout=0.0)
            if imu is not None:
                stamp = (imu.header.stamp.sec, imu.header.stamp.nanosec)
                a = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z])
                frame = imu.header.frame_id.lstrip("/")
                if frame and frame != "pelvis":
                    from rclpy.time import Time
                    from rclpy.duration import Duration
                    try:
                        pelvis_imu = transform_matrix(self.mission._tf().lookup_transform(
                            "pelvis", frame, Time(), timeout=Duration(seconds=0.02)))
                        a = pelvis_imu[:3, :3] @ a
                    except Exception:
                        time.sleep(0.01)
                        continue
                if stamp not in stamps and 5.0 < np.linalg.norm(a) < 15.0:
                    gravity.append(a)
                    stamps.add(stamp)
            time.sleep(0.01)
        if not gravity:
            raise RuntimeError("no valid IMU gravity for initial attitude")
        ax, ay, az = np.median(gravity, axis=0)
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.hypot(ay, az))
        yaw = math.radians(float(self._anchor.get("yaw_deg", 0.0)))
        world_pelvis = np.eye(4)
        world_pelvis[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        position = self._anchor["pos"]
        world_pelvis[:2, 3] = position[:2]
        # Start status may specify only x/y, or a nominal floor/spawn z. Infer
        # actual pelvis height from the first scan, even when status has z.
        pelvis_points = points @ pelvis_lidar[:3, :3].T + pelvis_lidar[:3, 3]
        level_points = pelvis_points @ world_pelvis[:3, :3].T
        height, self.floor_quality = floor_height(level_points)
        world_pelvis[2, 3] = height  # public startup floor is z=0
        self._world_lidar0 = world_pelvis @ pelvis_lidar
        self.mission.log(f"[localizer] {self.backend}, IMU roll/pitch={math.degrees(roll):.1f}/{math.degrees(pitch):.1f} deg, pelvis height={height:.3f}m floor support={self.floor_quality:.2f}")

    def _recover_registration(self, points, pelvis_lidar, stamp, reason):
        """Discard a corrupt local scan map, retaining the last trusted pose.

        This loses motion since the last trusted scan; it does not correct
        accumulated drift or obtain a fresh absolute pose. No simulator status
        or IMU reinitialization is used in this sensor-only recovery.
        """
        now = time.monotonic()
        self._recovery_times = [t for t in self._recovery_times if now - t < 30.0]
        if self._recovery_blocked or len(self._recovery_times) >= 2:
            self._recovery_blocked = True
            self.quality = 0.0
            self.last_error = "LiDAR recovery limit reached (2 per 30s); stopped until respawn/restart"
            self.mission.log(f"[localizer] {self.last_error}")
            return self._pose
        trusted = self._last_transform.copy()
        if not np.isfinite(trusted).all() or not np.isfinite(pelvis_lidar).all():
            self._recovery_blocked = True
            self.quality = 0.0
            self.last_error = "nonfinite trusted pose/extrinsic; cannot recover registration"
            return self._pose
        odometry = self._new_odometry()
        odometry.register_frame(points, np.empty(0, dtype=np.float64))
        if not np.isfinite(odometry.last_pose).all():
            self._recovery_blocked = True
            self.quality = 0.0
            self.last_error = "nonfinite first scan after registration recovery"
            return self._pose
        self._odom = odometry
        # The new local map starts at this scan. Preserve exactly the last
        # trusted pelvis transform instead of accepting the rejected jump.
        self._world_lidar0 = trusted @ pelvis_lidar @ np.linalg.inv(odometry.last_pose)
        self._map_offset = np.eye(4)
        self._stamp = stamp
        self._last_received = now
        self.quality = 0.5
        self.recovery_count += 1
        self._recovery_times.append(now)
        self.last_error = "sensor scan-map reanchored at last trusted pose; lost motion remains uncertain"
        self.mission.log(f"[localizer] sensor reanchor #{self.recovery_count}: {reason}; "
                         "retained last trusted pose, no absolute relocalization")
        return self._pose

    def update(self) -> tuple[float, float, float]:
        status = self.mission.last_status() or {}
        seq = int(status.get("seq", -1))
        if seq > self._last_status_seq:
            self._last_status_seq = seq
            if status.get("event") == "respawned" and status.get("pos") is not None:
                self._anchor = dict(status)
                self._world_lidar0 = None
                self._odom = self._new_odometry()
                self._stamp = None
                self._attempt_stamp = None
                self._gyro_rotation = None
                self.quality = 0.0
                self._last_transform = np.eye(4)
                self._map_offset = np.eye(4)
                self._recovery_times = []
                self._recovery_blocked = False
        if self._recovery_blocked:
            self.quality = 0.0
            return self._pose
        message = self.mission.latest("/lidar/points", PointCloud2, timeout=0.0)
        if message is None:
            return self._pose
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if stamp == self._stamp or stamp == self._attempt_stamp:
            return self._pose
        if self._stamp is not None and stamp < self._stamp:
            self.quality = 0.0
            self.last_error = "out-of-order LiDAR timestamp; waiting for newer scan"
            return self._pose
        try:
            if message.header.frame_id.lstrip("/") != "lidar_link":
                raise RuntimeError("unexpected LiDAR frame: " + message.header.frame_id)
            pelvis_lidar = self._pelvis_lidar(message.header.stamp)
            points = cloud_xyz(message)
            if len(points) < 100:
                self.quality = 0.0
                self.last_error = "insufficient LiDAR points"
                return self._pose
            if (isinstance(self._odom, GuardedKissICP) and self._stamp is None
                    and not self._gyro.covers(stamp_seconds(message.header.stamp))):
                raise RegistrationRejected("waiting for IMU coverage of initial scan")
            if self._world_lidar0 is None:
                self._initialize(pelvis_lidar, points)
            if isinstance(self._odom, GuardedKissICP):
                dt = (stamp[0] - self._stamp[0] + (stamp[1] - self._stamp[1]) * 1e-9
                      if self._stamp is not None else 0.1)
                rotation = self._scan_rotation(message.header.stamp)
                def validate_pose(candidate):
                    self._validate_motion(self._map_offset @ self._world_lidar0 @ candidate
                                          @ np.linalg.inv(pelvis_lidar), stamp)
                self._odom.register_frame(points, np.empty(0, dtype=np.float64), dt=dt,
                                          rotation=rotation, validator=validate_pose)
                self.metrics = dict(self._odom.metrics)
            else:
                self._odom.register_frame(points, np.empty(0, dtype=np.float64))
            raw_world_lidar = self._world_lidar0 @ self._odom.last_pose
            corrected_world_lidar = self._map_offset @ raw_world_lidar
            pose = corrected_world_lidar @ np.linalg.inv(pelvis_lidar)
            if self.map_matcher is not None:
                world_points = points @ corrected_world_lidar[:3, :3].T + corrected_world_lidar[:3, 3]
                correction = self.map_matcher.correction(world_points, pose[:2, 3])
                self._map_offset = correction @ self._map_offset
                pose = correction @ pose
            yaw = math.atan2(pose[1, 0], pose[0, 0])
            if self._stamp is not None:
                dt=max(.01,stamp[0]-self._stamp[0]+(stamp[1]-self._stamp[1])*1e-9)
                translation=np.linalg.norm(pose[:2,3]-self._last_transform[:2,3])
                turn=abs((yaw-self._pose[2]+math.pi)%(2*math.pi)-math.pi)
                if (not np.isfinite(pose).all() or translation>max(.40,1.6*dt)
                        or turn>max(.30,1.5*dt)):
                    if isinstance(self._odom, GuardedKissICP):
                        self._recovery_blocked = True
                        raise RegistrationRejected("post-correction pelvis motion jump")
                    return self._recover_registration(
                        points, pelvis_lidar, stamp,
                        f"rejected translation={translation:.3f}m turn={math.degrees(turn):.1f}deg dt={dt:.3f}s")
            self._pose = (float(pose[0, 3]), float(pose[1, 3]), yaw)
            self._last_transform = pose
            self.quality = float(getattr(self._odom, "quality", 1.0))
            self._last_received = time.monotonic()
            self._stamp = stamp
            self._attempt_stamp = stamp
            self.last_error = ""
        except RegistrationRejected as error:
            self.rejected_scans += 1
            self._attempt_stamp = stamp
            self.last_error = str(error)
            self.quality = 0.0
            if self._stamp is not None:
                elapsed = stamp[0] - self._stamp[0] + (stamp[1] - self._stamp[1]) * 1e-9
                if elapsed > 1.0:
                    self._recovery_blocked = True
                    self.last_error += '; registration gap exceeded 1s; stopped until respawn/restart'
            # Retain the map and the last measurement. Navigator stops while
            # fresh scans retry registration against this same trusted map.
            self.metrics = dict(self._odom.metrics)
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            self.quality = 0.0
        return self._pose

    def transform(self) -> np.ndarray:
        """Estimated world-from-pelvis transform for sensor target projection."""
        return self._last_transform.copy()
