"""Validate KISS-ICP proposals before changing its scan map.

Uses the public KISS-ICP 1.3 API. The upstream register_frame() updates its
map immediately, so calling it and rejecting a pose afterwards is too late.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class RegistrationRejected(RuntimeError):
    pass


class GuardedKissICP:
    def __init__(self, config):
        from kiss_icp.kiss_icp import KissICP
        self.icp = KissICP(config)
        self.last_pose = np.eye(4)
        self.last_delta = np.eye(4)
        self.last_dt = 0.1
        self.quality = 0.0
        self.metrics = {}

    def register_frame(self, points, timestamps, *, dt=0.1, rotation=None, validator=None):
        started = time.monotonic()
        source, downsample = self.icp.voxelize(points)
        if len(source) < 100:
            raise RegistrationRejected("insufficient downsampled scan support")
        if self.icp.local_map.empty():
            if validator is not None:
                validator(self.last_pose)
            self.icp.local_map.update(downsample, self.last_pose)
            self.quality = 1.0
            self.metrics = {"support": 1.0, "rmse": 0.0, "seconds": time.monotonic() - started}
            return
        if not math.isfinite(dt) or not 0 < dt <= 1.0:
            raise RegistrationRejected(f"scan interval outside 0..1s: {dt}")
        predicted = self.last_pose.copy()
        # KISS's constant-frame delta assumes an unchanged scan period. Scale
        # only an initial guess by actual sensor dt, never publish integration.
        scale = min(3.0, dt / self.last_dt)
        predicted[:3, 3] += self.last_pose[:3, :3] @ (self.last_delta[:3, 3] * scale)
        predicted[:3, :3] = self.last_pose[:3, :3] @ (
            rotation if rotation is not None else
            Rotation.from_rotvec(Rotation.from_matrix(self.last_delta[:3, :3]).as_rotvec() * scale).as_matrix())
        sigma = min(0.25, max(0.06, self.icp.adaptive_threshold.get_threshold()))
        candidate = self.icp.registration.align_points_to_map(
            points=source, voxel_map=self.icp.local_map, initial_guess=predicted,
            max_correspondance_distance=3 * sigma, kernel=sigma)
        if not np.isfinite(candidate).all():
            raise RegistrationRejected("nonfinite ICP proposal")
        translation = float(np.linalg.norm(candidate[:3, 3] - self.last_pose[:3, 3]))
        angle = float(Rotation.from_matrix(self.last_pose[:3, :3].T @ candidate[:3, :3]).magnitude())
        gyro_error = (float(Rotation.from_matrix(predicted[:3, :3].T @ candidate[:3, :3]).magnitude())
                      if rotation is not None else None)
        if translation > max(0.40, 1.6 * dt) or angle > max(0.30, 1.5 * dt):
            raise RegistrationRejected(f"ICP motion jump: {translation:.3f}m/{math.degrees(angle):.1f}deg dt={dt:.3f}s")
        if gyro_error is not None and gyro_error > 0.12:
            raise RegistrationRejected(f"ICP/IMU rotation disagreement: {math.degrees(gyro_error):.1f}deg")
        target = self.icp.local_map.point_cloud()
        moved = source @ candidate[:3, :3].T + candidate[:3, 3]
        distance, _ = cKDTree(target).query(moved, workers=1)
        mask = distance < 0.22
        support = float(mask.mean())
        rmse = float(np.sqrt(np.mean(distance[mask] ** 2))) if mask.any() else 1e9
        self.metrics = {"dt": dt, "translation": translation, "rotation": angle,
                        "gyro_error": gyro_error, "support": support, "rmse": rmse,
                        "source_points": len(source), "map_points": len(target),
                        "seconds": time.monotonic() - started}
        if support < 0.55 or rmse > 0.16:
            raise RegistrationRejected(f"ICP support={support:.2f} rmse={rmse:.3f}m")
        if validator is not None:
            validator(candidate)
        # Commit only after all checks; rejected scans cannot pollute the map,
        # prediction or adaptive threshold. The trusted map survives retries.
        self.icp.adaptive_threshold.update_model_deviation(np.linalg.inv(predicted) @ candidate)
        self.icp.local_map.update(downsample, candidate)
        self.last_delta = np.linalg.inv(self.last_pose) @ candidate
        self.last_pose = candidate
        self.last_dt = dt
        self.quality = support
