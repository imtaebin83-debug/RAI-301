"""Timestamped raw gyroscope integration; no commanded velocity or truth input."""
from __future__ import annotations

from collections import deque
import threading

import numpy as np
from scipy.spatial.transform import Rotation


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class GyroBuffer:
    """Bounded history in one IMU frame, integrated over the *scan* interval.

    A missing endpoint or sample gap is an error, never an invitation to
    extrapolate rotation indefinitely. Translation is always estimated by ICP.
    """

    def __init__(self, max_samples=6000, max_gap=0.08):
        self._samples = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self.max_gap = max_gap
        self.frame = None
        self.dropped = 0

    def add(self, stamp, omega, frame):
        omega = np.asarray(omega, dtype=float)
        if (not np.isfinite(stamp) or omega.shape != (3,)
                or not np.isfinite(omega).all() or np.linalg.norm(omega) > 20):
            return False
        frame = frame.lstrip("/")
        if not frame:
            return False
        with self._lock:
            if self.frame is None:
                self.frame = frame
            if frame != self.frame or (self._samples and stamp <= self._samples[-1][0]):
                self.dropped += 1
                return False
            self._samples.append((float(stamp), omega.copy()))
        return True

    def covers(self, stamp):
        with self._lock:
            return bool(len(self._samples) >= 2 and self._samples[0][0] <= stamp <= self._samples[-1][0])

    def rotation(self, begin, end, lidar_from_imu):
        if not np.isfinite([begin, end]).all() or end <= begin:
            raise ValueError("non-increasing gyroscope integration interval")
        with self._lock:
            samples = list(self._samples)
        if len(samples) < 2 or samples[0][0] > begin or samples[-1][0] < end:
            raise ValueError("IMU samples do not bracket scan timestamps")
        times = np.asarray([s[0] for s in samples])
        lo = max(0, int(np.searchsorted(times, begin, side="right")) - 1)
        hi = int(np.searchsorted(times, end, side="left"))
        times = times[lo:hi + 1]
        omega = np.asarray([s[1] for s in samples[lo:hi + 1]])
        if np.any(np.diff(times) > self.max_gap):
            raise ValueError("IMU gap exceeds integration limit")
        knots = np.r_[begin, times[(times > begin) & (times < end)], end]
        rates = np.column_stack([np.interp(knots, times, omega[:, axis]) for axis in range(3)])
        rates = rates @ np.asarray(lidar_from_imu).T
        result = np.eye(3)
        for dt, a, b in zip(np.diff(knots), rates[:-1], rates[1:]):
            result = result @ Rotation.from_rotvec(0.5 * (a + b) * dt).as_matrix()
        return result
