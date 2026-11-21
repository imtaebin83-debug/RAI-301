"""Optional scan alignment to fixed wall coordinates published in README.

This is a sparse public geometric prior, not a simulator/scenario map. Only
published building/corridor/cliff boundaries are represented. Door openings
are removed; east-door width is not published, so a generous 2.6 m opening is
excluded instead of inventing a narrow wall there. No object truth is read.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def _subtract(low, high, openings):
    intervals = [(low, high)]
    for begin, end in openings:
        new = []
        for a, b in intervals:
            if end <= a or begin >= b:
                new.append((a, b))
            else:
                if a < begin:
                    new.append((a, begin))
                if end < b:
                    new.append((end, b))
        intervals = new
    return intervals


def public_wall_samples():
    # (normal axis 0=x/1=y, fixed coordinate, varying low, varying high).
    segments = [(0, 5.5, 0.0, 26.0), (0, 23.0, 0.0, 26.0),
                (1, 0.0, 5.5, 23.0), (1, 26.0, 5.5, 23.0),
                (1, 10.5, 5.5, 13.0)]
    west_doors = [(9.1 - 0.75, 9.1 + 0.75),
                  (11.375 - 0.6, 11.375 + 0.6),
                  (20.125 - 0.6, 20.125 + 0.6)]
    east_doors = [(y - 1.3, y + 1.3) for y in (3.59, 10.5, 17.5)]
    segments += [(0, 13.0, a, b) for a, b in _subtract(0.0, 21.0, west_doors)]
    segments += [(0, 18.0, a, b) for a, b in _subtract(0.0, 21.0, east_doors)]
    segments += [(1, 21.0, a, b) for a, b in _subtract(13.0, 18.0, [(14.8, 16.2)])]
    points, axes, ids = [], [], []
    for number, (axis, fixed, low, high) in enumerate(segments):
        # Do not use endpoints for matching: uncertain wall thickness and
        # doorway corners should not generate a point obstacle in free space.
        values = np.arange(low + 0.10, high - 0.09, 0.10)
        if not len(values):
            continue
        xy = np.empty((len(values), 2))
        xy[:, axis], xy[:, 1 - axis] = fixed, values
        points.append(xy)
        axes.extend([axis] * len(values))
        ids.extend([number] * len(values))
    return np.vstack(points), np.asarray(axes), np.asarray(ids)


class PublicWallMatcher:
    """Return a bounded incremental SE(2) correction, or identity if unsure."""
    def __init__(self):
        self.points, self.axes, self.ids = public_wall_samples()
        self.tree = cKDTree(self.points)
        self.last_quality = 0.0
        self.last_error_before = None
        self.last_error_after = None
        self.accepted = 0

    def correction(self, world_points, robot_xy):
        identity = np.eye(4)
        self.last_quality = 0.0
        cloud = np.asarray(world_points)
        valid = np.isfinite(cloud).all(axis=1) & (cloud[:, 2] > 0.7) & (cloud[:, 2] < 2.2)
        cloud = cloud[valid, :2]
        if len(cloud) < 80:
            return identity
        # Collapse vertical beams to avoid weighting a wall by its ring count.
        _, keep = np.unique(np.floor(cloud / 0.07).astype(np.int32), axis=0, return_index=True)
        cloud = cloud[np.sort(keep)]
        if len(cloud) < 50:
            return identity
        distance, nearest = self.tree.query(cloud)
        mask = distance < 0.45
        if mask.sum() < 45:
            return identity
        source, targets = cloud[mask], self.points[nearest[mask]]
        axes = self.axes[nearest[mask]]
        # A local line normal must agree with the candidate public wall.
        # This rejects many furniture edges and oblique object surfaces.
        source_tree = cKDTree(cloud)
        _, nn = source_tree.query(source, k=min(9, len(cloud)))
        neighbors = cloud[nn]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]
        agreement = np.abs(normals[np.arange(len(normals)), axes])
        linear = eigenvalues[:, 0] < 0.18 * np.maximum(eigenvalues[:, 1], 1e-8)
        mask = (agreement > 0.94) & linear
        source, targets, axes = source[mask], targets[mask], axes[mask]
        if len(source) < 35:
            return identity
        if max(np.ptp(source[:, 0]), np.ptp(source[:, 1])) < 2.0:
            return identity
        # Balance orthogonal directions so a nearby wall cannot dominate all
        # observations. One direction may still constrain its normal + yaw;
        # the prior keeps translation along an unseen wall unchanged.
        chosen = []
        for axis in (0, 1):
            idx = np.flatnonzero(axes == axis)
            if len(idx) > 180:
                idx = idx[np.linspace(0, len(idx) - 1, 180).astype(int)]
            chosen.extend(idx)
        chosen = np.asarray(chosen, dtype=int)
        source, targets, axes = source[chosen], targets[chosen], axes[chosen]
        center = np.asarray(robot_xy, dtype=float)[:2]
        relative = source - center

        def projected(parameters):
            dx, dy, yaw = parameters
            c, s = math.cos(yaw), math.sin(yaw)
            rot = np.array([[c, -s], [s, c]])
            return relative @ rot.T + center + [dx, dy]

        def errors(parameters, prior=True):
            delta = projected(parameters) - targets
            data = delta[np.arange(len(delta)), axes] / 0.04
            return np.r_[data, np.asarray(parameters) / [0.08, 0.08, math.radians(0.8)]] if prior else data

        before = errors(np.zeros(3), prior=False)
        self.last_error_before = float(np.median(np.abs(before)) * 0.04)
        # Avoid fitting measurement noise when already aligned.
        if self.last_error_before < 0.012:
            return identity
        bound = np.array([0.08, 0.08, math.radians(0.8)])
        fit = least_squares(errors, np.zeros(3), bounds=(-bound, bound), loss="soft_l1", f_scale=1.0,
                            max_nfev=25, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        delta = fit.x.copy()
        norm = np.linalg.norm(delta[:2])
        if norm > 0.08:
            delta[:2] *= 0.08 / norm
        after = errors(delta, prior=False)
        self.last_error_after = float(np.median(np.abs(after)) * 0.04)
        robust_before = float(np.mean(np.minimum(before ** 2, 9.0)))
        robust_after = float(np.mean(np.minimum(after ** 2, 9.0)))
        # Require meaningful improvement and enough inliers. Correction only
        # happens within the deliberately narrow correspondence gate above.
        support = float(np.mean(np.abs(after) < 2.0))
        if (not fit.success or support < 0.50 or robust_after >= robust_before * 0.95
                or self.last_error_after > self.last_error_before - 0.001):
            return identity
        c, s = math.cos(delta[2]), math.sin(delta[2])
        rot = np.array([[c, -s], [s, c]])
        result = np.eye(4)
        result[:2, :2] = rot
        result[:2, 3] = center + delta[:2] - rot @ center
        self.last_quality = support
        self.accepted += 1
        return result
