"""Pure camera calibration and RGB object measurements; no teacher control logic.

Reads only RGB images and robot proprioception. Geometry is fixed object size,
camera calibration and table height, never simulator object poses or scenarios.
The collector must have the unmodified referee verify completed demonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class CameraModel:
    k: np.ndarray
    base_from_optical: np.ndarray

    def __post_init__(self):
        self.k = np.asarray(self.k, dtype=float).reshape(3, 3)
        self.base_from_optical = np.asarray(self.base_from_optical, dtype=float).reshape(4, 4)
        if not np.isfinite(self.k).all() or not np.isfinite(self.base_from_optical).all():
            raise ValueError('Camera calibration must be finite')
        if min(self.k[0, 0], self.k[1, 1]) <= 0:
            raise ValueError('Camera focal lengths must be positive')

    @classmethod
    def from_pose(cls, k, position, quaternion_wxyz, *, optical=False):
        """Camera metadata pose is MuJoCo/GL unless optical=True is explicit."""
        q=np.asarray(quaternion_wxyz,dtype=float)
        if q.shape!=(4,):
            raise ValueError('Expected quaternion [w,x,y,z]')
        t=np.eye(4)
        t[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        if not optical:
            t[:3,:3]=t[:3,:3]@np.diag([1.,-1.,-1.])
        t[:3,3]=position
        return cls(k,t)

    @classmethod
    def panda_agentview_default(cls):
        """Public, fixed sensors.yaml calibration; overridden cameras need their K/T."""
        focal = 128.0 / math.tan(math.radians(46.0) / 2)
        k = np.array([[focal, 0, 127.5], [0, focal, 127.5], [0, 0, 1.]])
        # MuJoCo cameras look down -Z with +Y up; ROS/pixels use +Z, +Y down.
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat([.24496836, .24496836, .66331780, .66331780]).as_matrix() @ np.diag([1., -1., -1.])
        t[:3, 3] = [1.28, 0., .92]
        return cls(k, t)

    def on_plane(self, pixels, height):
        pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
        rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(self.k).T
        directions = rays @ self.base_from_optical[:3, :3].T
        origin = self.base_from_optical[:3, 3]
        if np.any(np.abs(directions[:, 2]) < 1e-8):
            raise ValueError('Ray is parallel to the table plane')
        scale = (float(height) - origin[2]) / directions[:, 2]
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError('Plane intersection lies behind camera')
        return origin + scale[:, None] * directions


@dataclass
class SceneMeasurement:
    pencil_xy: np.ndarray
    pencil_yaw: float
    holder_xy: np.ndarray
    pencil_pixel: tuple[float, float]
    holder_pixel: tuple[float, float]
    pencil_area: int
    holder_area: int

    def as_dict(self):
        return {'pencil_xy': self.pencil_xy.tolist(), 'pencil_yaw': self.pencil_yaw,
                'holder_xy': self.holder_xy.tolist(), 'pencil_pixel': self.pencil_pixel,
                'holder_pixel': self.holder_pixel, 'pencil_area': self.pencil_area,
                'holder_area': self.holder_area}


def measure_scene(image, camera: CameraModel, table_z=0., holder_height=.15,
                  pencil_radius=.0145) -> SceneMeasurement:
    """Find the yellow pencil and gray square holder from their visible pixels."""
    image = np.asarray(image)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError('Expected unrotated 256x256 RGB agentview')
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pencil = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) &
              (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 155)).astype(np.uint8)
    legacy_pencil = pencil.astype(bool) & (hsv[:, :, 0] >= 20)
    # Connect metal bands splitting yellow paint, but measure original pixels.
    expanded = cv2.dilate(pencil, np.ones((3, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(expanded, 8)
    choices = []
    for label in range(1, count):
        rows, cols = np.nonzero((labels == label) & pencil.astype(bool))
        if len(rows) < 30 or np.ptp(cols) < 15:
            continue
        points = camera.on_plane(np.c_[cols, rows], table_z + pencil_radius * 1.7)
        center = points[:, :2].mean(0)
        _, singular, vt = np.linalg.svd(points[:, :2] - center, full_matrices=False)
        if singular[0] < 3 * max(singular[1], 1e-8):
            continue
        # Public arm reach is a validity check, not a task object-position prior.
        if not .15 < np.linalg.norm(center) < .95:
            continue
        axis = vt[0]
        # Restore H18/H19 body pixels without changing the arbitrary directed
        # PCA branch consumed by the learned policy. Reference only H20 pixels
        # belonging to this same H18 component, centered at their own mean.
        old_rows, old_cols = np.nonzero((labels == label) & legacy_pencil)
        if len(old_rows) >= 30:
            old_points = camera.on_plane(np.c_[old_cols, old_rows],
                                         table_z + pencil_radius * 1.7)[:, :2]
            _, _, old_vt = np.linalg.svd(old_points - old_points.mean(0),
                                        full_matrices=False)
            if np.dot(axis, old_vt[0]) < 0:
                axis = -axis
        # Resolve the otherwise pi-ambiguous PCA axis with the pink eraser.
        # The desired directed axis points from eraser toward graphite tip.
        pink=(hsv[:,:,0]>145)&(hsv[:,:,0]<179)&(hsv[:,:,1]>70)&(hsv[:,:,2]>90)
        pr,pc=np.nonzero(pink)
        nearby=(pr>=rows.min()-12)&(pr<=rows.max()+12)&(pc>=cols.min()-18)&(pc<=cols.max()+18)
        pr,pc=pr[nearby],pc[nearby]
        if len(pr):
            projected=camera.on_plane(np.c_[pc,pr],table_z+pencil_radius*1.7)[:,:2]
            along=(projected-center)@axis
            perpendicular=np.linalg.norm(projected-center-along[:,None]*axis,axis=1)
            candidates=(perpendicular<.022)&(np.abs(along)>.06)&(np.abs(along)<.20)
            if candidates.sum()>=2:
                eraser_direction=float(np.median(along[candidates]))
                if eraser_direction>0:
                    axis=-axis
        yaw = math.atan2(axis[1], axis[0])
        choices.append((len(rows), center, yaw, (float(cols.mean()), float(rows.mean()))))
    if not choices:
        raise ValueError('No elongated yellow pencil in calibrated workspace')
    pa, pxy, yaw, ppixel = max(choices, key=lambda row: row[0])

    gray = ((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 160)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gray, 8)
    cups = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < 120 or min(w, h) < 12 or not .45 < w/h < 1.6:
            continue
        if x < 2 or y < 3 or x+w > 254 or y+h > 254:
            continue
        # The open rim occupies the upper part of the visible fixed-height cup.
        # This is an image-relative measurement; no random object position is used.
        pixel = (x + (w-1)*.5, y + (h-1)*.25)
        xy = camera.on_plane([pixel], table_z + holder_height)[0, :2]
        if .15 < np.linalg.norm(xy) < .95 and np.linalg.norm(xy-pxy) > .12:
            cups.append((area, xy, pixel))
    if not cups:
        raise ValueError('No gray holder rim in calibrated workspace')
    ca, cxy, cpixel = max(cups, key=lambda row: row[0])
    return SceneMeasurement(pxy, yaw, cxy, ppixel, cpixel, pa, ca)
