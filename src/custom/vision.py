"""RGB-D object observations; no scene files or simulator truth are read."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class Detection:
    x: float
    y: float
    distance: float
    pixel: tuple[float, float]
    area: float

    def valid(self):
        return bool(np.isfinite([self.x,self.y,self.distance,self.area,*self.pixel]).all()
                    and self.distance>0 and self.area>0)


BALL_RADIUS = .11          # published soccer-ball radius (README)
PITCH_XY = (5.6, 12.95, .55, 10.45)   # public soccer-room bounds incl. goal mouth
WALL_X = (5.5, 13.)                    # soccer-room walls (README coordinate table)
WALL_Y_MAX = 10.5


@dataclass
class BallDetection(Detection):
    score: float = 0.
    rms: float = float('nan')
    top: float = float('nan')
    white: float = 0.
    dark: float = 0.


def _fit_sphere_center(points, start, radius=BALL_RADIUS, iterations=8):
    """Fixed-radius least-squares sphere centre (Gauss-Newton)."""
    centre = np.asarray(start, dtype=np.float64).copy()
    for _ in range(iterations):
        offset = points-centre
        norm = np.linalg.norm(offset, axis=1)
        norm = np.where(norm < 1e-6, 1e-6, norm)
        residual = norm-radius
        jacobian = -offset/norm[:, None]
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        centre += np.clip(step, -.1, .1)
        if np.linalg.norm(step) < 1e-4:
            break
    residual = np.linalg.norm(points-centre, axis=1)-radius
    return centre, float(np.sqrt(np.mean(residual**2)))


def _grass_height(world, height, grass, iterations=60, tolerance=.015):
    """Height above a plane re-fitted to visible grass (robust to the shared
    floor estimate's per-frame jitter and to pose pitch error)."""
    samples = world[grass]
    if len(samples) < 300:
        return height
    rng = np.random.default_rng(len(samples))
    if len(samples) > 6000:
        samples = samples[rng.choice(len(samples), 6000, replace=False)]
    best, best_count = None, 0
    for _ in range(iterations):
        tri = samples[rng.choice(len(samples), 3, replace=False)]
        n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        size = np.linalg.norm(n)
        if size < 1e-9:
            continue
        n /= size
        if n[2] < 0:
            n = -n
        if n[2] < .9:
            continue
        count = int((np.abs((samples-tri[0])@n) < tolerance).sum())
        if count > best_count:
            best, best_count = (n, tri[0]), count
    if best is None or best_count < .3*len(samples):
        return height
    n, point = best
    inliers = samples[np.abs((samples-point)@n) < tolerance]
    centre = inliers.mean(0)
    _, _, vt = np.linalg.svd(inliers-centre, full_matrices=False)
    n = vt[-1] if vt[-1][2] > 0 else -vt[-1]
    return (world-centre)@n


def detect_balls(rgb, depth, world, height, valid, fx, viewer_xy=None, reasons=None):
    """Soccer-ball candidates from one RGB-D frame (pure function, no ROS).

    A candidate is a compact low-saturation (white/black) object standing on
    the floor: its 3D size and image size match a 0.11 m sphere, its top is
    below ~0.27 m, nothing tall continues directly above it (posts, net,
    walls), and its depth points fit a sphere of the published radius.
    Flat floor markings and the placement ring fail the height test.
    ``reasons`` (optional list) collects why sizeable components were rejected.
    """
    def reject(why, **info):
        if reasons is not None and area >= 40:
            reasons.append(dict(why=why, pixel=tuple(int(v) for v in centroids[label]), area=area, **info))
    finite = valid & np.isfinite(world).all(-1) & np.isfinite(depth) & np.isfinite(height)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    x0, x1, y0, y1 = PITCH_XY
    inside = (world[:, :, 0] > x0) & (world[:, :, 0] < x1) & (world[:, :, 1] > y0) & (world[:, :, 1] < y1)
    # Grass hue ~50-65; the teal walls (hue 86-87, sat ~78) must not join the fit.
    grass = finite & (sat > 60) & (hue >= 35) & (hue <= 80) & (np.abs(height) < .2)
    height = _grass_height(world, height, grass)
    # Geometry comes from depth alone: the colour and depth images can be
    # captured at slightly different head poses (the G1 steps in place), so
    # colour is only checked in a margin around the depth blob. White panels
    # are unsaturated and black panels very dark; saturated grass is neither.
    white_px = (sat < 40) & (val > 140)
    dark_px = val < 45
    low = finite & inside & (height > .03) & (height < .30)
    tall = finite & (height > .32) & (height < 3.)
    mask = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool) & finite
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    rows, cols_n = depth.shape
    out = []
    for label in range(1, count):
        left, top_row, width, tall_px, area = (int(v) for v in stats[label])
        if area < 12 or min(width, tall_px) < 3:
            continue
        selected = labels == label
        d = float(np.median(depth[selected]))
        if not (.3 < d < 11.):
            reject('range', d=d)
            continue
        expected = 2*BALL_RADIUS*fx/d
        at_bottom = top_row+tall_px >= rows-2
        at_side = left <= 1 or left+width >= cols_n-1     # ball cut by the image edge
        if width > 1.6*expected or tall_px > 1.6*expected:
            reject('too large', w=width, h=tall_px, expected=expected)
            continue
        if not at_side and width < .45*expected:
            reject('too narrow', w=width, expected=expected)
            continue
        if not at_bottom and tall_px < .45*expected:
            reject('too short', h=tall_px, expected=expected)
            continue
        points = world[selected].astype(np.float64)
        h = height[selected]
        horizontal = float(np.max(np.ptp(points[:, :2], axis=0)))
        top = float(np.percentile(h, 97))
        if horizontal > .34 or top > .25 or top < .10:
            reject('3d size', horizontal=horizontal, top=top)
            continue
        # Anything tall immediately above, at a similar depth, means a post,
        # net, wall or person rather than a free-standing ball.
        band = slice(max(0, top_row-max(4, int(.6*expected))), top_row)
        cols = slice(left, left+width)
        above = tall[band, cols] & (np.abs(depth[band, cols]-d) < .5)
        if above.sum() > .08*width*max(1, band.stop-band.start):
            reject('tall above')
            continue
        margin = int(max(12, .8*expected))
        window = (slice(max(0, top_row-margin), min(rows, top_row+tall_px+margin)),
                  slice(max(0, left-margin), min(cols_n, left+width+margin)))
        ball_area = max(float(area), .5*expected**2)
        white = min(1., float(white_px[window].sum())/ball_area)
        dark = min(1., float(dark_px[window].sum())/ball_area)
        median = np.median(points, axis=0)
        # The centre lies behind the visible cap. Several horizontal starting
        # offsets avoid the inside-out local minimum when the viewer position
        # (pose estimate) is wrong; the best fit wins.
        offsets = [np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2*math.pi, 8, endpoint=False)]
        if viewer_xy is not None:
            away = median[:2]-np.asarray(viewer_xy, dtype=np.float64)
            if np.linalg.norm(away) > 1e-6:
                offsets.insert(0, away/np.linalg.norm(away))
        if dark < .04 or white < .15:   # a soccer ball shows both panels
            reject('panels', white=white, dark=dark)
            continue
        centre, rms = None, np.inf
        for offset in offsets:
            start = median.copy()
            start[:2] += offset*BALL_RADIUS*.6
            c, e = _fit_sphere_center(points, start)
            if np.isfinite(c).all() and e < rms:
                centre, rms = c, e
            if rms < .004:
                break
        if centre is None or rms > .015 or np.linalg.norm(centre[:2]-median[:2]) > .25:
            reject('sphere fit', rms=rms)
            continue
        # A ball centre is at least one radius from the walls; the corner
        # flag bases (black foot + white pole) sit in the pitch corners.
        cx, cy = float(centre[0]), float(centre[1])
        if not (WALL_X[0]+BALL_RADIUS-.03 < cx < WALL_X[1]-BALL_RADIUS+.03 and cy < WALL_Y_MAX-BALL_RADIUS+.03) or \
                min(math.hypot(cx-kx, cy-ky) for kx in WALL_X for ky in (0., WALL_Y_MAX)) < .45:
            reject('wall/corner', x=cx, y=cy)
            continue
        size_fit = 1.-min(1., abs(max(width, tall_px if not at_bottom else width)/expected-1.))
        score = (2.*min(white, .6) + 1.5*min(dark, .25) + size_fit + (1.-rms/.015)
                 + .5*min(1., area/(.6*expected**2)))
        result = BallDetection(float(centre[0]), float(centre[1]), d, tuple(centroids[label]), float(area),
                               score=float(score), rms=rms, top=top, white=white, dark=dark)
        if result.valid():
            out.append(result)
    return out


def _image(msg, depth=False):
    if depth:
        dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
        if msg.encoding.upper() == '16UC1':
            dtype = np.dtype('>u2' if msg.is_bigendian else '<u2')
        view = np.ndarray((msg.height, msg.width), dtype=dtype,
                          buffer=msg.data, strides=(msg.step, dtype.itemsize)).astype(np.float32)
        return view * .001 if msg.encoding.upper() == '16UC1' else view
    channels = 4 if msg.encoding.lower() in ('rgba8', 'bgra8') else 3
    view = np.ndarray((msg.height, msg.width, channels), dtype=np.uint8,
                      buffer=msg.data, strides=(msg.step, channels, 1))[:, :, :3].copy()
    return view[:, :, ::-1] if msg.encoding.lower().startswith('bgr') else view


class RGBD:
    def __init__(self, mission, nav):
        self.mission, self.nav = mission, nav
        self._stamp = None
        self._received_at = 0.
        self._cache = None
        self._rng = np.random.default_rng(301)
        self._floor_z = 0.
        self._floor_normal = np.array([0.,0.,1.])

    def frame(self, timeout=4.):
        m = self.mission
        rgb = m.latest('/camera/color/image_raw', Image, timeout=timeout)
        dep = m.latest('/camera/depth/image_raw', Image, timeout=timeout)
        info = m.latest('/camera/camera_info', CameraInfo, timeout=timeout)
        if rgb is None or dep is None or info is None:
            return None
        stamp = (dep.header.stamp.sec, dep.header.stamp.nanosec)
        if stamp == self._stamp:
            return self._cache if time.monotonic()-self._received_at<.7 else None
        # Account for simulation time through the /clock-sourced node clock.
        if abs(rgb.header.stamp.sec + rgb.header.stamp.nanosec*1e-9 -
               stamp[0] - stamp[1]*1e-9) > .25:
            return None
        image, depth = _image(rgb), _image(dep, True)
        if image.shape[:2] != depth.shape:
            return None
        fy, fx, cy, cx = info.k[4], info.k[0], info.k[5], info.k[2]
        if min(fx, fy) <= 0:
            return None
        rows, cols = np.indices(depth.shape, dtype=np.float32)
        optical = np.stack(((cols-cx)*depth/fx, (rows-cy)*depth/fy, depth), axis=-1)
        valid = np.isfinite(optical).all(-1) & (depth > .15) & (depth < 12.)
        try:
            world = np.asarray(self.nav.camera_to_world(optical.reshape(-1,3), dep.header.frame_id)).reshape(optical.shape)
        except Exception:
            return None
        valid &= np.isfinite(world).all(-1)
        # Re-estimate the floor from measured depth. This rejects the flat white
        # placement ring while retaining ball pixels above the floor.
        samples = world[::10, ::10][valid[::10, ::10]]
        normal, intercept, best = self._floor_normal.copy(), -self._floor_z, 0
        if len(samples)>100:
            for _ in range(45):
                tri = samples[self._rng.choice(len(samples), 3, replace=False)]
                n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
                size = np.linalg.norm(n)
                if size<1e-6:
                    continue
                n /= size
                if n[2]<0:
                    n *= -1
                d = -n @ tri[0]
                # Odometry height can drift while XY remains useful. Estimate
                # height *above the observed floor*, rather than rejecting a
                # floor merely because its map z is no longer near zero.
                if n[2]<.88 or abs(d)>3.:
                    continue
                count = int((np.abs(samples@n+d)<.025).sum())
                if count>best:
                    normal, intercept, best = n, float(d), count
            self._floor_z = -intercept
            self._floor_normal = normal.copy()
        height = world @ normal + intercept
        self._stamp = stamp
        self._received_at = time.monotonic()
        self._cache = (image, depth, world, height, valid)
        return self._cache

    def cone(self, color):
        data = self.frame()
        if data is None:
            return None
        rgb, depth, world, height, valid = data
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        bands = {'blue': (92,135), 'green': (35,90), 'orange': (3,28)}
        if color not in bands:
            return None
        low, high = bands[color]
        mask = (valid & (hsv[:,:,0]>=low) & (hsv[:,:,0]<=high) &
                (hsv[:,:,1]>95) & (hsv[:,:,2]>45) & (height>.08) &
                (world[:,:,1]>21.2) & (world[:,:,1]<26.) &
                (world[:,:,0]>13.) & (world[:,:,0]<18.))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        choices = []
        for label in range(1,count):
            _, _, width, tall, area = stats[label]
            if area<16 or tall<width*.65:
                continue
            selected = (labels==label) & valid & np.isfinite(world).all(-1) & np.isfinite(depth)
            if selected.sum()<16:
                continue
            xy = np.median(world[selected,:2], axis=0)
            result = Detection(float(xy[0]),float(xy[1]),float(np.median(depth[selected])),
                               tuple(centroids[label]),float(selected.sum()))
            if result.valid():
                choices.append(result)
        return max(choices,key=lambda d:d.area,default=None)

    def _capture_ball_frame(self, data, result):
        """Debug-only RGB-D dump for offline ball replay (RAI_M6_CAPTURE=dir)."""
        directory = os.environ.get('RAI_M6_CAPTURE')
        if not directory or self._stamp == getattr(self, '_captured_stamp', None):
            return
        now = time.monotonic()
        count = getattr(self, '_captured_count', 0)
        if count >= 400 or now-getattr(self, '_captured_at', 0.) < .25:
            return
        try:
            info = self.mission.latest('/camera/camera_info', CameraInfo, timeout=0.)
            rgb, depth, world, height, valid = data
            pose = np.asarray(self.nav.localizer.T_world_pelvis, dtype=np.float64)
            Path(directory).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                Path(directory)/f'frame_{count:04d}.npz', rgb=rgb, depth=depth.astype(np.float32),
                world=world.astype(np.float32), height=height.astype(np.float32), valid=valid,
                K=np.asarray(info.k if info is not None else [np.nan]*9, dtype=np.float64),
                T_world_pelvis=pose, floor_z=self._floor_z, floor_normal=self._floor_normal,
                stamp=np.asarray(self._stamp, dtype=np.int64), wall=time.time(),
                chosen=np.asarray([result.x, result.y, result.distance, *result.pixel, result.area]
                                  if result is not None else [np.nan]*6))
            self._captured_stamp, self._captured_at, self._captured_count = self._stamp, now, count+1
        except Exception as exc:  # never let diagnostics stop a mission
            self.mission.log(f'[vision] capture skipped: {exc}')

    def ball(self, prior=None, gate=None):
        """Best soccer-ball candidate in the newest RGB-D frame.

        ``prior``/``gate`` (world xy, metres) restrict tracking to candidates
        near a previously observed ball; first acquisition passes neither.
        """
        data = self.frame()
        if data is None:
            return None
        info = self.mission.latest('/camera/camera_info', CameraInfo, timeout=0.)
        fx = float(info.k[0]) if info is not None and info.k[0] > 0 else None
        viewer = None
        try:
            viewer = self.nav.localizer.T_world_pelvis[:2, 3].copy()
        except Exception:
            pass
        choices = detect_balls(*data, fx=fx, viewer_xy=viewer) if fx else []
        if prior is not None and gate is not None:
            choices = [c for c in choices if math.hypot(c.x-prior[0], c.y-prior[1]) <= gate]
        result = max(choices, key=lambda c: c.score, default=None)
        self.last_ball_candidates = choices
        self._capture_ball_frame(data, result)
        return result

    def observe(self,kind,seconds=1.5,color=None):
        deadline,observations = time.monotonic()+seconds,[]
        previous = None
        while time.monotonic()<deadline:
            result = self.cone(color) if kind=='cone' else self.ball()
            if result is not None and result.valid() and self._stamp!=previous:
                observations.append(result)
                previous = self._stamp
            time.sleep(.08)
        if len(observations)<2:
            self.mission.log(f'[vision] {kind}: {len(observations)} stable frames; observed floor_z={self._floor_z:.2f}')
            return None
        xs,ys = [o.x for o in observations],[o.y for o in observations]
        if np.std(xs)>.35 or np.std(ys)>.35:
            self.mission.log(f'[vision] {kind}: unstable XY std=({np.std(xs):.2f},{np.std(ys):.2f})')
            return None
        result = observations[-1]
        result.x,result.y = float(np.median(xs)),float(np.median(ys))
        return result if result.valid() else None
