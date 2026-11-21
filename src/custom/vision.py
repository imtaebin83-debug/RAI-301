"""RGB-D object observations; no scene files or simulator truth are read."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

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

    def ball(self):
        data = self.frame()
        if data is None:
            return None
        rgb, depth, world, height, valid = data
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = (valid & (height>.045) & (height<.27) & (hsv[:,:,1]<85) &
                (hsv[:,:,2]>65) & (world[:,:,0]>6.) & (world[:,:,0]<12.6) &
                (world[:,:,1]>1.5) & (world[:,:,1]<10.))
        original_mask = mask
        # An even-sized morphology kernel shifts the mask by a pixel; that can
        # select invalid depth immediately beside a ball silhouette. Preserve
        # the original measured-pixel support after a symmetric opening.
        mask = cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask = mask.astype(bool) & original_mask & valid & np.isfinite(world).all(-1)
        count,labels,stats,centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        choices = []
        for label in range(1,count):
            _,_,width,tall,area = stats[label]
            if area<8 or min(width,tall)<2 or max(width,tall)>min(width,tall)*4:
                continue
            selected = (labels==label) & valid & np.isfinite(world).all(-1) & np.isfinite(depth)
            if selected.sum()<8:
                continue
            points = world[selected,:2]
            if np.max(np.ptp(points,axis=0))>.6:
                continue
            xy = np.median(points,axis=0)
            result = Detection(float(xy[0]),float(xy[1]),float(np.median(depth[selected])),
                               tuple(centroids[label]),float(selected.sum()))
            if result.valid():
                choices.append(result)
        return max(choices,key=lambda d:d.area,default=None)

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
