"""Soccer-ball detector checks on synthetic RGB-D scenes (no ROS, no simulator).

These verify geometry/colour rejection rules of ``detect_balls``; they are not
a substitute for simulator runs with the official referee.
"""
import math
import unittest

import numpy as np

from custom.vision import BALL_RADIUS, detect_balls

FX = 617.2
H, W = 240, 320
CX, CY = W/2, H/2
GRASS = (60, 110, 55)


def render(camera, yaw, pitch_deg=15., spheres=(), posts=(), rings=(), wall_y=None):
    """Ray-cast a pinhole camera over a flat floor (z=0).

    spheres: (centre_xyz, radius) soccer balls with black/white panels
    posts: (x, y, radius, height, rgb) vertical cylinders
    rings: (x, y, r_in, r_out) flat white rings 2 mm high
    Returns rgb, depth, world, height, valid like ``RGBD.frame``.
    """
    pitch = math.radians(pitch_deg)
    cam = np.asarray(camera, dtype=float)
    forward = np.array([math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.cos(pitch), -math.sin(pitch)])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.])
    up = np.cross(forward, left)
    v, u = np.indices((H, W), dtype=float)
    rays = forward[None, None]+((CX-u)/FX)[..., None]*left+((CY-v)/FX)[..., None]*up
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    t_best = np.full((H, W), np.inf)
    colour = np.zeros((H, W, 3), dtype=np.uint8)
    # floor
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -cam[2]/rays[..., 2]
    floor = (t > 0) & np.isfinite(t)
    t_best[floor] = t[floor]
    colour[floor] = GRASS
    if wall_y is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            tw = (wall_y-cam[1])/rays[..., 1]
        hit = (tw > 0) & (tw < t_best)
        colour[hit] = (95, 140, 135)
        t_best[hit] = tw[hit]
    for x, y, r_in, r_out in rings:
        with np.errstate(divide='ignore', invalid='ignore'):
            tr = (.002-cam[2])/rays[..., 2]
        p = cam+tr[..., None]*rays
        d = np.hypot(p[..., 0]-x, p[..., 1]-y)
        hit = (tr > 0) & (d > r_in) & (d < r_out) & (tr < t_best)
        colour[hit] = (230, 230, 230)
        t_best[hit] = tr[hit]
    for x, y, radius, height, rgb in posts:
        o = cam[:2]-np.array([x, y])
        a = (rays[..., :2]**2).sum(-1)
        b = 2*(rays[..., :2]*o).sum(-1)
        c = (o**2).sum()-radius**2
        disc = b*b-4*a*c
        with np.errstate(invalid='ignore'):
            tc = (-b-np.sqrt(disc))/(2*a)
        z = cam[2]+tc*rays[..., 2]
        hit = (disc > 0) & (tc > 0) & (z >= 0) & (z <= height) & (tc < t_best)
        colour[hit] = rgb
        t_best[hit] = tc[hit]
    for centre, radius in spheres:
        centre = np.asarray(centre, dtype=float)
        o = cam-centre
        b = 2*(rays*o).sum(-1)
        c = (o**2).sum()-radius**2
        disc = b*b-4*c
        with np.errstate(invalid='ignore'):
            ts = (-b-np.sqrt(disc))/2
        hit = (disc > 0) & (ts > 0) & (ts < t_best)
        p = cam+ts[..., None]*rays-centre
        az = np.arctan2(p[..., 1], p[..., 0])
        el = np.arcsin(np.clip(p[..., 2]/radius, -1, 1))
        black = (np.floor(az/(math.pi/5))+np.floor(el/(math.pi/5))) % 3 == 0
        colour[hit & black] = (15, 15, 15)
        colour[hit & ~black] = (240, 240, 240)
        t_best[hit] = ts[hit]
    valid = np.isfinite(t_best)
    world = cam+np.where(valid, t_best, np.nan)[..., None]*rays
    depth = np.where(valid, t_best*(rays@forward), np.nan).astype(np.float32)
    return colour, depth, world, world[..., 2].copy(), valid


class BallDetectionTests(unittest.TestCase):
    ball = (9.25, 8.15, BALL_RADIUS)

    def view(self, **extra):
        return render((12.0, 9.1, 1.2), math.atan2(8.15-9.1, 9.25-12.0),
                      spheres=[(self.ball, BALL_RADIUS)], rings=[(9.25, 8.15, .2, .26)], **extra)

    def test_sphere_centre_from_front_surface(self):
        found = detect_balls(*self.view(), fx=FX, viewer_xy=(12.0, 9.1))
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].x, 9.25, delta=.03)
        self.assertAlmostEqual(found[0].y, 8.15, delta=.03)
        self.assertLess(found[0].rms, .01)

    def test_flat_ring_and_markings_alone_are_rejected(self):
        frame = render((12.0, 9.1, 1.2), math.atan2(8.15-9.1, 9.25-12.0), rings=[(9.25, 8.15, .2, .26)])
        self.assertEqual(detect_balls(*frame, fx=FX), [])

    def test_white_post_base_is_rejected(self):
        frame = render((9.25, 6.0, 1.2), -math.pi/2,
                       posts=[(9.0, 3.5, .06, 2.0, (240, 240, 240))])
        self.assertEqual(detect_balls(*frame, fx=FX), [])

    def test_uniform_white_dome_without_panels_is_rejected(self):
        rgb, depth, world, height, valid = self.view()
        ball_px = (height > .02) & valid
        rgb[ball_px] = (240, 240, 240)
        self.assertEqual(detect_balls(rgb, depth, world, height, valid, fx=FX), [])

    def test_colour_shift_relative_to_depth_is_tolerated(self):
        rgb, depth, world, height, valid = self.view()
        shifted = np.roll(rgb, 8, axis=0)      # colour captured at another head pose
        found = detect_balls(shifted, depth, world, height, valid, fx=FX)
        self.assertEqual(len(found), 1)

    def test_invalid_depth_border_keeps_estimate_finite(self):
        rgb, depth, world, height, valid = self.view()
        ball_px = (height > .02) & valid
        border = ball_px & ~np.roll(ball_px, 1, axis=1)
        valid = valid & ~border
        depth[border] = np.nan
        world[border] = np.nan
        height[border] = np.nan
        found = detect_balls(rgb, depth, world, height, valid, fx=FX)
        self.assertEqual(len(found), 1)
        self.assertTrue(np.isfinite([found[0].x, found[0].y, found[0].distance]).all())

    def test_ball_against_corner_is_rejected(self):
        frame = render((11.0, 8.5, 1.2), math.atan2(10.35-8.5, 12.85-11.0),
                       spheres=[((12.85, 10.35, BALL_RADIUS), BALL_RADIUS)])
        self.assertEqual(detect_balls(*frame, fx=FX), [])

    def test_reasons_are_reported(self):
        reasons = []
        frame = render((9.25, 6.0, 1.2), -math.pi/2, posts=[(9.0, 3.5, .06, 2.0, (240, 240, 240))])
        self.assertEqual(detect_balls(*frame, fx=FX, reasons=reasons), [])
        self.assertTrue(reasons)
        self.assertTrue(all('why' in r for r in reasons))


if __name__ == '__main__':
    unittest.main()
