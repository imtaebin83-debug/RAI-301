"""Sensor-decoding and invalid-depth regressions, not mission success tests."""
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
from sensor_msgs.msg import Image

from custom.vision import Detection, RGBD, _image


class ImageDecodingTests(unittest.TestCase):
    def test_bgr_rows_with_padding_are_rgb(self):
        message = Image()
        message.height, message.width, message.step = 2, 3, 12
        message.encoding = 'bgr8'
        rows = np.zeros((2,12),dtype=np.uint8)
        bgr = np.array([[[1,2,3],[4,5,6],[7,8,9]],
                        [[10,11,12],[13,14,15],[16,17,18]]],dtype=np.uint8)
        rows[:,:9] = bgr.reshape(2,9)
        rows[:,9:] = 233  # padding must not become a pixel
        message.data = rows.tobytes()
        np.testing.assert_array_equal(_image(message),bgr[:,:,::-1])

    def test_big_endian_float_depth_preserves_padding_and_nan(self):
        values = np.array([[1.,2.,3.],[4.,np.nan,6.]],dtype=np.float32)
        message = Image()
        message.height, message.width, message.step = 2, 3, 16
        message.encoding, message.is_bigendian = '32FC1', 1
        rows = np.zeros((2,16),dtype=np.uint8)
        for index in range(2):
            rows[index,:12] = np.frombuffer(values[index].astype('>f4').tobytes(),dtype=np.uint8)
        rows[:,12:] = 219
        message.data = rows.tobytes()
        np.testing.assert_allclose(_image(message,depth=True),values,equal_nan=True)

    def test_uint16_depth_is_converted_from_mm_to_m(self):
        message = Image()
        message.height, message.width, message.step = 1, 3, 6
        message.encoding, message.is_bigendian = '16UC1', 0
        message.data = np.array([[0,500,1250]],dtype='<u2').tobytes()
        np.testing.assert_allclose(_image(message,depth=True),[[0.,.5,1.25]])


class InvalidDepthRegressionTests(unittest.TestCase):
    def test_ball_silhouette_cannot_admit_nan_depth_neighbours(self):
        # A rendered ball silhouette borders invalid depth. RGBD.ball() must
        # return a finite sphere-centre estimate from measured pixels only.
        from custom.tests.test_ball import render, FX
        rgb, depth, world, height, valid = render(
            (12.0, 9.1, 1.2), math.atan2(8.15-9.1, 9.25-12.0), spheres=[((9.25, 8.15, .11), .11)])
        ball_px = (height > .02) & valid
        border = ball_px & ~np.roll(ball_px, 1, axis=1)
        valid = valid & ~border
        depth[border] = np.nan
        world[border] = np.nan
        height[border] = np.nan
        detector = RGBD.__new__(RGBD)
        detector._stamp = (1, 0)
        detector.frame = lambda: (rgb, depth, world, height, valid)
        info = SimpleNamespace(k=[FX, 0., 160., 0., FX, 120., 0., 0., 1.])
        detector.mission = SimpleNamespace(latest=lambda *a, **k: info, log=lambda text: None)
        detector.nav = SimpleNamespace(localizer=SimpleNamespace(T_world_pelvis=np.eye(4)))
        result = detector.ball()
        self.assertIsNotNone(result)
        self.assertTrue(result.valid())
        self.assertTrue(np.isfinite([result.x,result.y,result.distance]).all())
        np.testing.assert_allclose([result.x,result.y],[9.25,8.15],atol=.03)

    def test_invalid_candidate_is_not_an_observation(self):
        detector = RGBD.__new__(RGBD)
        detector._stamp = (1,0)
        detector._floor_z = 0.
        detector.mission = SimpleNamespace(log=lambda text: None)
        detector.ball = lambda: Detection(math.nan,8.,2.,(10.,10.),40.)
        with patch('custom.vision.time.monotonic',side_effect=[0.,0.,2.]), \
             patch('custom.vision.time.sleep'):
            self.assertIsNone(detector.observe('ball',seconds=1.))

    def test_detection_rejects_nonfinite_fields_and_nonpositive_range(self):
        self.assertTrue(Detection(1.,2.,3.,(4.,5.),6.).valid())
        for values in (
            (math.nan,2.,3.,(4.,5.),6.),
            (1.,math.inf,3.,(4.,5.),6.),
            (1.,2.,0.,(4.,5.),6.),
            (1.,2.,3.,(math.nan,5.),6.),
            (1.,2.,3.,(4.,5.),math.inf),
        ):
            self.assertFalse(Detection(*values).valid())


if __name__=='__main__':
    unittest.main()
