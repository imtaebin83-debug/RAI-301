"""Sensor-only localization regressions; these are not simulator success tests."""
import math
import unittest
from unittest import mock

import cv2
import numpy as np

from custom.mission_5 import perception


class PencilPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.camera = perception.CameraModel.panda_agentview_default()
        self.real_svd = np.linalg.svd

    def scene(self, legacy_count=None, *, distractor=False, eraser=False):
        hsv = np.zeros((256, 256, 3), dtype=np.uint8)
        hsv[:, :, 2] = 255
        hsv[100:108, 85:155] = [19, 220, 230]
        if legacy_count is None:
            hsv[100:108, 85:113, 0] = 20
        elif legacy_count:
            # Exactly 29/30 legacy pixels, along the pencil's long axis.
            hsv[100, 85:85 + legacy_count, 0] = 20
        if distractor:
            hsv[35:40, 45:85] = [25, 220, 230]
        if eraser:
            hsv[100:108, 158:162] = [165, 180, 220]
        hsv[145:185, 190:225] = [0, 0, 100]
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def pixels(self, image, lower_hue):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 0] >= lower_hue) & (hsv[:, :, 0] <= 42)
                & (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 155))
        # Restrict expected pixels to the synthetic pencil, excluding distractors.
        mask[:100] = False
        rows, cols = np.nonzero(mask)
        return self.camera.on_plane(np.c_[cols, rows], .0145 * 1.7)[:, :2]

    def axis(self, points):
        return self.real_svd(points - points.mean(0), full_matrices=False)[2][0]

    def force_new_axis(self, reference, sign):
        """Simulate either mathematically equivalent SVD sign across BLAS builds."""
        def signed_svd(values, *args, **kwargs):
            u, singular, vt = self.real_svd(values, *args, **kwargs)
            if len(values) == 70 * 8 and np.dot(vt[0], reference) * sign < 0:
                u[:, 0] *= -1
                vt[0] *= -1
            return u, singular, vt
        return mock.patch.object(perception.np.linalg, 'svd', side_effect=signed_svd)

    @staticmethod
    def measured_axis(measurement):
        return np.array([math.cos(measurement.pencil_yaw),
                         math.sin(measurement.pencil_yaw)])

    def test_h19_body_restores_full_paint_center_and_area(self):
        image = self.scene()
        full_points = self.pixels(image, 18)
        old_points = self.pixels(image, 20)
        measured = perception.measure_scene(image, self.camera)
        np.testing.assert_allclose(measured.pencil_xy, full_points.mean(0), atol=1e-12)
        self.assertGreater(np.linalg.norm(measured.pencil_xy - old_points.mean(0)), .05)
        self.assertEqual(measured.pencil_area, 70 * 8)
        np.testing.assert_allclose(measured.pencil_pixel, [119.5, 103.5])
        self.assertEqual(set(measured.as_dict()), {
            'pencil_xy', 'pencil_yaw', 'holder_xy', 'pencil_pixel',
            'holder_pixel', 'pencil_area', 'holder_area'})

    def test_new_pca_sign_keeps_legacy_heading(self):
        image = self.scene()
        legacy_axis = self.axis(self.pixels(image, 20))
        for sign in [-1, 1]:
            with self.subTest(new_pca_sign=sign), self.force_new_axis(legacy_axis, sign):
                measured = perception.measure_scene(image, self.camera)
            self.assertGreater(np.dot(self.measured_axis(measured), legacy_axis), .99)

    def test_legacy_reference_requires_at_least_30_pixels(self):
        for count in [29, 30]:
            image = self.scene(legacy_count=count)
            legacy_axis = self.axis(self.pixels(image, 20))
            with self.subTest(legacy_pixels=count), self.force_new_axis(legacy_axis, -1):
                measured = perception.measure_scene(image, self.camera)
            dot = np.dot(self.measured_axis(measured), legacy_axis)
            if count < 30:
                self.assertLess(dot, -.99)
            else:
                self.assertGreater(dot, .99)

    def test_separate_h20_distractor_does_not_choose_h19_pencil_branch(self):
        image = self.scene(legacy_count=0, distractor=True)
        raw_axis = self.axis(self.pixels(image, 18))
        for sign in [-1, 1]:
            with self.subTest(raw_pca_sign=sign), self.force_new_axis(raw_axis, sign):
                measured = perception.measure_scene(image, self.camera)
            self.assertGreater(np.dot(self.measured_axis(measured), sign * raw_axis), .99)
            self.assertEqual(measured.pencil_area, 70 * 8)

    def test_visible_eraser_still_defines_tip_direction(self):
        image = self.scene(eraser=True)
        legacy_axis = self.axis(self.pixels(image, 20))
        eraser_xy = self.camera.on_plane([[159.5, 103.5]], .0145 * 1.7)[0, :2]
        for sign in [-1, 1]:
            with self.subTest(new_pca_sign=sign), self.force_new_axis(legacy_axis, sign):
                measured = perception.measure_scene(image, self.camera)
            self.assertLess(np.dot(self.measured_axis(measured), eraser_xy - measured.pencil_xy), 0)


if __name__ == '__main__':
    unittest.main()
