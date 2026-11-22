"""Contact-constrained grasp regressions; these are not task success trials."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from custom.mission_5.keypose_policy import Policy, materialize, average_pencil_yaw


class ContactGraspTests(unittest.TestCase):
    def policy(self):
        checkpoint = Path(__file__).resolve().parents[1] / 'keyposes.json'
        p = Policy(checkpoint)
        p._scene = {'pencil_xy': [.4, -.1], 'holder_xy': [.45, .2], 'pencil_yaw': 0.}
        p._index = next(i for i, phase in enumerate(p.model['phases']) if phase['name'] == 'close')
        p.stage = 'close'
        p._start = p._stage_started = 0.
        pose = materialize(p.model['phases'][p._index-1], p._scene)
        # A table-supported pencil prevents descending to the deflected demo
        # endpoint. The old close target remains >6mm below this contact pose.
        pose[2, 3] = max(pose[2, 3], .025)
        state = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec(), .014, -.014]
        return p, state

    def test_contact_blocks_old_endpoint_but_stable_grasp_can_lift(self):
        p, state = self.policy()
        old = materialize(p.model['phases'][p._index], p._scene)
        self.assertGreater(np.linalg.norm(old[:3, 3]-state[:3]), p.controller['position_tolerance_m'])
        for i in range(15):
            action = p.act({'time': i*.1, 'state': state.copy()})['actions'][0]
            if p.stage != 'close':
                break
            np.testing.assert_allclose(action[:6], 0., atol=1e-6)
            self.assertEqual(action[6], 1.)
        self.assertEqual(p.stage, 'lift')
        self.assertIsNone(p.failed)

    def test_empty_hand_is_rejected(self):
        p, state = self.policy()
        state[6:] = [0., 0.]
        for i in range(25):
            p.act({'time': i*.1, 'state': state})
        self.assertIn('Empty grasp', p.failed)
        self.assertEqual(p.stage, 'close')

    def test_wide_hand_does_not_advance_and_times_out(self):
        p, state = self.policy()
        state[6:] = [.04, -.04]
        for i in range(130):
            p.act({'time': i*.1, 'state': state})
        self.assertIn('timed out', p.failed)
        self.assertEqual(p.stage, 'close')

    def test_sliding_arm_does_not_count_as_stable_grasp(self):
        p, state = self.policy()
        p.act({'time': 0., 'state': state.copy()})
        state[0] += .02
        for i in range(1, 130):
            p.act({'time': i*.1, 'state': state})
        self.assertIn('timed out', p.failed)

    def test_moving_fingers_need_settling_and_reset_clears_anchor(self):
        p, state = self.policy()
        for i in range(20):
            state[6:] = [.012 if i % 2 else .018, -.012 if i % 2 else -.018]
            p.act({'time': i*.1, 'state': state.copy()})
        self.assertEqual(p.stage, 'close')
        self.assertIsNone(p._grasp_stable_since)
        p.reset()
        self.assertIsNone(p._grasp_anchor)
        self.assertIsNone(p._grasp_width)

    def test_opposite_pca_signs_keep_first_grasp_branch(self):
        yaw = average_pencil_yaw([-.9, -.9+np.pi, -.91, -.91+np.pi])
        self.assertAlmostEqual(yaw, -.905, places=5)
        other = average_pencil_yaw([-.9+np.pi, -.9, -.91+np.pi, -.91])
        self.assertAlmostEqual(abs(other-yaw), np.pi, places=5)

    def test_inconsistent_line_axes_are_not_accepted(self):
        with self.assertRaises(ValueError):
            average_pencil_yaw([0., np.pi/2, 0.])


if __name__ == '__main__':
    unittest.main()
