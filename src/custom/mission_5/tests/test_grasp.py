"""Contact-constrained grasp regressions; these are not task success trials."""
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from custom.mission_5.keypose_policy import Policy, materialize, average_pencil_yaw


class ContactGraspTests(unittest.TestCase):
    def policy(self, at_target=True):
        checkpoint = Path(__file__).resolve().parents[1] / 'keyposes.json'
        p = Policy(checkpoint)
        p._scene = {'pencil_xy': [.4, -.1], 'holder_xy': [.45, .2], 'pencil_yaw': 0.}
        p._index = next(i for i, phase in enumerate(p.model['phases']) if phase['name'] == 'close')
        p.stage = 'close'
        p._start = p._stage_started = 0.
        index = p._index if at_target else p._index-1
        pose = materialize(p.model['phases'][index], p._scene)
        if not at_target:
            pose[2, 3] = max(pose[2, 3], .025)
        state = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec(), .04, -.04]
        return p, state

    def start_closing(self, p, state, start=0.):
        self.assertEqual(p.act({'time': start, 'state': state.copy()})['actions'][0, 6], -1.)
        self.assertEqual(p.act({'time': start+.1, 'state': state.copy()})['actions'][0, 6], 1.)
        self.assertIsNotNone(p._grasp_anchor)

    def test_open_hand_aligns_before_contact_blocks_arm_motion(self):
        p, state = self.policy(at_target=False)
        target = materialize(p.model['phases'][p._index], p._scene)
        initial_distance = np.linalg.norm(target[:3, 3]-state[:3])
        self.assertGreater(initial_distance, p.controller['position_tolerance_m'])
        opened_moves, closed_commands = 0, 0
        for i in range(70):
            action = p.act({'time': i*.1, 'state': state.copy()})['actions'][0]
            if action[6] < 0:
                # Toy plant: target reachable while open; closing creates a
                # table-supported contact that prevents further arm motion.
                opened_moves += int(np.linalg.norm(action[:3]) > 0)
                state[:3] += action[:3]*.05
                current = Rotation.from_rotvec(state[3:6])
                state[3:6] = (Rotation.from_rotvec(action[3:6]*.5)*current).as_rotvec()
            else:
                closed_commands += 1
                np.testing.assert_allclose(action[:6], 0., atol=1e-6)
                self.assertLess(np.linalg.norm(target[:3, 3]-state[:3]), .003)
                state[6:] = [.014, -.014]
            if p.stage != 'close':
                break
        self.assertGreater(opened_moves, 0)
        self.assertGreater(closed_commands, 1)
        self.assertEqual(p.stage, 'lift')
        self.assertIsNone(p.failed)

    def test_alignment_requires_distinct_timestamps(self):
        p, state = self.policy()
        for _ in range(4):
            self.assertEqual(p.act({'time': 0., 'state': state})['actions'][0, 6], -1.)
        self.assertIsNone(p._grasp_anchor)
        self.assertEqual(p._grasp_align_hits, 1)
        self.assertEqual(p.act({'time': .1, 'state': state})['actions'][0, 6], 1.)
        self.assertAlmostEqual(p._grasp_closed_at, .1)

    def test_unreachable_open_alignment_fails_before_closing(self):
        p, state = self.policy(at_target=False)
        for t in [0., 1., 2., 3., 4., 4.9]:
            self.assertEqual(p.act({'time': t, 'state': state})['actions'][0, 6], -1.)
            self.assertIsNone(p.failed)
        # Even arrival on the deadline cannot turn into an unbounded wait or
        # a last-moment close after the alignment budget has expired.
        _, reached = self.policy()
        action = p.act({'time': 5., 'state': reached})['actions'][0]
        self.assertIn('alignment timed out', p.failed)
        self.assertEqual(action[6], -1.)
        self.assertIsNone(p._grasp_anchor)

    def test_learned_dwell_begins_at_actual_closing(self):
        p, state = self.policy(at_target=False)
        p.act({'time': 0., 'state': state})
        _, state = self.policy()
        self.start_closing(p, state, start=4.3)
        state[6:] = [.014, -.014]
        for t in [4.5, 4.6, 4.7, 4.8, 5.0, 5.3]:
            p.act({'time': t, 'state': state})
        self.assertEqual(p.stage, 'close')
        self.assertLess(p.debug['close_elapsed'], p.model['phases'][p._index]['minimum_seconds'])
        p.act({'time': 5.5, 'state': state})
        self.assertEqual(p.stage, 'lift')

    def test_empty_hand_is_rejected(self):
        p, state = self.policy()
        self.start_closing(p, state)
        state[6:] = [0., 0.]
        p.act({'time': 2., 'state': state})
        self.assertIsNone(p.failed)
        for i in range(21, 25):
            p.act({'time': i*.1, 'state': state})
        self.assertIn('Empty grasp', p.failed)
        self.assertEqual(p.stage, 'close')

    def test_wide_hand_does_not_advance_and_times_out(self):
        p, state = self.policy()
        self.start_closing(p, state)
        state[6:] = [.04, -.04]
        for i in range(130):
            p.act({'time': i*.1, 'state': state})
        self.assertIn('timed out', p.failed)
        self.assertEqual(p.stage, 'close')

    def test_sliding_arm_does_not_count_as_stable_grasp(self):
        p, state = self.policy()
        self.start_closing(p, state)
        state[6:] = [.014, -.014]
        state[0] += .02
        for i in range(2, 130):
            p.act({'time': i*.1, 'state': state})
        self.assertIn('timed out', p.failed)

    def test_moving_fingers_need_settling_and_reset_clears_anchor(self):
        p, state = self.policy()
        self.start_closing(p, state)
        for i in range(2, 20):
            state[6:] = [.012 if i % 2 else .018, -.012 if i % 2 else -.018]
            p.act({'time': i*.1, 'state': state.copy()})
        self.assertEqual(p.stage, 'close')
        self.assertIsNone(p._grasp_stable_since)
        p.reset()
        self.assertIsNone(p._grasp_anchor)
        self.assertIsNone(p._grasp_closed_at)
        self.assertIsNone(p._last_align_time)
        self.assertEqual(p._grasp_align_hits, 0)
        self.assertIsNone(p._grasp_width)

    def test_alignment_does_not_extend_whole_stage_timeout(self):
        p, state = self.policy(at_target=False)
        p.act({'time': 0., 'state': state})
        _, reached = self.policy()
        self.start_closing(p, reached, start=4.3)
        p.act({'time': 12.1, 'state': reached})
        self.assertIn('stage timed out', p.failed)
        self.assertLess(12.1-p._grasp_closed_at, p.controller['stage_timeout_min'])

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
