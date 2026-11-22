"""CPU regressions for cone tracking and mission failure handling."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from custom.mission_2.mission_2 import Mission2
from custom.mission_2 import evaluate
from custom.mission_2.perception import ConeSensor, consensus
from custom.vision import Detection, RGBD


def detection(x=15.5, y=24.72):
    return Detection(x, y, 2., (12., 12.), 100.)


class ConsensusTests(unittest.TestCase):
    def test_projection_uses_paired_snapshot_despite_live_camera_updates(self):
        sequence = [0]

        def latest(topic, *args, **kwargs):
            if 'color' in topic:
                sequence[0] += 1
            return SimpleNamespace(encoding='32FC1' if 'depth' in topic else 'rgb8',
                header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=sequence[0]*1000)))

        mission = SimpleNamespace(latest=latest, stop=Mock(), log=Mock())
        sensor = ConeSensor(mission, Mock())

        def project(**kwargs):
            color = sensor._snapshot.latest('/camera/color/image_raw')
            depth = sensor._snapshot.latest('/camera/depth/image_raw')
            latest('/camera/color/image_raw')  # A live callback advances the cache.
            self.assertEqual(color.header.stamp.nanosec, depth.header.stamp.nanosec)
            self.assertIs(sensor._snapshot.latest('/camera/color/image_raw'), color)
            return ('fixed-data',)

        clock = iter(np.arange(0., 2., .01))
        with patch.object(sensor._projection, 'frame', side_effect=project), \
             patch.object(sensor, 'cone', return_value=detection()), \
             patch('custom.mission_2.perception.time.monotonic', side_effect=lambda: next(clock)), \
             patch('custom.mission_2.perception.time.sleep'):
            self.assertIsNotNone(sensor.observe_cone('blue', .3))

    def test_manifest_covers_three_colors_and_prior_mission_endpoints(self):
        cases = evaluate.cases()
        self.assertEqual(len({item['id'] for item in cases}), 12)
        for color in ('blue', 'green', 'orange'):
            self.assertTrue(any(item['m2_color'] == color and item['missions'] == [1, 2]
                                for item in cases))
            self.assertEqual(sum(item['m2_color'] == color and item['missions'] == [2]
                                 for item in cases), 3)

    def test_evaluator_refuses_active_training(self):
        with patch.object(evaluate.subprocess, 'run', return_value=SimpleNamespace(stdout='89688, python')):
            with self.assertRaises(RuntimeError):
                evaluate.check_gpu()

    def test_three_consistent_frames_required(self):
        self.assertIsNone(consensus([detection(), detection()]))
        self.assertIsNotNone(consensus([detection(), detection(), detection()]))

    def test_rejects_different_track_and_unstable_clusters(self):
        self.assertIsNone(consensus([detection(17.)]*3, previous=detection(14.)))
        self.assertIsNone(consensus([detection(14.), detection(15.), detection(17.)]))

    def test_one_outlier_does_not_move_robust_center(self):
        result = consensus([detection(15.49), detection(15.50), detection(15.51), detection(17.)])
        self.assertAlmostEqual(result.x, 15.5)

    def test_nonfinite_measurement_is_rejected(self):
        self.assertIsNone(consensus([detection(), detection(), detection(float('nan'))]))

    def test_every_color_uses_measured_world_points(self):
        for color, hue in [('blue', 115), ('green', 60), ('orange', 15)]:
            with self.subTest(color=color):
                hsv = np.zeros((30, 30, 3), dtype=np.uint8)
                hsv[4:26, 10:20] = [hue, 220, 210]
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                depth = np.full((30, 30), 2., dtype=np.float32)
                world = np.zeros((30, 30, 3))
                world[:] = [16.23, 24.61, .3]
                height = np.full((30, 30), .3)
                valid = np.ones((30, 30), dtype=bool)
                sensor = RGBD.__new__(RGBD)
                data = rgb, depth, world, height, valid
                result = sensor.cone(color, data=data)
                self.assertIsNotNone(result)
                np.testing.assert_allclose([result.x, result.y], [16.23, 24.61])
                self.assertIsNone(sensor.cone(color, data=(rgb, depth, world,
                                  np.zeros_like(height), valid)))

    def test_cached_pair_never_counts_as_new_observations(self):
        from sensor_msgs.msg import Image
        rgb, depth = Image(), Image()
        rgb.header.stamp.sec = depth.header.stamp.sec = 1
        rgb.encoding, depth.encoding = 'rgb8', '32FC1'
        mission = SimpleNamespace(stop=Mock(), log=Mock(), latest=lambda topic, *a, **k:
                                  depth if 'depth' in topic else rgb)
        sensor = ConeSensor(mission, Mock())
        with patch('custom.mission_2.perception.time.monotonic', side_effect=[0., 0., .1, 2.]), \
             patch('custom.mission_2.perception.time.sleep'), \
             patch.object(RGBD, 'frame') as frame:
            self.assertIsNone(sensor.observe_cone('blue', .5))
            frame.assert_not_called()


class MissionFailureTests(unittest.TestCase):
    def mission(self):
        return SimpleNamespace(begin_mission=Mock(return_value={}), params={'color': 'blue'},
            m2_color='green', log=Mock(), result=lambda **data: data, stop=Mock())

    def test_begin_failure_still_stops(self):
        mission = self.mission()
        mission.begin_mission.side_effect = TimeoutError('missing simulator')
        result = Mission2.run(mission)
        self.assertIn('missing simulator', result['error'])
        mission.stop.assert_called_once()

    def test_command_color_precedes_practice_color_and_stops_on_failed_route(self):
        mission = self.mission()
        nav = Mock()
        nav.route_to.return_value = False
        with patch('custom.mission_2.mission_2.Navigator', return_value=nav):
            result = Mission2.run(mission)
        self.assertIn('observation point', result['error'])
        mission.log.assert_called_with('[mission 2] target cone: blue')
        mission.stop.assert_called_once()

    def test_invalid_command_color_is_not_replaced_by_practice_color(self):
        mission = self.mission()
        mission.params['color'] = 'red'
        with patch('custom.mission_2.mission_2.Navigator') as nav:
            self.assertIn('valid cone color', Mission2.run(mission)['error'])
            nav.assert_not_called()

    def test_missing_observations_prevent_blind_approach(self):
        mission = self.mission()
        nav = Mock()
        with patch('custom.mission_2.mission_2.Navigator', return_value=nav), \
             patch('custom.mission_2.mission_2.ConeSensor') as sensor:
            sensor.return_value.observe_cone.return_value = None
            self.assertIn('No stable', Mission2.run(mission)['error'])
        nav.goto.assert_not_called()
        mission.stop.assert_called_once()

    def test_measured_approach_verifies_final_stop(self):
        mission = self.mission()
        position = [15.5, 22.3, 1.57]
        nav = Mock()
        nav.pose.side_effect = lambda: tuple(position)

        def goto(x, y, **kwargs):
            bound = 3.5
            self.assertLessEqual(np.hypot(x-position[0], y-position[1]), bound)
            self.assertLessEqual(kwargs['speed'], .18)
            position[:2] = [x, y]
            return True

        nav.goto.side_effect = goto
        with patch('custom.mission_2.mission_2.Navigator', return_value=nav), \
             patch('custom.mission_2.mission_2.ConeSensor') as sensor:
            sensor.return_value.observe_cone.side_effect = lambda *a, **k: detection()
            result = Mission2.run(mission)
            self.assertTrue(result['reached'])
            self.assertEqual(sensor.return_value.observe_cone.call_count, 1)
            self.assertLessEqual(result['final_leg_m'], 3.5)
            self.assertLess(result['final_pose_error_m'], .12)
        self.assertEqual(nav.goto.call_count, 1)
        mission.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
