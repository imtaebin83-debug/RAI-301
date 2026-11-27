"""Numerical scan/IMU and map-commit regressions; CPU only."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from custom.inertial import GyroBuffer, ImuPending
from custom.icp_registration import GuardedKissICP, RegistrationRejected
from custom.localization import Localizer
from custom.navigation import ContinuousPose, Navigator, translation_command


class GyroTests(unittest.TestCase):
    def test_irregular_scan_endpoints_and_axis_transform(self):
        gyro = GyroBuffer()
        for t in np.arange(0., .51, .01):
            gyro.add(t, [1., 0., 0.], 'imu')
        extrinsic = Rotation.from_euler('y', -np.pi / 2).as_matrix()
        actual = Rotation.from_matrix(gyro.rotation(.035, .407, extrinsic)).as_rotvec()
        np.testing.assert_allclose(actual, [0., 0., .372], atol=1e-10)

    def test_missing_endpoint_never_extrapolates(self):
        gyro = GyroBuffer()
        gyro.add(1., [0., 0., 1.], 'imu')
        gyro.add(1.01, [0., 0., 1.], 'imu')
        self.assertFalse(gyro.covers(.99))
        self.assertTrue(gyro.covers(1.))
        self.assertFalse(gyro.covers(1.02))
        for interval in ((.99, 1.), (1., 1.02), (1.01, 1.)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                gyro.rotation(*interval, np.eye(3))

    def test_sample_gap_is_not_integrated(self):
        gyro = GyroBuffer()
        gyro.add(0., [0., 0., 1.], 'imu')
        gyro.add(.2, [0., 0., 1.], 'imu')
        with self.assertRaisesRegex(ValueError, 'gap'):
            gyro.rotation(0., .2, np.eye(3))

    def test_delayed_endpoint_can_retry_same_interval(self):
        gyro = GyroBuffer()
        gyro.add(1., [0., 0., 1.], 'imu')
        gyro.add(1.01, [0., 0., 1.], 'imu')
        with self.assertRaises(ImuPending):
            gyro.rotation(1., 1.02, np.eye(3))
        gyro.add(1.02, [0., 0., 1.], 'imu')
        np.testing.assert_allclose(Rotation.from_matrix(
            gyro.rotation(1., 1.02, np.eye(3))).as_rotvec(), [0., 0., .02], atol=1e-12)

    def test_malformed_and_reordered_imu_not_accepted(self):
        gyro = GyroBuffer()
        self.assertTrue(gyro.add(1., [0., 0., 1.], 'imu'))
        for stamp, omega, frame in ((1., [0, 0, 1], 'imu'), (.9, [0, 0, 1], 'imu'),
                                     (2., [np.nan, 0, 0], 'imu'),
                                     (2., [0, 0, 1], 'another_imu')):
            self.assertFalse(gyro.add(stamp, omega, frame))


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        from kiss_icp.config import KISSConfig
        config = KISSConfig()
        config.mapping.voxel_size = .15
        config.registration.max_num_threads = 1
        config.registration.max_num_iterations = 40
        self.odom = GuardedKissICP(config)
        rng = np.random.default_rng(701)
        self.points = rng.uniform(-4., 4., (6000, 3))
        self.odom.register_frame(self.points, [])

    def _map(self):
        points = self.odom.icp.local_map.point_cloud()
        return points[np.lexsort(points.T)]

    def test_real_icp_with_irregular_scan_period(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('z', .04).as_matrix()
        pose[:3, 3] = [.12, -.02, .01]
        current = (self.points - pose[:3, 3]) @ pose[:3, :3]
        self.odom.register_frame(current, [], dt=.3, rotation=pose[:3, :3])
        np.testing.assert_allclose(self.odom.last_pose, pose, atol=.007)
        self.assertGreater(self.odom.quality, .95)
        json.dumps(self.odom.metrics, allow_nan=False)

    def test_rejected_jump_does_not_modify_trusted_map(self):
        before = self._map()
        bad = np.eye(4)
        bad[0, 3] = 2.
        with patch.object(self.odom.icp.registration, 'align_points_to_map', return_value=bad):
            with self.assertRaisesRegex(RegistrationRejected, 'jump'):
                self.odom.register_frame(self.points, [], dt=.1)
        np.testing.assert_array_equal(before, self._map())
        np.testing.assert_array_equal(self.odom.last_pose, np.eye(4))
        self.assertEqual(self.odom.metrics['translation'], 2.)
        json.dumps(self.odom.metrics, allow_nan=False)

    def test_pelvis_validation_precedes_map_commit(self):
        before = self._map()
        def reject(candidate):
            raise RegistrationRejected('pelvis transform invalid')
        with self.assertRaisesRegex(RegistrationRejected, 'pelvis'):
            self.odom.register_frame(self.points, [], validator=reject)
        np.testing.assert_array_equal(before, self._map())

    def test_bad_scan_does_not_destroy_following_good_scan(self):
        before = self._map()
        with patch.object(self.odom.icp.registration, 'align_points_to_map', return_value=np.eye(4)):
            with self.assertRaises(RegistrationRejected):
                self.odom.register_frame(self.points + 20., [], dt=.1)
        np.testing.assert_array_equal(before, self._map())
        self.odom.register_frame(self.points, [], dt=.2, rotation=np.eye(3))
        self.assertGreater(self.odom.quality, .95)
        np.testing.assert_allclose(self.odom.last_pose, np.eye(4), atol=.007)

    def test_gyro_disagreement_cannot_pollute_map(self):
        before = self._map()
        bad = np.eye(4)
        bad[:3, :3] = Rotation.from_euler('z', .2).as_matrix()
        with patch.object(self.odom.icp.registration, 'align_points_to_map', return_value=bad):
            with self.assertRaisesRegex(RegistrationRejected, 'IMU'):
                self.odom.register_frame(self.points, [], rotation=np.eye(3))
        np.testing.assert_array_equal(before, self._map())

    def test_large_sensor_gap_cannot_be_hidden_by_prediction(self):
        before = self._map()
        with self.assertRaisesRegex(RegistrationRejected, 'interval'):
            self.odom.register_frame(self.points, [], dt=1.1)
        np.testing.assert_array_equal(before, self._map())


class FakeMission:
    gap_falls = 0

    def __init__(self):
        self.messages = {}
        self.commands = []
        self.logs = []

    def latest(self, topic, *args, **kwargs):
        return self.messages.get(topic)

    def stop(self):
        self.commands.append((0., 0., 0.))

    def cmd_vel(self, *command):
        self.commands.append(command)

    def log(self, text):
        self.logs.append(text)


def pose_message(stamp=10, x=16., frame='map'):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.header.stamp.sec = stamp
    msg.pose.position.x = x
    msg.pose.position.y = 11.
    msg.pose.orientation.w = 1.
    return msg


class PoseHealthTests(unittest.TestCase):
    def setUp(self):
        self.mission = FakeMission()
        self.pose = ContinuousPose(self.mission)
        self.mission.messages['/custom/estimated_pose'] = pose_message()

    def test_duplicate_measurement_does_not_renew_freshness(self):
        with patch('custom.navigation.time.monotonic', return_value=100.):
            self.pose.update()
            self.assertFalse(self.pose.stale)
        with patch('custom.navigation.time.monotonic', return_value=101.3):
            self.pose.update()
            self.assertTrue(self.pose.stale)

    def test_invalid_pose_does_not_overwrite_trusted_transform(self):
        self.pose.update()
        trusted = self.pose.T_world_pelvis.copy()
        for invalid in (pose_message(11, np.nan), pose_message(11, frame='odom')):
            self.mission.messages['/custom/estimated_pose'] = invalid
            self.pose.update()
            self.assertTrue(self.pose.stale)
            np.testing.assert_array_equal(self.pose.T_world_pelvis, trusted)

    def test_localizer_failure_propagates_even_with_recent_pose(self):
        self.pose.update()
        self.mission.messages['/custom/localization_status'] = String(data=json.dumps(
            {'version': 1, 'valid': False, 'terminal': True, 'reason': 'registration lost'}))
        self.pose.update()
        self.assertTrue(self.pose.stale)
        self.assertTrue(self.pose.terminal)
        self.assertEqual(self.pose.last_error, 'registration lost')

    def test_sensor_clock_detects_transport_lag(self):
        self.mission.messages['/custom/localization_status'] = String(data=json.dumps(
            {'version': 1, 'valid': True}))
        clock = Clock()
        clock.clock.sec = 12
        self.mission.messages['/clock'] = clock
        self.pose.update()
        self.assertTrue(self.pose.stale)

    def test_sensor_clock_rejects_future_estimate(self):
        self.mission.messages['/custom/localization_status'] = String(data=json.dumps(
            {'version': 1, 'valid': True}))
        clock = Clock()
        clock.clock.sec = 9
        self.mission.messages['/clock'] = clock
        self.pose.update()
        self.assertTrue(self.pose.stale)

    def test_old_publisher_without_health_topic_remains_compatible(self):
        self.pose.update()
        self.assertFalse(self.pose.stale)

    def test_stalled_robot_gets_bounded_reobservations_then_stops(self):
        clock = [100.]
        nav = Navigator.__new__(Navigator)
        nav.mission = self.mission
        nav.localizer = SimpleNamespace(_stamp=0)
        nav._last_log = 0.
        def sample():
            nav.localizer._stamp += 1
            return 16., 11., 0.
        nav._observed_pose = lambda deadline: sample()
        with patch('custom.navigation.time.monotonic', side_effect=lambda: clock[0]), \
                patch('custom.navigation.time.sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0] + dt)):
            self.assertFalse(nav.goto(17., 11., timeout=100.))
        self.assertEqual(nav.last_failure, 'stalled')
        self.assertLess(clock[0] - 100., 26.)
        self.assertEqual(self.mission.commands[-1], (0., 0., 0.))


class PendingScanTests(unittest.TestCase):
    def localizer(self):
        loc = object.__new__(Localizer)
        cloud = SimpleNamespace(header=SimpleNamespace(frame_id='lidar_link',
            stamp=SimpleNamespace(sec=10, nanosec=100000000)))
        loc.mission = SimpleNamespace(last_status=lambda: {'seq': -1}, latest=lambda *a, **k: cloud)
        loc._last_status_seq, loc._recovery_blocked = -1, False
        loc._stamp = loc._attempt_stamp = (10, 0)
        loc._last_received, loc.quality = 100., .99
        loc._pose = (16., 11., 0.)
        world = np.eye(4)
        world[:2, 3] = [16., 11.]
        loc._last_transform = loc._world_lidar0 = world
        loc._map_offset, loc.map_matcher = np.eye(4), None
        loc._odom = Mock(spec=GuardedKissICP)
        loc._odom.last_pose, loc._odom.quality, loc._odom.metrics = np.eye(4), .98, {}
        loc.metrics, loc.rejected_scans = {}, 0
        loc._pelvis_lidar = lambda stamp: np.eye(4)
        return loc

    def test_pending_scan_preserves_age_and_retries_after_endpoint_arrives(self):
        loc = self.localizer()
        loc._scan_rotation = Mock(side_effect=[ImuPending('waiting endpoint'), np.eye(3)])
        with patch('custom.localization.cloud_xyz', return_value=np.ones((120, 3))), \
                patch('custom.localization.time.monotonic', return_value=100.2):
            loc.update()
            self.assertFalse(loc.stale)
            self.assertEqual(loc.measurement_stamp, (10, 0))
            self.assertEqual(loc._last_received, 100.)
            loc._odom.register_frame.assert_not_called()
            loc.update()
        self.assertEqual(loc.measurement_stamp, (10, 100000000))
        self.assertEqual(loc.rejected_scans, 0)
        loc._odom.register_frame.assert_called_once()

    def test_permanent_pending_never_refreshes_old_measurement(self):
        loc = self.localizer()
        loc._scan_rotation = Mock(side_effect=ImuPending('waiting endpoint'))
        with patch('custom.localization.cloud_xyz', return_value=np.ones((120, 3))), \
                patch('custom.localization.time.monotonic', return_value=101.1):
            loc.update()
            self.assertTrue(loc.stale)
        self.assertEqual(loc.measurement_stamp, (10, 0))
        self.assertEqual(loc._last_received, 100.)
        loc._odom.register_frame.assert_not_called()


class TranslationTests(unittest.TestCase):
    def test_long_goal_preserves_world_bearing_during_small_heading_error(self):
        for yaw, delta in ((1.7, [0., 7.]), (-3.07, [-6., 0.]), (.15, [4., 0.])):
            with self.subTest(yaw=yaw):
                command = translation_command(*delta, yaw, .28)
                c, s = np.cos(yaw), np.sin(yaw)
                world = np.array([[c, -s], [s, c]]) @ command
                expected = .28 * np.asarray(delta) / np.linalg.norm(delta)
                np.testing.assert_allclose(world, expected, atol=1e-12)
                self.assertLessEqual(np.linalg.norm(command), .28 + 1e-12)

    def test_final_correction_gain_and_body_limits_are_preserved(self):
        np.testing.assert_allclose(translation_command(.1, -.1, 0., .38), [.09, -.09])
        np.testing.assert_allclose(translation_command(-10., 0., 0., .38), [-.16, 0.])
        np.testing.assert_allclose(translation_command(0., 10., 0., .38), [0., .20])


class RouteTests(unittest.TestCase):
    def test_room_departure_uses_door_before_corridor_target(self):
        self.assertEqual(Navigator._route_points(20., 10.5, 15., 5.),
                         [(19., 10.5), (16., 10.5), (15., 5.)])

    def test_same_room_does_not_force_unnecessary_exit(self):
        self.assertEqual(Navigator._route_points(20., 17.5, 19., 17.), [(19., 17.)])

    def test_two_different_rooms_use_both_doors(self):
        self.assertEqual(Navigator._route_points(20., 17.5, 20., 3.59),
                         [(19., 17.5), (16., 17.5), (16., 3.59), (19., 3.59), (20., 3.59)])

    def test_cone_room_departure_is_measured_not_teleported(self):
        self.assertEqual(Navigator._route_points(15.5, 24., 16., 11.),
                         [(15.5, 22.), (15.5, 20.3), (16., 11.)])

    def test_replan_occurs_only_once_and_uses_updated_measurement(self):
        nav = Navigator.__new__(Navigator)
        nav.mission = FakeMission()
        poses = iter([(20., 10.5, 0.), (16., 10.5, 0.)])
        nav._observed_pose = lambda deadline: next(poses)
        calls = []
        def attempt(x, y, **kwargs):
            calls.append((x, y))
            nav.last_failure = 'stalled'
            return len(calls) > 1
        nav.goto = attempt
        self.assertTrue(nav.route_to(15., 5.))
        self.assertEqual(calls, [(19., 10.5), (15., 5.)])


class Mission1InputTests(unittest.TestCase):
    def mission(self, goal):
        from custom.mission_1.mission_1 import Mission1
        mission = object.__new__(Mission1)
        mission.params = {'goal': goal}
        mission.begin_mission = lambda number: {'pos': [99., 99.], 'yaw_deg': 90.}
        mission.stop = lambda: setattr(mission, 'stopped', True)
        mission.result = lambda **data: data
        return mission

    def test_invalid_direct_params_do_not_fall_back_to_practice_goal(self):
        for goal in ([], [True, 4.], [float('nan'), 5.], [18., 10.], [16., 21.], '16,11'):
            mission = self.mission(goal)
            with self.subTest(goal=goal), patch('custom.mission_1.mission_1.Navigator') as nav:
                result = mission.run()
                self.assertIn('error', result)
                self.assertTrue(mission.stopped)
                nav.assert_not_called()

    def test_public_target_limits_and_api_are_preserved(self):
        for goal in ([14., 1.], [17., 20.]):
            mission = self.mission(goal)
            with patch('custom.mission_1.mission_1.Navigator') as nav:
                nav.return_value.route_to.return_value = True
                result = mission.run()
                nav.return_value.route_to.assert_called_once_with(*goal, tolerance=.18, timeout=180)
                self.assertTrue(result['controller_reached'])
                self.assertTrue(mission.stopped)

    def test_begin_mission_exception_also_stops(self):
        mission = self.mission([16., 11.])
        mission.begin_mission = lambda number: (_ for _ in ()).throw(TimeoutError('no ack'))
        self.assertEqual(mission.run()['error'], 'no ack')
        self.assertTrue(mission.stopped)


if __name__ == '__main__':
    unittest.main()
