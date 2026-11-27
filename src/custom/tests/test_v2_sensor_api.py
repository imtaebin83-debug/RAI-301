"""Public sensor-cache contract; no ROS context or simulator is started."""
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mission_api import Mission as OfficialMission
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty

from custom.sensor_mission import SensorMission, Mission, check_sensor_topic


class PublicSensorApiTests(unittest.TestCase):
    def make_node(self):
        subscriptions = []

        def initialize(node, node_name=None):
            node._api_group = object()
            node.test_node_name = node_name

        def subscribe(node, msg_type, topic, callback, qos, **kwargs):
            subscription = SimpleNamespace(msg_type=msg_type, topic=topic,
                                           callback=callback, qos=qos, **kwargs)
            subscriptions.append(subscription)
            return subscription

        # Keep the real inheritance/API surface, replacing only ROS allocation.
        with patch.object(OfficialMission, '__init__', initialize), \
             patch.object(SensorMission, 'create_subscription', subscribe):
            node = SensorMission('sensor_only_node')
        node.create_subscription = lambda *args, **kwargs: subscribe(node, *args, **kwargs)
        return node, subscriptions

    def test_official_inheritance_and_mission_zero_node(self):
        node, subscriptions = self.make_node()
        self.assertTrue(issubclass(SensorMission, OfficialMission))
        self.assertIs(Mission, SensorMission)
        self.assertEqual(node.MISSION, 0)
        self.assertEqual(node.test_node_name, 'sensor_only_node')
        self.assertEqual([s.topic for s in subscriptions], ['/g1/respawned'])
        self.assertFalse(hasattr(SensorMission, 'last_status'))

    def test_forbidden_topics_never_create_subscriptions(self):
        node, subscriptions = self.make_node()
        for topic in ['/g1/mission_status', '/g1/mission_end', '/g1/mission_begin',
                      '/g1/randomize', '/g1/respawn', '/odom', '/panda/task_status',
                      '/panda/retry', '/panda/base_pose', '/custom/',
                      '/custom/../g1/mission_status', 'imu', None]:
            with self.subTest(topic=topic), self.assertRaises(ValueError):
                node.latest(topic, Empty, timeout=0.)
        self.assertEqual(len(subscriptions), 1)

    def test_nonblocking_cache_is_shared_and_type_checked(self):
        node, subscriptions = self.make_node()
        self.assertIsNone(node.latest('/imu', Imu, timeout=0.))
        subscription = subscriptions[-1]
        message = Imu()
        subscription.callback(message)
        self.assertIs(node.latest('/imu', Imu, timeout=0.), message)
        self.assertIs(node.latest('/imu', Imu, timeout=.1), message)
        self.assertEqual(len(subscriptions), 2)
        with self.assertRaises(TypeError):
            node.latest('/imu', Empty, timeout=0.)

    def test_wait_wakes_on_sensor_arrival(self):
        node, subscriptions = self.make_node()
        node.latest('/imu', Imu, timeout=0.)
        message = Imu()
        results = []
        worker = threading.Thread(target=lambda: results.append(node.latest('/imu', Imu, timeout=.5)))
        worker.start()
        subscriptions[-1].callback(message)
        worker.join(timeout=1.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 1)
        self.assertIs(results[0], message)

    def test_missing_sensor_wait_is_bounded_and_timeout_is_validated(self):
        node, subscriptions = self.make_node()
        start = time.monotonic()
        self.assertIsNone(node.latest('/imu', Imu, timeout=.02))
        self.assertGreaterEqual(time.monotonic()-start, .015)
        self.assertLess(time.monotonic()-start, .5)
        for timeout in [-1., float('nan'), float('inf')]:
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                node.latest('/joint_states', Empty, timeout=timeout)
        self.assertEqual(len(subscriptions), 2)

    def test_respawns_use_only_public_empty_notifications(self):
        node, subscriptions = self.make_node()
        self.assertEqual(node.respawn_count, 0)
        first, second = Empty(), Empty()
        subscriptions[0].callback(first)
        subscriptions[0].callback(second)
        self.assertEqual(node.respawn_count, 2)
        self.assertEqual(node.gap_falls, 2)
        self.assertIs(node.latest('/g1/respawned', Empty, timeout=0.), second)
        self.assertEqual(len(subscriptions), 1)

    def test_team_derived_topics_and_latched_tf_are_supported(self):
        from rclpy.qos import DurabilityPolicy
        node, subscriptions = self.make_node()
        self.assertEqual(check_sensor_topic('/custom/estimated_pose'), '/custom/estimated_pose')
        node.latest('/custom/estimated_pose', Empty, timeout=0.)
        node.latest('/tf_static', Empty, timeout=0.)
        self.assertEqual(subscriptions[-1].qos.durability, DurabilityPolicy.TRANSIENT_LOCAL)


if __name__ == '__main__':
    unittest.main()
