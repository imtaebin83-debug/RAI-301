"""CPU fault injection for v2; no ROS executor or simulator is started."""
import threading
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

from custom.sensor_mission import Mission as SensorMission
from custom.mission_2.mission_2 import Mission2
from custom.mission_2.perception import ConeSensor
from custom.mission_2.test_mission_2 import mission_stub, detection


class V2ConeTests(unittest.TestCase):
    def test_public_sensor_base_and_none_begin_contract(self):
        self.assertTrue(issubclass(Mission2, SensorMission))
        mission = mission_stub(begin_mission=Mock(return_value=None), params={'color':'blue'},
            m2_color='green', log=Mock(), stop=Mock(), result=lambda **data: data)
        nav = Mock()
        nav.pose.side_effect = [(15.5,22.3,1.57), (15.5,24.05,1.57)]
        with patch('custom.mission_2.mission_2.Navigator', return_value=nav) as factory, \
             patch('custom.mission_2.mission_2.ConeSensor') as sensor, \
             patch('custom.mission_2.mission_2.time.sleep'):
            sensor.return_value.observe_cone.return_value = detection()
            result = Mission2.run(mission)
        factory.assert_called_once_with(mission, None)
        self.assertTrue(result['reached'])

    def test_respawn_at_each_phase_never_claims_success_or_reuses_target(self):
        for phase in ('begin', 'route', 'survey', 'approach', 'settle', 'final_pose'):
            with self.subTest(phase=phase):
                mission = mission_stub(begin_mission=Mock(return_value=None),
                    params={'color':'blue'}, m2_color='green',
                    log=Mock(), stop=Mock(), result=lambda **data: data)
                nav, sensor = Mock(), Mock()
                nav.pose.side_effect = [(15.5,22.3,1.57), (15.5,24.05,1.57)]
                sensor.observe_cone.return_value = detection()
                def respawn(value):
                    mission.respawn_count += 1
                    return value
                if phase=='begin':
                    mission.begin_mission.side_effect=lambda *a,**k: respawn(None)
                if phase=='route':
                    nav.route_to.side_effect=lambda *a,**k: respawn(False)
                if phase=='survey':
                    sensor.observe_cone.side_effect=lambda *a,**k: respawn(detection())
                if phase=='approach':
                    nav.goto.side_effect=lambda *a,**k: respawn(True)
                if phase=='final_pose':
                    calls = [0]
                    def pose():
                        calls[0] += 1
                        return respawn((15.5,24.05,1.57)) if calls[0]==2 else (15.5,22.3,1.57)
                    nav.pose.side_effect=pose
                with patch('custom.mission_2.mission_2.Navigator', return_value=nav), \
                     patch('custom.mission_2.mission_2.ConeSensor', return_value=sensor), \
                     patch('custom.mission_2.mission_2.time.sleep',
                           side_effect=(lambda _: respawn(None)) if phase=='settle' else None):
                    result = Mission2.run(mission)
                self.assertFalse(result['reached'])
                self.assertTrue(result['restart_required'])
                self.assertTrue(result['observations_invalidated'])
                self.assertEqual(result['measurements'], [])
                self.assertIsNone(result['final_stop'])
                self.assertNotIn('cone_xy', result)
                mission.stop.assert_called()
                if phase in ('begin','route','survey'):
                    nav.goto.assert_not_called()

    def test_respawn_during_sensor_settling_discards_snapshot(self):
        mission = mission_stub(stop=Mock(), log=Mock(), latest=Mock())
        sensor = ConeSensor(mission, Mock())
        sensor._snapshot.messages['old'] = object()
        def respawn(_):
            mission.respawn_count += 1
        with patch('custom.mission_2.perception.time.sleep', side_effect=respawn):
            with self.assertRaisesRegex(RuntimeError, 'respawn'):
                sensor.observe_cone('blue', 3.)
        mission.latest.assert_not_called()
        self.assertEqual(sensor._snapshot.messages, {})

    def test_public_notification_stops_and_blocks_subsequent_motion(self):
        node = Mission2.__new__(Mission2)
        node._sensor_cv = threading.Condition()
        node._sensor_messages = {}
        node._respawn_count = node._run_respawn_count = 0
        with patch.object(SensorMission, 'cmd_vel', autospec=True) as publish:
            node.cmd_vel(.1,0,0)
            node._on_public_respawn(object())
            with self.assertRaisesRegex(RuntimeError, 'respawn'):
                node.cmd_vel(.1,0,0)
            node.stop()  # zero commands must remain usable during cleanup
        self.assertEqual(node.respawn_count, 1)
        self.assertEqual(publish.call_args_list[0].args[1:], (.1,0,0))
        self.assertTrue(all(call.args[1:]==(0.,0.,0.) for call in publish.call_args_list[1:]))


if __name__ == '__main__':
    unittest.main()
