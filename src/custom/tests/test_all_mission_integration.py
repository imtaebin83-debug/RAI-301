"""CPU contracts for the all-mission v2 integration."""
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from custom.sensor_mission import Mission
from custom.mission_3.mission_3 import Mission3
from custom.mission_6.mission_6 import Mission6, in_goal
from custom.testing.all_missions import cases


class IntegrationTests(unittest.TestCase):
    def test_all_six_use_public_sensor_adapter(self):
        import importlib
        for n in range(1,7):
            cls=getattr(importlib.import_module(f'custom.mission_{n}.mission_{n}'),f'Mission{n}')
            self.assertTrue(issubclass(cls,Mission))

    def test_manifest_runs_all_six_without_intermediate_reset(self):
        chain=next(c for c in cases() if c['id']=='all-six-blue')
        self.assertEqual(chain['missions'],[1,2,3,4,5,6])
        self.assertEqual(chain['m2_color'],'blue')

    def test_cliff_begin_error_still_stops(self):
        node=SimpleNamespace(begin_mission=Mock(side_effect=TimeoutError('no simulator')),
            result=lambda **data:data, stop=Mock(), gap_falls=0)
        self.assertIn('no simulator',Mission3.run(node)['error'])
        node.stop.assert_called_once()

    def test_cliff_routes_to_entry_before_course_waypoints(self):
        node=SimpleNamespace(begin_mission=Mock(return_value=None),result=lambda **d:d,
            stop=Mock(),log=Mock(),gap_falls=0,ENTRY=Mission3.ENTRY,COURSE=Mission3.COURSE)
        nav=Mock()
        nav.route_to.return_value=False
        with patch('custom.mission_3.mission_3.Navigator',return_value=nav):
            result=Mission3.run(node)
        self.assertIn('public doors',result['error'])
        nav.goto.assert_not_called()

    def test_soccer_begin_error_still_stops(self):
        node=SimpleNamespace(begin_mission=Mock(side_effect=TimeoutError('no simulator')),
            result=lambda **data:data, stop=Mock(),respawn_count=0)
        self.assertIn('no simulator',Mission6.run(node)['error'])
        node.stop.assert_called_once()

    def test_soccer_uses_shared_navigation_and_remaining_budget(self):
        node=SimpleNamespace(_check_run=Mock(),nav=Mock(),deadline=time.monotonic()+2.)
        Mission6._goto(node,10.,5.,timeout=40.,speed=.28)
        args=node.nav.goto.call_args
        self.assertEqual(args.args,(10.,5.))
        self.assertLessEqual(args.kwargs['timeout'],2.)
        self.assertEqual(args.kwargs['speed'],.28)

    def test_soccer_guard_rejects_respawn_and_expired_budget(self):
        node=SimpleNamespace(respawn_count=1,_run_respawn_count=0,
                             deadline=time.monotonic()+10,stop=Mock())
        with self.assertRaisesRegex(RuntimeError,'respawn'):
            Mission6._check_run(node)
        node.respawn_count=0
        node.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError):
            Mission6._check_run(node)

    def test_goal_hint_excludes_back_wall_and_nonfinite_ball(self):
        self.assertTrue(in_goal((9.25,1.4)))
        for xy in ((9.25,-1.),(9.25,8.),(5.,1.4),(float('nan'),1.4)):
            self.assertFalse(in_goal(xy))


if __name__=='__main__':
    unittest.main()
