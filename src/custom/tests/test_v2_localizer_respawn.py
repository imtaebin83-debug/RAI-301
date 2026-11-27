"""Public respawn reanchoring without simulator status access."""
from types import SimpleNamespace as S
import unittest
import numpy as np
from custom.localization import Localizer


class RespawnTests(unittest.TestCase):
    def test_public_empty_counter_resets_to_documented_start_and_skips_cached_scan(self):
        scan = S(header=S(stamp=S(sec=12, nanosec=34)))
        mission = S(respawn_count=1, latest=lambda *a, **kw: scan)
        loc = object.__new__(Localizer)
        loc.mission = mission
        loc._last_respawn_count = 0
        loc._new_odometry = lambda: 'new sensor map'
        loc._world_lidar0 = np.eye(4)
        loc._recovery_blocked = True
        loc._recovery_times = [1., 2.]
        self.assertEqual(loc.update(), (16., 11., 0.))
        self.assertIsNone(loc._world_lidar0)
        self.assertIsNone(loc.measurement_stamp)
        self.assertEqual(loc._attempt_stamp, (12, 34))
        self.assertEqual(loc._anchor, {'pos': [16., 11.], 'yaw_deg': 0.})
        self.assertEqual(loc.quality, 0.)
        self.assertEqual(loc._recovery_times, [])
        self.assertFalse(loc._recovery_blocked)
        # The same cached cloud cannot initialize the map after the respawn.
        self.assertEqual(loc.update(), (16., 11., 0.))
        self.assertIsNone(loc._world_lidar0)


if __name__ == '__main__':
    unittest.main()
