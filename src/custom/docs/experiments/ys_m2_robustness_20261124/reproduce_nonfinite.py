"""Evaluation only: run new regression against the published pre-fix controller."""
import subprocess
import unittest
from custom.mission_2 import mission_2

source = subprocess.check_output(['git', 'show',
    '447889e:src/custom/mission_2/mission_2.py'], text=True)
exec(compile(source, mission_2.__file__, 'exec'), mission_2.__dict__)
from custom.mission_2.test_mission_2 import MissionFailureTests
suite = unittest.TestSuite([MissionFailureTests(
    'test_nonfinite_pose_cannot_start_approach_or_claim_a_successful_stop')])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
