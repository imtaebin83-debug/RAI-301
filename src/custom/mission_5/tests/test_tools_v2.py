"""Pure starter-v2 tool checks; no ROS node, simulator, GPU or reset is used."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np
import yaml

from custom.mission_5 import benchmark, collect, evaluate


class StarterV2ToolTests(unittest.TestCase):
    def test_removed_collect_retry_fails_before_output_or_ros(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'not-created' / 'episodes'
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                collect.main(['--out', str(out), '--retry'])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('removed /panda/retry', error.getvalue())
            self.assertIn('fresh simulator', error.getvalue())
            self.assertFalse(out.parent.exists())

    def test_evaluate_requires_one_trial_and_explains_fresh_process_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'not-created' / 'result.json'
            for episodes in [0, 2, 50]:
                error = io.StringIO()
                with self.subTest(episodes=episodes), contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                    evaluate.main(['--out', str(out), '--episodes', str(episodes)])
                self.assertIn('benchmark', error.getvalue())
                self.assertIn('fresh simulator processes', error.getvalue())
            self.assertFalse(out.parent.exists())
            for compatibility in [[], ['--keep-first']]:
                args = evaluate.parse_args(['--out', str(out), *compatibility])
                self.assertEqual(args.episodes, 1)
                self.assertEqual(args.keep_first, bool(compatibility))

    def fake_evaluation_modules(self, controller_error=None):
        # Deliberately expose no publisher/randomize/judge on the mission node.
        # Referee must receive the original runner's link, not the mission node.
        result = SimpleNamespace(mission=5)
        node = SimpleNamespace(run=mock.Mock(return_value=result, side_effect=controller_error))
        runner = mock.MagicMock()
        runner.__enter__.return_value = runner
        runner.api = SimpleNamespace(wait_for_sim=mock.Mock())
        runner.start.return_value = node
        referee = mock.Mock()
        referee.close.return_value = {'passed': 1}
        modules = {
            'mission_api': SimpleNamespace(MissionRunner=mock.Mock(return_value=runner)),
            'referee': SimpleNamespace(Referee=mock.Mock(return_value=referee)),
            'custom.mission_5.mission_5': SimpleNamespace(Mission5=type('Mission5', (), {})),
        }
        return modules, runner, referee, node, result

    def test_evaluate_runs_current_placement_once_and_judges_through_simlink(self):
        modules, runner, referee, node, result = self.fake_evaluation_modules()
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(['--out', str(Path(temporary) / 'result.json')])
        runner.api.wait_for_sim.assert_called_once_with(timeout=40)
        runner.start.assert_called_once()
        node.run.assert_called_once_with()
        referee.begin.assert_called_once_with(5)
        referee.finish.assert_called_once_with(runner.api, result)
        runner.finish.assert_called_once_with(node)
        referee.close.assert_called_once_with(complete=True)
        run = modules['referee'].Referee.call_args.kwargs['run']
        self.assertEqual(run['placement'], 'existing')
        self.assertFalse(run['reset_available'])
        self.assertEqual(run['episodes'], 1)

    def test_evaluate_releases_owned_node_after_controller_error(self):
        modules, runner, referee, node, _ = self.fake_evaluation_modules(RuntimeError('controller failed'))
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, 'controller failed'):
                evaluate.main(['--out', str(Path(temporary) / 'result.json')])
        referee.finish.assert_not_called()
        runner.finish.assert_called_once_with(node)

    def test_collection_metadata_uses_robot_sensors_panda(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom = root / 'custom'
            (custom / 'mission_5').mkdir(parents=True)
            panda = {'agentview': {'pos': [1.28, 0, .92], 'quat': [1, 0, 0, 0]},
                     'wrist': {'pos': [.05, 0, 0], 'fovy': 75}}
            robot = custom / 'robot.yaml'
            robot.write_text(yaml.safe_dump({'sensors': {'panda': panda}, 'walking': {}}))
            # Contradictory legacy file must not be read, even if one remains.
            (custom / 'sensors.yaml').write_text('panda: {legacy: true}\n')
            path = root / 'episode.h5'
            with mock.patch.object(collect, '__file__', str(custom / 'mission_5' / 'collect.py')):
                writer = collect.Writer(path, 10., 'synthetic-placement')
                writer.finish({'success': True}, 'synthetic-test')
            with h5py.File(path) as episode:
                self.assertEqual(json.loads(episode.attrs['camera_metadata_json']), panda)
                self.assertEqual(episode.attrs['camera_calibration_sha256'],
                                 hashlib.sha256(robot.read_bytes()).hexdigest())

    def test_public_fovy_pose_reproduces_agentview_calibration_without_camera_info(self):
        from custom.mission_5.perception import CameraModel
        config = {'fovy': 46., 'pos': [1.28, 0., .92],
                  'quat': [.66331780, .24496836, .24496836, .66331780]}
        camera = collect.panda_camera(config)
        np.testing.assert_allclose(camera.k, [[301.549103, 0., 127.5],
                                             [0., 301.549103, 127.5], [0., 0., 1.]], rtol=1e-7)
        np.testing.assert_allclose(camera.base_from_optical,
                                   CameraModel.panda_agentview_default().base_from_optical, atol=1e-12)
        for fovy in [0., 180., float('nan')]:
            with self.subTest(fovy=fovy), self.assertRaisesRegex(ValueError, 'fovy'):
                collect.panda_camera({**config, 'fovy': fovy})

    def test_benchmark_tracks_robot_yaml_and_ignores_removed_configuration_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom = root / 'custom'
            mission = custom / 'mission_5'
            mission.mkdir(parents=True)
            checkpoint = mission / 'keyposes.json'
            checkpoint.write_text('{}')
            (mission / 'policy.py').write_text('POLICY = 1\n')
            (root / 'main.py').write_text('# synthetic official main\n')
            robot = custom / 'robot.yaml'
            robot.write_text('sensors: {}\nwalking: {}\n')
            sensors = custom / 'sensors.yaml'
            walking = custom / 'walking.yaml'
            sensors.write_text('legacy: 1\n')
            walking.write_text('legacy: 1\n')
            first = benchmark.fingerprints(root, checkpoint)
            self.assertIn('custom/robot.yaml', first['files'])
            self.assertIn('custom/mission_5/policy.py', first['files'])
            self.assertNotIn('custom/sensors.yaml', first['files'])
            self.assertNotIn('custom/walking.yaml', first['files'])
            sensors.write_text('legacy: 2\n')
            walking.write_text('legacy: 2\n')
            self.assertEqual(first, benchmark.fingerprints(root, checkpoint))
            robot.write_text('sensors: {panda: {}}\nwalking: {}\n')
            changed = benchmark.fingerprints(root, checkpoint)
            self.assertNotEqual(first['files']['custom/robot.yaml'], changed['files']['custom/robot.yaml'])
            self.assertEqual(first['checkpoint'], changed['checkpoint'])


if __name__ == '__main__':
    unittest.main()
