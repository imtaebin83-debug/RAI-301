"""CPU regressions; these do not assert simulator success."""
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from sensor_msgs.msg import Image

from custom.mission_4.mission_4 import Mission4
from custom.mission_4 import evaluate
from custom.vlm import ROOM_LABELS, assign_rooms, classify_views, combine_views
from custom.mission_4.room_scoring import calibrated_observation, score_room_image


def observation(label):
    return {'label': label, 'scores': {key: -1. if key == label else -9. for key in ROOM_LABELS}}


class RoomEvidenceTests(unittest.TestCase):
    def test_calibration_corrects_independent_bias_without_changing_joint_margin(self):
        bias = {'classroom': -12., 'medical': -2., 'cafeteria': -15.}
        raw = [{'label': 'medical', 'scores': {
                    label: bias[label] + (4. if label == actual else -1.)
                    for label in ROOM_LABELS}} for actual in ROOM_LABELS]
        corrected = [calibrated_observation(item, {'scores': bias}, (640,480)) for item in raw]
        self.assertEqual([item['label'] for item in corrected], list(ROOM_LABELS))
        before, before_details = assign_rooms(raw, min_margin=.25)
        after, after_details = assign_rooms(corrected, min_margin=.25)
        self.assertEqual(before, after)
        self.assertAlmostEqual(before_details['runner_up_margin'], after_details['runner_up_margin'])
        self.assertEqual(raw[0]['label'], 'medical')
        self.assertEqual(corrected[0]['raw_scores'], raw[0]['scores'])
        self.assertEqual(corrected[0]['text'], corrected[0]['label'])

    def test_invalid_neutral_scores_and_overflow_are_rejected(self):
        raw = observation('classroom')
        for scores in ({'classroom': 0.}, dict.fromkeys(ROOM_LABELS, float('nan')),
                       dict.fromkeys(ROOM_LABELS, True)):
            with self.assertRaises(ValueError):
                calibrated_observation(raw, {'scores': scores}, (640,480))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            calibrated_observation({'scores': dict.fromkeys(ROOM_LABELS, 1e308)},
                                   {'scores': dict.fromkeys(ROOM_LABELS, -1e308)}, (640,480))

    def test_neutral_reference_is_cached_per_resolution_and_is_not_a_room_view(self):
        import tempfile
        from pathlib import Path
        from PIL import Image as PILImage
        seen = []
        def score(path):
            with PILImage.open(path) as image:
                seen.append((image.size, image.getpixel((0,0))))
            return observation('medical')
        cache = {}
        with tempfile.TemporaryDirectory() as directory:
            for index, size in enumerate(((40,30), (40,30), (20,15))):
                path = Path(directory)/f'{index}.png'
                PILImage.new('RGB', size, (10,20,30)).save(path)
                result = score_room_image(path, score, cache)
                self.assertEqual(result['calibration']['size'], list(size))
        self.assertEqual([color for _,color in seen],
                         [(127,127,127), (10,20,30), (10,20,30), (127,127,127), (10,20,30)])
        self.assertEqual(len(cache), 2)

    def test_manifest_has_six_distinct_orders_and_prior_mission_endpoint(self):
        cases = evaluate.cases()
        self.assertEqual(len({tuple(case['room_order']) for case in cases}), 6)
        self.assertTrue(any(case['missions'] == [2, 4] for case in cases))
        self.assertEqual(sum(case['split'] == 'held-out' for case in cases), 3)

    def test_all_six_permutations_use_view_scores(self):
        for order in itertools.permutations(ROOM_LABELS):
            evidence = [combine_views([observation(label), observation(label)]) for label in order]
            self.assertEqual(assign_rooms(evidence, min_margin=.25)[0], list(order))

    def test_repeated_views_are_averaged_not_counted_as_independent_rooms(self):
        item = observation('classroom')
        self.assertEqual(combine_views([item, item])['scores'], item['scores'])

    def test_disagreeing_api_views_do_not_fabricate_labels(self):
        with self.assertRaises(ValueError):
            combine_views([{'label': 'medical'}, {'label': 'classroom'}])

    def test_nonfinite_scores_and_empty_views_are_rejected(self):
        item = observation('classroom')
        item['scores']['medical'] = float('nan')
        for views in ([], [item]):
            with self.assertRaises(ValueError):
                combine_views(views)

    def test_small_assignment_margin_is_rejected(self):
        items = [observation(label) for label in ROOM_LABELS]
        items[1]['scores'] = {'medical': -1., 'cafeteria': -1.01, 'classroom': -9.}
        items[2]['scores'] = {'medical': -1.01, 'cafeteria': -1., 'classroom': -9.}
        with self.assertRaises(ValueError):
            assign_rooms(items, min_margin=.25)

    def test_batch_timeout_and_cpu_only_worker(self):
        with patch('custom.vlm.subprocess.run') as run, \
             patch.dict('os.environ', {}, clear=True):
            run.return_value = SimpleNamespace(returncode=0, stdout='[{"label":"medical"}]')
            image = Mock()
            self.assertEqual(classify_views([image], timeout=3)[0]['label'], 'medical')
            self.assertEqual(run.call_args.kwargs['timeout'], 3)
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')


class CameraTests(unittest.TestCase):
    def image(self, sec, encoding='bgr8'):
        item = Image()
        item.header.stamp.sec = sec
        item.encoding = encoding
        item.width, item.height, item.step = 2, 2, 8
        item.data = bytes([1, 2, 3, 4, 5, 6, 255, 255] * 2)
        return item

    def test_two_post_alignment_frames_required_and_padding_ignored(self):
        mission = SimpleNamespace(latest=Mock(side_effect=[self.image(1), self.image(1),
                                                         self.image(2), self.image(3)]))
        with patch('custom.mission_4.mission_4.time.sleep'):
            image = Mission4._image(mission, timeout=1.)
        np.testing.assert_array_equal(np.asarray(image)[0], [[3, 2, 1], [6, 5, 4]])
        self.assertEqual(mission._last_image_stamp, [3, 0])

    def test_stalled_camera_times_out(self):
        mission = SimpleNamespace(latest=Mock(return_value=self.image(1)))
        with patch('custom.mission_4.mission_4.time.monotonic', side_effect=[0., 0., .1, 2.]), \
             patch('custom.mission_4.mission_4.time.sleep'):
            with self.assertRaises(TimeoutError):
                Mission4._image(mission, timeout=.5)

    def test_begin_failure_stops_and_reports_stage(self):
        mission = SimpleNamespace(begin_mission=Mock(side_effect=TimeoutError('missing sim')),
                                  stop=Mock(), log=Mock(), result=lambda **data: data)
        result = Mission4.run(mission)
        self.assertEqual(result['failed_stage'], 'begin')
        self.assertEqual(result['rooms'], [])
        mission.stop.assert_called_once()

    def test_travel_uses_one_central_corridor_goal(self):
        nav = Mock()
        Mission4._travel(SimpleNamespace(_navigate=lambda nav, action, timeout: action(timeout)), nav, 17.5, lambda limit: limit)
        nav.route_to.assert_called_once_with(16.0, 17.5, tolerance=.18, timeout=100)
        nav.goto.assert_not_called()

    def test_failed_waypoint_prevents_room_entry(self):
        nav = Mock()
        nav.pose.return_value = (16., 11., 0.)
        nav.route_to.return_value = False
        with self.assertRaises(RuntimeError):
            Mission4._travel(SimpleNamespace(_navigate=lambda nav, action, timeout: action(timeout)), nav, 17.5, lambda limit: limit)
        nav.goto.assert_not_called()

    def test_correct_room_labels_do_not_hide_failed_delivery(self):
        mission = SimpleNamespace(begin_mission=Mock(return_value={}), stop=Mock(),
            log=Mock(), result=lambda **data: data, _travel=Mock(), _face_room=Mock(),
            _image=Mock(return_value=Mock()), _last_image_stamp=[2, 0],
            _navigate=lambda nav, action, timeout: action(timeout))
        nav = Mock()
        nav.pose.return_value = (16.7, 3.59, 0.)
        nav.goto.return_value = False
        labels = ('medical', 'cafeteria', 'classroom')
        views = [observation(label) for label in labels]
        with patch('custom.mission_4.mission_4.Navigator', return_value=nav), \
             patch('custom.mission_4.mission_4.classify_views', return_value=views), \
             patch('custom.mission_4.mission_4.time.sleep'), \
             patch.dict('os.environ', {}, clear=True):
            result = Mission4.run(mission)
        self.assertEqual(result['rooms'], list(labels))
        self.assertEqual(result['failed_stage'], 'classroom entry')
        self.assertIn('could not enter', result['error'])
        nav.goto.assert_called_once_with(20.0, 3.59, tolerance=.15,
                                        timeout=40, speed=.20)
        mission.stop.assert_called()



    def test_ambiguous_survey_gets_one_crop_retry_before_delivery(self):
        from PIL import Image as PILImage
        image = PILImage.new('RGB', (640, 480))
        mission = SimpleNamespace(begin_mission=Mock(return_value={}), stop=Mock(),
            log=Mock(), result=lambda **data: data, _travel=Mock(), _face_room=Mock(),
            _image=Mock(return_value=image), _last_image_stamp=[2, 0],
            _navigate=lambda nav, action, timeout: action(timeout))
        nav = Mock()
        nav.pose.return_value = (16., 3.59, 0.)
        labels = ('medical', 'cafeteria', 'classroom')
        ambiguous = [{'label': 'medical', 'scores': {label: -5. for label in ROOM_LABELS}}] * 3
        with patch('custom.mission_4.mission_4.Navigator', return_value=nav), \
             patch('custom.mission_4.mission_4.classify_views', side_effect=[
                 ambiguous, [observation(label) for label in labels]]) as classify, \
             patch('custom.mission_4.mission_4.time.sleep'), \
             patch.dict('os.environ', {}, clear=True):
            result = Mission4.run(mission)
        self.assertEqual(result['rooms'], list(labels))
        self.assertNotIn('error', result)
        self.assertEqual(classify.call_count, 2)
        self.assertTrue(all(im.size == (512, 384) for im in classify.call_args.args[0]))
        self.assertEqual(len(result['vlm_retries']), 1)
        self.assertTrue(all(len(o['views']) == 2 for o in result['vlm_observations']))
        nav.goto.assert_called_once()

    def test_still_ambiguous_crop_retry_never_enters_a_room(self):
        from PIL import Image as PILImage
        mission = SimpleNamespace(begin_mission=Mock(return_value={}), stop=Mock(),
            log=Mock(), result=lambda **data: data, _travel=Mock(), _face_room=Mock(),
            _image=Mock(return_value=PILImage.new('RGB', (640, 480))), _last_image_stamp=[2, 0],
            _navigate=lambda nav, action, timeout: action(timeout))
        nav = Mock()
        nav.pose.return_value = (16., 3.59, 0.)
        ambiguous = [{'label': 'medical', 'scores': {label: -5. for label in ROOM_LABELS}}] * 3
        with patch('custom.mission_4.mission_4.Navigator', return_value=nav), \
             patch('custom.mission_4.mission_4.classify_views', return_value=ambiguous) as classify, \
             patch('custom.mission_4.mission_4.time.sleep'), \
             patch.dict('os.environ', {}, clear=True):
            result = Mission4.run(mission)
        self.assertEqual(classify.call_count, 2)
        self.assertEqual(result['rooms'], [])
        self.assertEqual(result['failed_stage'], 'VLM crop retry')
        nav.goto.assert_not_called()
        mission.stop.assert_called()


class PoseRecoveryTests(unittest.TestCase):
    def mission(self, stamps):
        def message(stamp):
            return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp, nanosec=0)))
        mission = SimpleNamespace(stop=Mock(), log=Mock(), _recovery_attempts=0,
                                  _pose_recoveries=[], latest=Mock(side_effect=[message(s) for s in stamps]))
        return mission

    def test_retry_waits_for_sustained_poses_and_keeps_deadline(self):
        mission = self.mission([10, 10] + list(range(11, 60)))
        nav, clock = Mock(), [0.]
        def sleep(seconds):
            clock[0] += seconds
        action = Mock(side_effect=[RuntimeError('Stale localization: test'), True])
        with patch('custom.mission_4.mission_4.time.monotonic', side_effect=lambda: clock[0]), \
             patch('custom.mission_4.mission_4.time.sleep', side_effect=sleep):
            self.assertTrue(Mission4._navigate(mission, nav, action, 20))
        self.assertGreaterEqual(mission.latest.call_count, 23)
        self.assertGreaterEqual(clock[0], 1.)
        self.assertGreaterEqual(mission.stop.call_count, 4)
        nav.pose.assert_called_once()
        self.assertLess(action.call_args_list[1].args[0], action.call_args_list[0].args[0])
        self.assertTrue(mission._pose_recoveries[0]['recovered'])

    def test_intermittent_pose_bursts_do_not_resume_movement(self):
        # Two new stamps alone must not count as a sustained recovery.
        mission = self.mission(([10, 11, 12] + [12] * 20) * 9)
        clock = [0.]
        def sleep(seconds):
            clock[0] += seconds
        action = Mock(side_effect=RuntimeError('Stale localization: test'))
        with patch('custom.mission_4.mission_4.time.monotonic', side_effect=lambda: clock[0]), \
             patch('custom.mission_4.mission_4.time.sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, 'did not recover'):
                Mission4._navigate(mission, Mock(), action, 20)
        action.assert_called_once()

    def test_permanent_outage_never_retries_movement(self):
        mission = self.mission([10] * 200)
        nav, clock = Mock(), [0.]
        def sleep(seconds):
            clock[0] += seconds
        action = Mock(side_effect=RuntimeError('Stale localization: test'))
        with patch('custom.mission_4.mission_4.time.monotonic', side_effect=lambda: clock[0]), \
             patch('custom.mission_4.mission_4.time.sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, 'did not recover'):
                Mission4._navigate(mission, nav, action, 20)
        action.assert_called_once()
        self.assertLessEqual(clock[0], 8.01)
        self.assertFalse(mission._pose_recoveries[0]['recovered'])

    def test_unrelated_errors_and_exhausted_budget_are_not_retried(self):
        for error, attempts in [('bad target', 0), ('Stale localization: test', 3)]:
            mission = self.mission([])
            mission._recovery_attempts = attempts
            action = Mock(side_effect=RuntimeError(error))
            with self.assertRaisesRegex(RuntimeError, error):
                Mission4._navigate(mission, Mock(), action, 20)
            action.assert_called_once()
            mission.latest.assert_not_called()
            mission.stop.assert_called()

    def test_recovery_wait_cannot_extend_the_leg_deadline(self):
        mission = self.mission([10] * 100)
        clock = [0.]
        def sleep(seconds):
            clock[0] += seconds
        action = Mock(side_effect=RuntimeError('Stale localization: test'))
        with patch('custom.mission_4.mission_4.time.monotonic', side_effect=lambda: clock[0]), \
             patch('custom.mission_4.mission_4.time.sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, 'did not recover'):
                Mission4._navigate(mission, Mock(), action, .4)
        self.assertLessEqual(clock[0], .4)
        action.assert_called_once()

    def test_expired_deadline_never_starts_navigation(self):
        mission = self.mission([])
        action = Mock()
        with self.assertRaises(TimeoutError):
            Mission4._navigate(mission, Mock(), action, 0)
        action.assert_not_called()
        mission.stop.assert_called()

if __name__ == '__main__':
    unittest.main()
