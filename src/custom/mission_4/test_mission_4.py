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


def observation(label):
    return {'label': label, 'scores': {key: -1. if key == label else -9. for key in ROOM_LABELS}}


class RoomEvidenceTests(unittest.TestCase):
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
        Mission4._travel(SimpleNamespace(), nav, 17.5, lambda limit: limit)
        nav.route_to.assert_called_once_with(16.0, 17.5, tolerance=.18, timeout=100)
        nav.goto.assert_not_called()

    def test_failed_waypoint_prevents_room_entry(self):
        nav = Mock()
        nav.pose.return_value = (16., 11., 0.)
        nav.route_to.return_value = False
        with self.assertRaises(RuntimeError):
            Mission4._travel(SimpleNamespace(), nav, 17.5, lambda limit: limit)
        nav.goto.assert_not_called()

    def test_correct_room_labels_do_not_hide_failed_delivery(self):
        mission = SimpleNamespace(begin_mission=Mock(return_value={}), stop=Mock(),
            log=Mock(), result=lambda **data: data, _travel=Mock(), _face_room=Mock(),
            _image=Mock(return_value=Mock()), _last_image_stamp=[2, 0])
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


if __name__ == '__main__':
    unittest.main()
