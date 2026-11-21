"""Room assignment uses image scores and a public uniqueness prior only."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("tested_vlm", Path(__file__).resolve().parents[1] / "vlm.py")
vlm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vlm)


class RoomAssignmentTests(unittest.TestCase):
    def observations(self):
        return [
            {"label": "classroom", "scores": {"classroom": -1., "medical": -8., "cafeteria": -9.}},
            {"label": "medical", "scores": {"classroom": -9., "medical": -1., "cafeteria": -7.}},
            {"label": "medical", "scores": {"classroom": -9., "medical": -1., "cafeteria": -2.}},
        ]

    def test_duplicate_labels_use_actual_scores_for_joint_assignment(self):
        result, detail = vlm.assign_rooms(self.observations())
        self.assertEqual(result, ["classroom", "medical", "cafeteria"])
        self.assertEqual(detail["unconstrained_labels"], ["classroom", "medical", "medical"])
        self.assertGreater(detail["runner_up_margin"], 0)

    def test_no_fixed_spatial_room_order(self):
        observations = self.observations()[::-1]
        self.assertEqual(vlm.assign_rooms(observations)[0], ["cafeteria", "medical", "classroom"])

    def test_duplicate_labels_without_scores_rejected(self):
        with self.assertRaises(ValueError):
            vlm.assign_rooms([{"label": "medical"}] * 3)

    def test_identical_scores_cannot_fabricate_assignment(self):
        scores = {label: -1. for label in vlm.ROOM_LABELS}
        with self.assertRaises(ValueError):
            vlm.assign_rooms([{"label": "medical", "scores": scores}] * 3)

    def test_nonfinite_scores_rejected(self):
        observations = self.observations()
        observations[0]["scores"]["medical"] = float("nan")
        with self.assertRaises(ValueError):
            vlm.assign_rooms(observations)


if __name__ == "__main__":
    unittest.main()
