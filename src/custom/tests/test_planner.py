"""Offline planner contract checks, including API boundaries without network calls.

Run: python -m unittest discover -s custom/tests -p test_planner.py -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("tested_planner",
    Path(__file__).resolve().parents[1] / "planner" / "planner.py")
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {"HAIC_ALLOW_RULE_PLANNER": "1"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_english_order_and_color(self):
        self.assertEqual(planner.plan("Go to the blue cone, then score a goal."),
                         [(2, {"color": "blue"}), 6])

    def test_mixed_language_order(self):
        self.assertEqual(planner.plan("Score a goal, 그다음 주황 콘, then cross the cliff."),
                         [6, (2, {"color": "orange"}), 3])

    def test_goal_coordinates_are_not_soccer(self):
        self.assertEqual(planner.plan("Navigate to goal (14.2, 7.7), then cross the gaps."),
                         [(1, {"goal": [14.2, 7.7]}), 3])

    def test_negative_coordinates_remain_coordinates(self):
        self.assertEqual(planner.plan("좌표 (-1.25, 2.5) 이동"),
                         [(1, {"goal": [-1.25, 2.5]})])

    def test_explicit_numeric_sequence(self):
        self.assertEqual(planner.plan("missions 6, 3, 4"), [6, 3, 4])

    def test_mixed_explicit_arguments(self):
        self.assertEqual(planner.plan("미션 2 blue cone 다음 미션 1 (16, 5)"),
                         [(2, {"color": "blue"}), (1, {"goal": [16.0, 5.0]})])

    def test_number_does_not_supply_goal(self):
        with self.assertRaises(ValueError):
            planner.plan("missions 1, 2")

    def test_bracketed_numbers_are_not_coordinate(self):
        with self.assertRaises(ValueError):
            planner.plan("missions [1, 2]")

    def test_missing_color_rejected(self):
        with self.assertRaises(ValueError):
            planner.plan("find the cone")

    def test_unknown_or_empty_instruction_rejected(self):
        for instruction in ("", "  ", "make coffee"):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                planner.plan(instruction)

    def test_development_fallback_is_visible(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            planner.plan("score a goal")
        self.assertIn("DEVELOPMENT FALLBACK", stream.getvalue())

    def test_strict_gate_refuses_missing_llm(self):
        with patch.dict("os.environ", {"HAIC_REQUIRE_LLM": "1"}):
            with patch.object(planner.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 1, "", "weights missing")), self.assertRaises(RuntimeError):
                planner.plan("score a goal")

    def test_strict_gate_refuses_partial_llm_configuration(self):
        for extra in ({"HAIC_LLM_MODEL": "example"}, {"HAIC_LLM_API_KEY": "dummy"}):
            with self.subTest(extra=extra), patch.dict("os.environ", {
                    "HAIC_REQUIRE_LLM": "1", **extra}):
                with patch.object(planner.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], 1, "", "weights missing")), self.assertRaises(RuntimeError):
                    planner.plan("score a goal")

    def test_duplicate_numeric_steps_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            planner.plan("missions 6, 6")


class ValidationTests(unittest.TestCase):
    def test_actual_model_trailing_quote_preserves_all_steps(self):
        text = '{"steps":[{"mission":2,"params":{"color":"blue"}},{"mission":6,"params":{}}]}"'
        self.assertEqual(planner.parse_plan(text), [(2, {"color": "blue"}), 6])

    def test_multiple_json_plans_not_silently_merged_or_dropped(self):
        with self.assertRaises(ValueError):
            planner.parse_plan('{"steps":[{"mission":6}]}{"steps":[{"mission":3}]}')

    def test_trailing_mission_prose_rejected(self):
        with self.assertRaises(ValueError):
            planner.parse_plan('{"steps":[{"mission":6}]} then mission 3')

    def test_duplicate_json_keys_cannot_silently_drop_a_mission(self):
        text = '{"steps":[{"mission":2,"params":{"color":"blue"}}],"steps":[{"mission":6}]}'
        with self.assertRaisesRegex(ValueError, "duplicate"):
            planner.parse_plan(text)

    def test_nested_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            planner.parse_plan('{"steps":[{"mission":2,"mission":6}]}')

    def test_extra_root_field_cannot_hide_another_mission(self):
        with self.assertRaises(ValueError):
            planner.parse_plan('{"steps":[{"mission":2,"params":{"color":"blue"}}],"next":{"mission":6}}')

    def test_extra_step_field_cannot_hide_another_mission(self):
        with self.assertRaises(ValueError):
            planner.parse_plan('{"steps":[{"mission":6,"next":{"mission":3}}]}')

    def test_repeated_steps_from_llm_preserved(self):
        self.assertEqual(planner.validate({"steps": [{"mission": 6}, {"mission": 6}]}), [6, 6])

    def test_rejects_invalid_shapes(self):
        for payload in (None, [], {}, {"steps": []}, {"steps": "6"}, {"steps": [6]},
                        {"steps": [{"mission": True}]}, {"steps": [{"mission": 7}]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                planner.validate(payload)

    def test_rejects_invalid_goal_values(self):
        for goal in ([1], [1, 2, 3], [True, 2], [float("nan"), 2], [float("inf"), 2],
                     ["14", 3], None):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                planner.validate({"steps": [{"mission": 1, "params": {"goal": goal}}]})

    def test_rejects_hidden_or_unknown_parameters(self):
        for n, params in ((1, {"goal": [1, 2], "truth": [3, 4]}),
                          (2, {"color": "blue", "cone_pos": [1, 2]}),
                          (6, {"ball": [1, 2]})):
            with self.subTest(n=n), self.assertRaises(ValueError):
                planner.validate({"steps": [{"mission": n, "params": params}]})

    def test_invalid_cone_colors(self):
        for color in ("red", "Blue", None, 2):
            with self.subTest(color=color), self.assertRaises(ValueError):
                planner.validate({"steps": [{"mission": 2, "params": {"color": color}}]})


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {
            "HAIC_REQUIRE_LLM": "1", "HAIC_LLM_API_KEY": "offline-dummy",
            "HAIC_LLM_MODEL": "offline-model", "HAIC_LLM_BASE_URL": "https://example.invalid/v1"
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def _response(self, content):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    def test_strict_configured_llm_path_and_fenced_json(self):
        text = '```json\n{"steps":[{"mission":6},{"mission":2,"params":{"color":"green"}}]}\n```'
        with patch.object(planner.urllib.request, "urlopen", return_value=self._response(text)) as call:
            self.assertEqual(planner.plan("soccer then green cone"), [6, (2, {"color": "green"})])
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(call.call_args.kwargs["timeout"], 30)
        body = json.loads(request.data)
        self.assertEqual(body["messages"][-1]["content"], "soccer then green cone")

    def test_malformed_llm_json_never_silently_falls_back(self):
        with patch.object(planner.urllib.request, "urlopen", return_value=self._response("not JSON")):
            with self.assertRaises(json.JSONDecodeError):
                planner.plan("score a goal")

    def test_network_error_never_silently_falls_back(self):
        with patch.object(planner.urllib.request, "urlopen", side_effect=TimeoutError("offline")):
            with self.assertRaises(TimeoutError):
                planner.plan("score a goal")


class LocalLlmContractTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_default_path_invokes_actual_llm_helper_contract(self):
        result = subprocess.CompletedProcess([], 0,
            '{"steps":[{"mission":2,"params":{"color":"orange"}},{"mission":6}]}', "")
        with patch.object(planner.subprocess, "run", return_value=result) as call:
            self.assertEqual(planner.plan("orange cone then soccer"),
                             [(2, {"color": "orange"}), 6])
        self.assertEqual(call.call_args.kwargs["timeout"], 45)
        self.assertEqual(json.loads(call.call_args.kwargs["input"])["instruction"],
                         "orange cone then soccer")
        self.assertTrue(call.call_args.args[0][1].endswith("local_model.py"))

    def test_local_failure_never_becomes_rule_plan(self):
        result = subprocess.CompletedProcess([], 1, "", "weights missing")
        with patch.object(planner.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "weights missing"):
                planner.plan("score a goal")

    def test_local_invalid_json_rejected(self):
        result = subprocess.CompletedProcess([], 0, "I will score a goal", "")
        with patch.object(planner.subprocess, "run", return_value=result):
            with self.assertRaises(json.JSONDecodeError):
                planner.plan("score a goal")

    def test_local_timeout_propagates(self):
        with patch.object(planner.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 45)):
            with self.assertRaises(subprocess.TimeoutExpired):
                planner.plan("score a goal")

    def test_strict_gate_accepts_successful_local_llm(self):
        result = subprocess.CompletedProcess([], 0, '{"steps":[{"mission":6}]}', "")
        with patch.dict("os.environ", {"HAIC_REQUIRE_LLM": "1", "HAIC_ALLOW_RULE_PLANNER": "1"}):
            with patch.object(planner.subprocess, "run", return_value=result) as call:
                self.assertEqual(planner.plan("score a goal"), [6])
                call.assert_called_once()

    def test_configured_api_precedes_opted_in_rules(self):
        response = io.BytesIO(json.dumps({"choices": [{"message": {
            "content": '{"steps":[{"mission":3}]}'}}]}).encode())
        with patch.dict("os.environ", {"HAIC_ALLOW_RULE_PLANNER": "1", "HAIC_LLM_MODEL": "test",
                                       "HAIC_LLM_API_KEY": "dummy"}):
            with patch.object(planner.urllib.request, "urlopen", return_value=response) as call:
                self.assertEqual(planner.plan("cross cliff"), [3])
                call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
