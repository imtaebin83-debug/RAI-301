"""Persistent helper lifecycle and deadline checks without loading a model."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from custom.planner.worker import LocalWorker
from custom.planner import planner


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def worker(self, code):
        helper = Path(self.directory.name) / 'helper.py'
        helper.write_text(code)
        worker = LocalWorker(helper)
        self.addCleanup(worker.close)
        return worker

    def test_one_process_serves_two_independent_requests(self):
        worker = self.worker('import sys,json,os\nfor line in sys.stdin:\n'
            ' data=json.loads(line)\n print(json.dumps({"answer":data["instruction"]}),flush=True)\n')
        self.assertEqual(worker.generate({'instruction': 'first'}), 'first')
        first = worker.process
        self.assertEqual(worker.generate({'instruction': 'second'}), 'second')
        self.assertIs(worker.process, first)
        worker.close()
        self.assertIsNotNone(first.poll())

    def test_inference_timeout_terminates_owned_child(self):
        worker = self.worker('import time\ntime.sleep(30)\n')
        with self.assertRaises(subprocess.TimeoutExpired):
            worker.generate({'instruction': 'anything'}, timeout=.1)
        self.assertIsNone(worker.process)

    def test_partial_reply_has_same_deadline(self):
        worker = self.worker('import sys,time\nsys.stdout.write("partial");sys.stdout.flush()\ntime.sleep(30)\n')
        with self.assertRaises(subprocess.TimeoutExpired):
            worker.generate({'instruction': 'anything'}, timeout=.1)
        self.assertIsNone(worker.process)

    def test_error_response_cannot_become_a_plan(self):
        worker = self.worker('import sys,json\nfor line in sys.stdin:\n'
            ' print(json.dumps({"error":"weights missing"}),flush=True)\n')
        with self.assertRaisesRegex(RuntimeError, 'weights missing'):
            worker.generate({'instruction': 'anything'})
        self.assertIsNone(worker.process)

    def test_unexpected_eof_reports_failure_and_closes(self):
        worker = self.worker('import sys\nprint("failed to load",file=sys.stderr)\nsys.exit(3)\n')
        with self.assertRaisesRegex(RuntimeError, 'failed to load'):
            worker.generate({'instruction': 'anything'})
        self.assertIsNone(worker.process)

    def test_malformed_reply_is_rejected(self):
        worker = self.worker('import sys\nfor line in sys.stdin:\n print("not JSON",flush=True)\n')
        with self.assertRaises(json.JSONDecodeError):
            worker.generate({'instruction': 'anything'})
        self.assertIsNone(worker.process)


class PersistentPlanTests(unittest.TestCase):
    def test_default_uses_persistent_actual_llm_with_deadline(self):
        with patch.dict('os.environ', {'HAIC_REQUIRE_LLM': '1'}, clear=True), \
                patch.object(planner, 'generate_local', return_value='{"steps":[{"mission":6}]}') as call:
            self.assertEqual(planner.plan('Score a goal.'), [6])
        self.assertEqual(call.call_args.kwargs['timeout'], 45)
        self.assertEqual(call.call_args.args[0]['instruction'], 'Score a goal.')

    def test_invalid_persistent_output_never_uses_rules(self):
        with patch.dict('os.environ', {'HAIC_REQUIRE_LLM': '1', 'HAIC_ALLOW_RULE_PLANNER': '1'}, clear=True), \
                patch.object(planner, 'generate_local', return_value='{"steps":[]}'):
            with self.assertRaisesRegex(ValueError, 'no executable'):
                planner.plan('Score a goal.')


if __name__ == '__main__':
    unittest.main()
