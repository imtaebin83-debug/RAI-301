"""Small additional actual-model check in the canonical v2 CPU environment."""
import json
from pathlib import Path
import time

from custom.planner.planner import SYSTEM, parse_plan
from custom.planner.worker import generate_local, _WORKER

cases = [
    ('먼저 (14.7, 6.2)로 간 다음 파란색 콘으로 이동하고, 마지막으로 축구 득점해.',
     [[1, {'goal': [14.7, 6.2]}], [2, {'color': 'blue'}], 6]),
    ('Go to the orange cone, then navigate to (16.2, 13.4).',
     [[2, {'color': 'orange'}], [1, {'goal': [16.2, 13.4]}]]),
    ('Find a cone, then cross the gap.', None),
]
records = []
for instruction, expected in cases:
    began = time.monotonic()
    raw = generate_local({'system': SYSTEM, 'instruction': instruction}, timeout=45.)
    record = {'instruction': instruction, 'expected': expected, 'raw_completion': raw,
              'seconds': time.monotonic() - began, 'worker_pid': _WORKER.process.pid,
              'kind': 'actual local Qwen CPU, no answer cache'}
    try:
        actual = json.loads(json.dumps(parse_plan(raw)))
        record.update(actual=actual, passed=actual == expected)
    except ValueError as error:
        record.update(error=str(error), passed=expected is None)
    records.append(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
Path('../experiments/CSH/m1/v2_planner_probe.json').write_text(
    json.dumps(records, ensure_ascii=False, indent=2) + '\n')
_WORKER.close()
if not all(r['passed'] for r in records):
    raise SystemExit(1)
