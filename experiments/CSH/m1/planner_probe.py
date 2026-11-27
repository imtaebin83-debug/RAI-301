import json
from pathlib import Path
import time

from custom.planner.planner import SYSTEM, parse_plan
from custom.planner.worker import generate_local, _WORKER

EXTRA = '''
Understand both English and Korean. Output every requested action in order,
including the final action in a long sentence. Color values must be English:
초록/초록색/녹색 = green; 파랑/파란색/청색 = blue; 주황/주황색 = orange.
절벽/틈 건너기 = mission 3; 방 알아내기/교실에 연필 배달 = mission 4;
연필꽂이에 연필 꽂기 = mission 5; 축구/득점 = mission 6.
Example instruction: 초록색 콘으로 이동하고 절벽을 건넌 후 교실에 연필을 배달해.
Example answer: {"steps":[{"mission":2,"params":{"color":"green"}},{"mission":3,"params":{}},{"mission":4,"params":{}}]}
Example instruction: (16, 4.5)로 가고 주황색 콘을 찾아가.
Example answer: {"steps":[{"mission":1,"params":{"goal":[16,4.5]}},{"mission":2,"params":{"color":"orange"}}]}
Example instruction: 콘을 찾아가.
Example answer: {"steps":[]}
Example instruction: Make coffee.
Example answer: {"steps":[]}
'''

CASES = [
    ('dev', 'Go to the blue cone, then score a goal.', [[2, {'color': 'blue'}], 6]),
    ('dev', '좌표 (14.2, 7.7)로 이동한 다음 주황색 콘을 찾아가.', [[1, {'goal': [14.2, 7.7]}], [2, {'color': 'orange'}]]),
    ('dev', '초록색 콘으로 간 다음 절벽을 건너고 교실에 연필을 배달해.', [[2, {'color': 'green'}], 3, 4]),
    ('eval', '파란색 콘을 찾고 나서 (16.6, 2.3)으로 이동해.', [[2, {'color': 'blue'}], [1, {'goal': [16.6, 2.3]}]]),
    ('eval', '먼저 연필을 연필꽂이에 꽂고, 축구 득점을 해 줘.', [5, 6]),
    ('eval', 'Cross the gap course, deliver the pencil to the classroom, then find the orange cone.', [3, 4, [2, {'color': 'orange'}]]),
    ('eval', 'Walk to (14.05, 19.7), then to (16.8, 1.1).', [[1, {'goal': [14.05, 19.7]}], [1, {'goal': [16.8, 1.1]}]]),
    ('eval', '콘을 찾아가.', None),
    ('eval', 'Make coffee.', None),
]

output = Path(__file__).with_name('planner_probe.json')
results = []
try:
    for index, (split, instruction, expected) in enumerate(CASES):
        started = time.monotonic()
        record = {'split': split, 'instruction': instruction, 'expected': expected}
        try:
            answer = generate_local({'system': SYSTEM, 'instruction': instruction})
            record['raw'] = answer
            try:
                record['plan'] = json.loads(json.dumps(parse_plan(answer)))
                record['correct'] = record['plan'] == expected
            except ValueError as error:
                record['error'] = str(error)
                record['correct'] = expected is None
        except Exception as error:
            record['error'] = str(error)
            record['correct'] = False
        record['seconds'] = time.monotonic() - started
        results.append(record)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(index, record['correct'], round(record['seconds'], 2), flush=True)
finally:
    _WORKER.close()
