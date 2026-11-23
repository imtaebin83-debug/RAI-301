"""Evaluation only: layouts reserved before the final candidate results."""
import json
from pathlib import Path
from custom.mission_2.evaluate import run_case

root = Path('/home/aril/rai-worktrees/ys-m2-fixes-20261123/src')
output = Path('/tmp/ys-m2-candidate-fresh-layouts-1123')
output.mkdir(exist_ok=False)
cases = [
    dict(id='fresh-blue', split='held-out-layout', missions=[2],
         cone_order=['blue', 'orange', 'green'], m2_color='blue'),
    dict(id='fresh-green', split='held-out-layout', missions=[2],
         cone_order=['green', 'blue', 'orange'], m2_color='green'),
    dict(id='fresh-orange', split='held-out-layout', missions=[2],
         cone_order=['orange', 'green', 'blue'], m2_color='orange'),
]
(output / 'manifest.json').write_text(json.dumps(cases, indent=2))
records = []
for case in cases:
    records.append(run_case(case, root, output, 900.))
    (output / 'summary.json').write_text(json.dumps(records, indent=2))
