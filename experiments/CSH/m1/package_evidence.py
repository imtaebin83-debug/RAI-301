"""Keep small original verdicts and summaries with the custom-only report."""
import collections
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

root = Path(__file__).resolve().parent
out = root.parents[2] / 'src/custom/docs/experiments/CSH_m1_20261127'
official = out / 'official'
official.mkdir(parents=True, exist_ok=True)
manifest = []
for path in sorted([*root.glob('**/official.json'), *root.glob('**/seed*_case*.json')]):
    target = official / ('__'.join(path.relative_to(root).parts))
    shutil.copy2(path, target)
    manifest.append({'source': str(path.relative_to(root)), 'file': str(target.relative_to(out)),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
for name in ('all_results.json', 'planner_probe.json', 'planner_latency.json',
             'trajectory_comparison.json', 'velocity_comparison.json', 'model_manifest.json',
             'SPLITS.json', 'dependencies.json'):
    shutil.copy2(root / name, out / name)
shutil.copy2(root / 'final-web/web_smoke.json', out / 'web_smoke.json')

summary = []
for path in sorted([*root.glob('**/localization.jsonl'), *root.glob('**/trace.jsonl')]):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    unique, previous = [], None
    for r in rows:
        if r['valid'] and r['measurement_stamp'] != previous:
            unique.append(r)
            previous = r['measurement_stamp']
    record = {'case': str(path.parent.relative_to(root)), 'accepted_measurements': len(unique),
              'rejected_scans': max((r['rejected_scans'] for r in rows), default=0),
              'reanchors': max((r['recovery_count'] for r in rows), default=0),
              'invalid_reasons': dict(collections.Counter(r['reason'] for r in rows if not r['valid'])),
              'pending_input_records': sum(r['metrics'].get('state') == 'waiting_imu' for r in rows)}
    for key in ('dt', 'seconds', 'support', 'rmse', 'gyro_error'):
        values = [r['metrics'][key] for r in unique if r['metrics'].get(key) is not None]
        if values:
            record[key] = dict(zip(('min', 'p50', 'p95', 'max'),
                                  map(float, np.quantile(values, [0., .5, .95, 1.]))))
    summary.append(record)
(out / 'trace_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
(out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print('packaged original verdict files:', len(manifest))
