"""Collect original verdicts without turning probe PASS into task success."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
repo = root.parents[2]
baseline = repo.parent / (repo.name + '-baseline') / 'src/custom/docs/validation'
for name in ('seed301_case0.json', 'seed301_case0.log', 'seed301_case1.json',
             'seed301_case1.log', 'suite_seed301.json'):
    shutil.copy2(baseline / name, root / 'baseline' / name)
default = repo / 'src/results/run_20261127-092035.json'
if default.exists():
    shutil.copy2(default, root / 'candidate-v1/official.json')
progress = root / 'final-web/web_smoke.json'
if progress.exists():
    value = json.loads(progress.read_text())
    if value.get('results_file') and Path(value['results_file']).exists():
        shutil.copy2(value['results_file'], progress.parent / 'official.json')

current = hashlib.sha256((repo / 'src/custom/navigation.py').read_bytes()).hexdigest()
rows = []
for path in sorted([*root.glob('**/official.json'), *root.glob('**/seed*_case*.json')]):
    data = json.loads(path.read_text())
    if 'official_result' in data:
        hashes = data.get('code_sha256', {})
        official = data['official_result']
    else:
        official = data
        stamp = path.parent / 'code_sha256.json'
        hashes = json.loads(stamp.read_text()) if stamp.exists() else {}
    for record in official.get('missions', []):
        report = record.get('data', {})
        referee = record.get('referee', {})
        rows.append({'evidence': str(path.relative_to(root)), 'mission': record['mission'],
            'kind': official.get('run', {}).get('kind', 'stock mission'),
            'verdict': record['verdict'], 'seconds': record['seconds'],
            'error': report.get('error'), 'failed_stage': report.get('failed_stage'),
            'actual_pelvis': referee.get('pelvis'), 'actual_goal_error_m': referee.get('distance_m'),
            'rooms_correct': record.get('rooms', {}).get('correct'),
            'summary': referee.get('summary'),
            'normalized_navigation': hashes.get('custom/navigation.py') == current if hashes else None,
            'navigation_sha256': hashes.get('custom/navigation.py')})
(root / 'all_results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
print('official verdict records:', len(rows))
for row in rows:
    if row['normalized_navigation']:
        print(row['evidence'], row['mission'], row['verdict'], row['seconds'])
