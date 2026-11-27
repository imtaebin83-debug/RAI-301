"""Package Starter v2 evidence separately from the preserved v1 results."""
import collections
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
repo = root.parents[2]
out = repo / 'src/custom/docs/experiments/CSH_m1_20261127/v2'
out.mkdir(parents=True, exist_ok=True)
progress = root / 'v2-web/web_smoke.json'
if progress.exists():
    result = json.loads(progress.read_text()).get('results_file')
    if result and Path(result).is_file():
        shutil.copy2(result, progress.parent / 'official.json')

rows, manifest, traces, probes = [], [], [], []
for directory in sorted(root.glob('v2-*')):
    if not directory.is_dir():
        continue
    paths = sorted([*directory.glob('**/official.json'), *directory.glob('**/seed*_case*.json')])
    for path in paths:
        data = json.loads(path.read_text())
        official = data.get('official_result', data)
        stamp = path.parent / 'code_sha256.json'
        hashes = data.get('code_sha256', json.loads(stamp.read_text()) if stamp.exists() else {})
        for record in official.get('missions', []):
            report, referee = record.get('data', {}), record.get('referee', {})
            rows.append({'evidence': str(path.relative_to(root)), 'mission': record['mission'],
                'kind': official.get('run', {}).get('kind', 'stock mission'),
                'baseline': directory.name.startswith('v2-baseline'),
                'teammate_overlay': directory.name == 'v2-integration',
                'verdict': record['verdict'], 'seconds': record['seconds'],
                'error': report.get('error'), 'actual_pelvis': referee.get('pelvis'),
                'actual_goal_error_m': referee.get('distance_m'),
                'summary': referee.get('summary'), 'source_sha256': hashes,
                'total': official.get('total')})
        target = out / 'official' / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append({'source': str(path.relative_to(root)), 'file': str(target.relative_to(out)),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in sorted(directory.glob('**/checkpoints.json')):
        points = json.loads(path.read_text())
        if not points:
            continue
        final = json.loads((path.parent / 'official.json').read_text())
        data = final['missions'][0].get('data', {})
        probes.append({'case': str(path.parent.relative_to(root)),
            'kind': 'navigation probe; final referee verdict alone is insufficient',
            'checkpoints': points, 'probe_error': data.get('error'),
            'probe_succeeded': len(points) == 4 and all(p['reached'] for p in points) and not data.get('error'),
            'max_actual_goal_error_m': max(p['actual_goal_error'] for p in points),
            'max_checkpoint_estimate_error_m': max(p['estimate_error'] for p in points)})
    for path in sorted(directory.glob('**/localization.jsonl')):
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        valid, previous = [], None
        for entry in entries:
            if entry['valid'] and entry['measurement_stamp'] != previous:
                valid.append(entry)
                previous = entry['measurement_stamp']
        record = {'case': str(path.parent.relative_to(root)), 'accepted_measurements': len(valid),
            'rejected_scans': max((e['rejected_scans'] for e in entries), default=0),
            'reanchors': max((e['recovery_count'] for e in entries), default=0),
            'invalid_reasons': dict(collections.Counter(e['reason'] for e in entries if not e['valid'])),
            'pending_input_records': sum(e['metrics'].get('state') == 'waiting_imu' for e in entries)}
        if path.parent.name == 'roundtrip' and valid:
            record['max_estimated_cross_track_m'] = max(abs(e['transform'][0][3]-16.) for e in valid)
        traces.append(record)

for name, data in [('v2_results.json', rows), ('v2_probes.json', probes),
                   ('v2_trace_summary.json', traces), ('manifest.json', manifest)]:
    (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    if name != 'manifest.json':
        shutil.copy2(out / name, root / name)
for name in ('SPLITS_v2.json', 'v2_imports.json', 'v2_dependencies.json', 'v2_common_tests.log',
             'v2_pip_check.txt', 'v2_integration_manifest.json', 'v2_planner_probe.json',
             'v2_scope_audit.json'):
    path = root / name
    if path.exists():
        shutil.copy2(path, out / name)
for row in rows:
    print(row['evidence'], row['mission'], row['verdict'], row['seconds'], row['error'])
for probe in probes:
    print('probe:', probe['case'], probe['probe_succeeded'], probe['probe_error'])
