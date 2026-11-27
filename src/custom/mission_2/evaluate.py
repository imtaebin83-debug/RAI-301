"""Evaluation only. Default lists cases; --run explicitly starts simulations.

Controllers never import this module. Scenarios and official answers stay here.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

MISSION = int(Path(__file__).parent.name.split('_')[-1])


def cases():
    if MISSION == 4:
        result = [{'id': f'rooms-{index}', 'split': 'development' if index < 3 else 'held-out',
                   'missions': [4], 'room_order': list(order)}
                  for index, order in enumerate(itertools.permutations(
                      ('classroom', 'medical', 'cafeteria')))]
        result.append({'id': 'after-cone', 'split': 'integration', 'missions': [2, 4],
                       'room_order': ['medical', 'cafeteria', 'classroom']})
        return result
    result = []
    # Preserve existing case IDs, then cover the three reverse permutations.
    # Each target now appears twice in every slot, with different neighbours.
    for index, order in enumerate((['blue', 'green', 'orange'],
                                   ['orange', 'blue', 'green'],
                                   ['green', 'orange', 'blue'],
                                   ['blue', 'orange', 'green'],
                                   ['green', 'blue', 'orange'],
                                   ['orange', 'green', 'blue'])):
        for color in ('blue', 'green', 'orange'):
            result.append({'id': f'order-{index}-{color}',
                           'split': 'development' if index < 3 else 'additional-coverage',
                           'missions': [2], 'cone_order': order, 'm2_color': color})
    for color in ('blue', 'green', 'orange'):
        rng = random.Random(9227 + ('blue', 'green', 'orange').index(color))
        order = ['blue', 'green', 'orange']
        rng.shuffle(order)
        result.append({'id': f'after-m1-{color}', 'split': 'held-out', 'missions': [1, 2],
                       'cone_order': order, 'm2_color': color,
                       'm1_goal': [round(rng.uniform(14.3, 16.7), 2),
                                   round(rng.uniform(4., 19.), 2), 90.]})
    return result


def check_gpu():
    status = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                             '--format=csv,noheader'], text=True, capture_output=True,
                            check=True, timeout=10)
    if status.stdout.strip():
        raise RuntimeError('GPU compute processes are active; waiting for an idle GPU is required')


def run_case(case, root, output, timeout):
    import yaml

    check_gpu()
    directory = output / case['id']
    directory.mkdir(parents=True, exist_ok=False)
    scenario = yaml.safe_load((root / 'scenarios/default.yaml').read_text())
    settings = yaml.safe_load((root / 'main_config.yaml').read_text())
    if list(settings['simulator']['start_pose']) != [16., 11., 0.]:
        raise ValueError('This case manifest requires the published default launch pose')
    for key in ('room_order', 'cone_order', 'm2_color', 'm1_goal'):
        if key in case:
            scenario[key] = case[key]
    (directory / 'evaluation_scenario.yaml').write_text(yaml.safe_dump(scenario))
    before = set((root / 'results').glob('*.json'))
    env = {**os.environ, 'HAIC_ROOT': str(root.parent),
           'PYTHONPATH': str(root) + os.pathsep + os.getenv('PYTHONPATH', ''),
           'PYTHONDONTWRITEBYTECODE': '1', 'ROS_LOCALHOST_ONLY': '1',
           'ROS_DOMAIN_ID': os.getenv('YS_EVAL_ROS_DOMAIN_ID', '74')}
    env.pop('HAIC_INITIAL_POSE', None)
    if MISSION == 4:
        env['HAIC_M4_CAPTURE_DIR'] = str(directory / 'camera')
    # Wrapper acquires the team's GPU simulator lock exactly once and creates
    # a temporary display-only config. No checked-in TA file is edited.
    command = ['bash', 'custom/tools/run_sim.sh', '--headless', '--missions',
               *map(str, case['missions']), '--no-wait', '--scenario',
               str(directory / 'evaluation_scenario.yaml')]
    record = {'case': case, 'command': command, 'source_root': str(root),
              'starter_version': 'v2', 'python': sys.executable,
              'config_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ('custom/robot.yaml', 'main_config.yaml')},
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                 text=True).strip(),
              'code_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (root / 'custom').rglob('*.py')},
              'initial_pose': [16., 11., 0.], 'ros_domain_id': env['ROS_DOMAIN_ID'],
              'runtime_options': {key: env.get(key) for key in
                  ('HAIC_MAP_CORRECTION', 'HAIC_VLM_THREADS', 'HAIC_M2_CAPTURE_DIR')}}
    started = time.monotonic()
    with (directory / 'simulator.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            record['returncode'] = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            record['interrupted'] = type(exc).__name__
            # Only this evaluator's newly created process group is signalled.
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                record['returncode'] = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                record['returncode'] = proc.wait()
    record['wall_seconds'] = time.monotonic() - started
    created = set((root / 'results').glob('*.json')) - before
    if len(created) == 1:
        record['official_result'] = json.loads(created.pop().read_text())
    else:
        record['result_error'] = f'Expected one new official result, found {len(created)}'
    (directory / 'record.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'case': case['id'], 'returncode': record['returncode'],
                      'total': record.get('official_result', {}).get('total')}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='start the selected simulations')
    parser.add_argument('--case', action='append', help='case ID; repeat to select several')
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[2], help='baseline/candidate src')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=900.)
    args = parser.parse_args()
    selected = [case for case in cases() if not args.case or case['id'] in args.case]
    if args.case and set(args.case) - {case['id'] for case in selected}:
        parser.error('unknown case ID')
    if not args.run:
        print(json.dumps(selected, indent=2))
        return
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.timeout > 1800:
        parser.error('timeout must be positive and at most 1800 seconds')
    root = args.source_root.resolve()
    output = (args.output or root / 'custom/docs/experiments' /
              f'ys_m{MISSION}_{time.strftime("%Y%m%d-%H%M%S")}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in selected:
        record = run_case(case, root, output, args.timeout)
        records.append(record)
        (output / 'summary.json').write_text(json.dumps(records, indent=2))
        if record['returncode'] == 75 or 'interrupted' in record:
            raise SystemExit(record['returncode'] or 1)
    if any(not record.get('official_result', {}).get('total', {}).get('complete')
           or record['official_result']['total'].get('failed', 1) for record in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
