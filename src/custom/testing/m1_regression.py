"""Evaluation only: stock referee, isolated spawns and long navigation probes.

Run under one outer flock on /tmp/rai-gpu-simulator.lock. --source can select
an unmodified baseline worktree. Probe checkpoints read the official judge
after stopping; those measurements are never sent back to the Navigator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

import yaml


CASES = {
    'default': {'missions': [1]},
    'low': {'missions': [1], 'goal': [14.1, 1.2]},
    'high': {'missions': [1], 'goal': [16.9, 19.8]},
    'room-start': {'missions': [1], 'spawn': [20., 10.5, 180.], 'goal': [15., 5.]},
    'cone-start': {'missions': [1], 'spawn': [15.5, 24., -90.], 'goal': [16., 11.]},
    'after-m1-green': {'missions': [1, 2], 'goal': [14.98, 7.99],
                       'm2_color': 'green', 'cone_order': ['orange', 'blue', 'green']},
    'roundtrip': {'probe': [[16., 18.5], [16., 3.5], [16., 18.5], [16., 11.]]},
    'doors': {'probe': [[19.2, 17.5], [16., 3.59], [19.2, 3.59], [16., 11.]]},
    'survey': {'probe': [[16.9, 17.5], [16.9, 10.5], [16.9, 3.59],
                         [16.9, 17.5], [18.65, 17.5], [16., 11.]]},
    'mission3': {'missions': [3]},
    'mission4': {'missions': [4]},
    'rooms-2': {'missions': [4], 'room_order': ['medical', 'classroom', 'cafeteria']},
    'rooms-3': {'missions': [4], 'room_order': ['medical', 'cafeteria', 'classroom']},
    'after-cone': {'missions': [2, 4], 'room_order': ['medical', 'cafeteria', 'classroom']},
    'after-delivery': {'missions': [4, 1], 'goal': [15., 5.],
                       'room_order': ['medical', 'cafeteria', 'classroom']},
    'mission6': {'missions': [6]},
}


def worker(source, directory):
    # Select source before importing main, mission_api, or any controller.
    sys.path.insert(0, str(source))
    import main as stock
    from mission_api import MissionRunner
    from referee import Referee
    from custom.mission_1.mission_1 import Mission1
    from custom.navigation import Navigator

    case = json.loads((directory / 'case.json').read_text())
    args = stock.parse_args(['--config', str(directory / 'config.yaml'),
                             '--scenario', str(directory / 'scenario.yaml'), '--no-wait'])
    simulator = rsp = bringup = None
    complete = False
    referee = Referee(directory / 'official.json', run={'case': case, 'source': str(source),
                       'kind': 'navigation probe' if 'probe' in case else 'stock mission'})
    try:
        simulator = stock.start_simulator(args)
        rsp = stock.start_robot_state_publisher()
        with MissionRunner('csh_m1_evaluation') as runner:
            runner.api.wait_for_sim(timeout=90)
            bringup = stock.start_bringup()
            if 'probe' in case:
                mission = runner.start(Mission1)
                referee.begin(1)
                checkpoints = []
                try:
                    import time
                    nav = Navigator(mission, mission.begin_mission(1))
                    for target in case['probe']:
                        began = time.monotonic()
                        reached = nav.route_to(*target, timeout=100.)
                        estimated = nav.pose()
                        verdict = runner.api.judge(1)  # Evaluator only, never controller input.
                        actual = verdict.get('pelvis', [])
                        error = (math.dist(actual[:2], target) if len(actual) >= 2 else None)
                        checkpoints.append({'target': target, 'reached': reached,
                            'seconds': time.monotonic() - began, 'estimated': estimated,
                            'official_checkpoint': verdict, 'actual_goal_error': error,
                            'estimate_error': math.dist(actual[:2], estimated[:2]) if len(actual) >= 2 else None})
                        (directory / 'checkpoints.json').write_text(json.dumps(checkpoints, indent=2))
                        if not reached:
                            break
                        if not nav.face(0., timeout=15.):
                            raise RuntimeError('probe heading was not reached')
                    result = mission.result(checkpoints=checkpoints)
                except Exception as error:
                    result = mission.result(error=str(error), checkpoints=checkpoints)
                finally:
                    mission.stop()
                referee.finish(runner.api, result)
                runner.finish(mission)
            else:
                for number in case['missions']:
                    stock.run_step(runner, referee, number)
            complete = True
    finally:
        stock.stop_all([(simulator, 'simulator'), (rsp, 'robot_state_publisher'),
                        (bringup, 'custom bringup')])
        referee.close(complete=complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=CASES)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cases', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.worker:
        worker(source, output)
        return
    selected = [(name, dict(CASES[name])) for name in args.case or []]
    if args.seed is not None:
        rng = random.Random(args.seed)
        for index in range(args.cases):
            selected.append((f'seed{args.seed}-{index}', {'missions': [1],
                'goal': [round(rng.uniform(14., 17.), 2), round(rng.uniform(1., 20.), 2)]}))
    if not selected:
        parser.error('select at least one case or seed')
    summary = []
    for name, case in selected:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=False)
        config = yaml.safe_load((source / 'main_config.yaml').read_text())
        config['simulator']['view'] = False
        config['instruction']['open_browser'] = False
        spawn = case.get('spawn', [16., 11., 0.])
        config['simulator']['start_pose'] = spawn
        scenario = yaml.safe_load((source / 'scenarios/default.yaml').read_text())
        if 'probe' in case:
            case['goal'] = case['probe'][-1]
        if 'goal' in case:
            scenario['m1_goal'] = [*case['goal'], 90.]
        for key in ('m2_color', 'cone_order', 'room_order'):
            if key in case:
                scenario[key] = case[key]
        for filename, value in (('config.yaml', config), ('scenario.yaml', scenario)):
            (directory / filename).write_text(yaml.safe_dump(value))
        (directory / 'case.json').write_text(json.dumps(case, indent=2))
        hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (source / 'custom').rglob('*.py') if '__pycache__' not in str(p)}
        (directory / 'code_sha256.json').write_text(json.dumps(hashes, indent=2))
        env = {**os.environ, 'HAIC_ROOT': str(source.parent),
               'PYTHONPATH': str(source) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               'HAIC_INITIAL_POSE': ','.join(map(str, spawn)),
               'HAIC_LOCALIZATION_TRACE': str(directory / 'localization.jsonl'),
               'MUJOCO_GL': 'egl', 'PYTHONDONTWRITEBYTECODE': '1'}
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                   '--source', str(source), '--output', str(directory)]
        with (directory / 'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGINT)
                try:
                    code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
        record = {'case': name, 'returncode': code, 'kind': 'probe' if 'probe' in case else 'mission'}
        if (directory / 'official.json').exists():
            official = json.loads((directory / 'official.json').read_text())
            record['total'] = official.get('total')
            if 'probe' in case:
                data = official['missions'][0].get('data', {}) if official.get('missions') else {}
                points = data.get('checkpoints', [])
                record['probe_succeeded'] = (len(points) == len(case['probe']) and not data.get('error')
                    and all(p['reached'] and p['actual_goal_error'] is not None
                            and p['actual_goal_error'] <= .5 for p in points))
                record['checkpoint_errors'] = [p['actual_goal_error'] for p in points]
        summary.append(record)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
