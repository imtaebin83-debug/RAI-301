"""Evaluation only: generate unseen scenarios, then let the stock referee judge.

Mission controllers never import this module or receive scenario contents.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import yaml


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=301)
    ap.add_argument('--cases',type=int,default=2)
    ap.add_argument('--missions',nargs='+',type=int,default=[1,2])
    ap.add_argument('--output',type=Path,help='write evidence here instead of tracked baseline fixtures')
    ap.add_argument('--source',type=Path,help='isolated source tree, for baseline comparison')
    args=ap.parse_args()
    if 5 in args.missions:
        ap.error('Mission 5 is owned by another active task')
    root=args.source.resolve() if args.source else Path(__file__).resolve().parents[2]
    rng=random.Random(args.seed)
    output=args.output.resolve() if args.output else root/'custom'/'docs'/'validation'
    output.mkdir(parents=True,exist_ok=True)
    summary=[]
    for case in range(args.cases):
        scenario=yaml.safe_load((root/'scenarios'/'default.yaml').read_text())
        scenario['m1_goal']=[round(rng.uniform(14.5,16.5),2),round(rng.uniform(3,19),2),90.]
        scenario['m2_color']=rng.choice(['blue','green','orange'])
        rng.shuffle(scenario['cone_order'])
        rng.shuffle(scenario['room_order'])
        scenario['ball_xy']=[round(rng.uniform(7.4,11),2),round(rng.uniform(5.7,8.3),2)]
        config=yaml.safe_load((root/'main_config.yaml').read_text())
        config['simulator']['view']=False
        with tempfile.TemporaryDirectory(prefix='rai-eval-') as directory:
            directory=Path(directory)
            scenefile=directory/'scenario.yaml'
            configfile=directory/'config.yaml'
            scenefile.write_text(yaml.safe_dump(scenario))
            configfile.write_text(yaml.safe_dump(config))
            before=set((root/'results').glob('*.json'))
            command=[sys.executable,'-u',str(root/'main.py'),'--missions',*map(str,args.missions),
                     '--no-wait','--config',str(configfile),'--scenario',str(scenefile)]
            env={**os.environ,'MUJOCO_GL':'egl','PYTHONDONTWRITEBYTECODE':'1',
                 'HAIC_ROOT':str(root.parent),
                 'PYTHONPATH':str(root)+os.pathsep+os.environ.get('PYTHONPATH','')}
            name=f'seed{args.seed}_case{case}'
            with (output/(name+'.log')).open('w') as log:
                process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    rc=process.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGINT)
                    try: rc=process.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        rc=process.wait()
            after=set((root/'results').glob('*.json'))-before
            hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root/'custom').rglob('*.py') if '__pycache__' not in str(p)}
            record={'case':case,'seed':args.seed,'returncode':rc,'evaluation_scenario':scenario,
                    'code_sha256':hashes}
            if after:
                result=json.loads(max(after,key=lambda p:p.stat().st_mtime).read_text())
                record['official_result']=result
            (output/(name+'.json')).write_text(json.dumps(record,indent=2))
            summary.append({'case':case,'returncode':rc,'total':record.get('official_result',{}).get('total')})
            print(json.dumps(summary[-1]),flush=True)
    (output/f'suite_seed{args.seed}.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
