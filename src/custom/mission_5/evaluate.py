"""Evaluate learned Mission5 once on your own already-running simulator.

Starter v2 removed the public Panda reset. This harness evaluates the current
placement once and reads the referee; it never silently repeats that placement.
Use benchmark.py for seeded trials, each in a fresh simulator process.
"""
import argparse,json
from pathlib import Path


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=1,
                   help='Exactly 1 current-placement trial; use benchmark.py for multiple fresh simulators')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--keep-first',action='store_true',
                   help='Compatibility no-op: the single trial always keeps the existing placement; no reset is available')
    args=p.parse_args(argv)
    if args.episodes!=1:
        p.error('--episodes must be 1: starter v2 has no public Panda reset. '
                'Use python -m custom.mission_5.benchmark for multiple seeded trials '
                'in fresh simulator processes.')
    return args


def main(argv=None):
    args=parse_args(argv)
    from mission_api import MissionRunner
    from referee import Referee
    from custom.mission_5.mission_5 import Mission5

    args.out.parent.mkdir(parents=True,exist_ok=True)
    with MissionRunner('m5_policy_evaluation') as runner:
        runner.api.wait_for_sim(timeout=40)
        referee=Referee(args.out,run={'kind':'learned_mission5_single_placement_evaluation',
            'episodes':1,'placement':'existing','reset_available':False,
            'keep_first_compatibility_flag':args.keep_first})
        node=runner.start(Mission5)
        try:
            referee.begin(5)
            result=node.run()
            referee.finish(runner.api,result)
        finally:
            runner.finish(node)
        total=referee.close(complete=True)
        print(json.dumps(total,indent=2))


if __name__=='__main__':main()
