"""External evaluation of learned Mission5 on your own reserved simulator.

Only this harness requests a fresh public Panda placement and reads the referee.
The submitted Mission5 controller does neither.
"""
import argparse,json,time
from pathlib import Path
from mission_api import MissionRunner
from referee import Referee
from custom.mission_5.mission_5 import Mission5


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--keep-first',action='store_true',help='First trial keeps the existing initial placement')
    args=p.parse_args()
    if not 1<=args.episodes<=50:p.error('episodes must be between1 and50')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with MissionRunner('m5_policy_evaluation') as runner:
        runner.api.wait_for_sim(timeout=40)
        from std_msgs.msg import Empty
        reset=runner.api.create_publisher(Empty,'/panda/retry',10)
        referee=Referee(args.out,run={'kind':'learned_mission5_random_placement_evaluation'})
        for index in range(args.episodes):
            if index or not args.keep_first:
                until=time.monotonic()+5
                while reset.get_subscription_count()==0 and time.monotonic()<until:time.sleep(.05)
                if reset.get_subscription_count()==0:
                    raise RuntimeError('Panda reset subscriber unavailable; refusing to label an unreset trial')
                reset.publish(Empty());time.sleep(2.)
            node=runner.start(Mission5)
            referee.begin(5)
            result=node.run()
            referee.finish(node,result)
            runner.finish(node)
        total=referee.close(complete=True)
        print(json.dumps(total,indent=2))


if __name__=='__main__':main()
