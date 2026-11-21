#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
import math
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone
from custom.mission_5.ros_io import PandaROSIO


def positive_seconds(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or not 0 < value <= 600:
        raise ValueError(name + " must be finite and between 0 and 600 seconds")
    return value


class Mission5(Mission):
    MISSION = 5

    def __init__(self):
        super().__init__()

    def run(self) -> MissionResult:
        interface = None
        commands = 0
        try:
            self.begin_mission(5, timeout=15.)
            from custom.mission_5.policy import load_policy,default_checkpoint
            checkpoint = Path(os.environ.get("M5_CHECKPOINT", str(default_checkpoint()))).expanduser()
            if not checkpoint.is_file():
                return self.result(error="Team-trained Mission5 checkpoint missing; collect/train first",
                                   checkpoint=str(checkpoint), learned_policy=False)
            wall_timeout = positive_seconds("M5_WALL_TIMEOUT", "180")
            sim_timeout = positive_seconds("M5_MAX_SECONDS", "90")
            # CPU inference preserves shared GPU capacity and works in the judge env.
            policy = load_policy(checkpoint, device="cpu")
            fps = float(policy.fps)
            if not math.isfinite(fps) or not 1 <= fps <= 30:
                raise ValueError("Checkpoint control fps must be in [1,30]")
            policy.reset()
            trace_dir = Path(os.environ['M5_TRACE_DIR']) if os.getenv('M5_TRACE_DIR') else None
            if trace_dir is not None:trace_dir.mkdir(parents=True,exist_ok=True)
            execution_horizon = int(os.environ.get("M5_EXECUTE_HORIZON", "1"))
            if not 1 <= execution_horizon <= 64:
                raise ValueError("M5_EXECUTE_HORIZON must be in [1,64]")
            interface = PandaROSIO(self)
            queue = deque()
            wall_end = time.monotonic() + wall_timeout
            initial_sim = previous_sim = None
            while time.monotonic() < wall_end:
                observation = interface.observation(timeout=min(3., max(.01, wall_end - time.monotonic())))
                sim_now = float(observation["time"])
                if initial_sim is None:
                    initial_sim = sim_now
                if sim_now - initial_sim >= sim_timeout:
                    break
                if previous_sim is not None and sim_now < previous_sim:
                    raise RuntimeError("Simulation clock moved backwards")
                if previous_sim is not None and sim_now - previous_sim < .8 / fps:
                    time.sleep(.005)
                    continue
                if not queue:
                    predicted = policy.act(observation)
                    if getattr(policy,'failed',None):raise RuntimeError(str(policy.failed))
                    if getattr(policy,'done',False):break
                    actions = np.asarray(predicted.get("actions"), dtype=np.float64)
                    if actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= 64:
                        raise ValueError("Policy must return a nonempty actions[N,7] chunk")
                    if not np.isfinite(actions).all():
                        raise ValueError("Policy returned nonfinite actions")
                    # Default to one predicted action, then observe/replan.
                    queue.extend(actions[:execution_horizon])
                action=queue.popleft()
                if trace_dir is not None and commands%5==0:
                    from PIL import Image as PILImage
                    row={'time':sim_now,'state':observation['state'].tolist(),'action':action.tolist(),
                         'policy_stage':getattr(policy,'stage',None),'policy_debug':getattr(policy,'debug',{})}
                    with (trace_dir/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    PILImage.fromarray(observation['image']).save(trace_dir/f'{commands:05d}_agent.png')
                    PILImage.fromarray(observation['wrist_image']).save(trace_dir/f'{commands:05d}_wrist.png')
                interface.command(action)
                commands += 1
                previous_sim = sim_now
                if commands % max(1, int(2 * fps)) == 0:
                    self.log(f"[mission 5] trained policy: {commands} commands, sim={sim_now-initial_sim:.1f}s")
            return self.result(learned_policy=True, commands=commands,
                               checkpoint=checkpoint.name, controller=getattr(policy,'policy_name','two-camera-cnn-bc'),
                               policy_complete=getattr(policy,'done',None),
                               detail="Bounded policy rollout complete; original referee determines insertion success")
        except Exception as error:
            return self.result(error=f"{type(error).__name__}: {error}", commands=commands)
        finally:
            if interface is not None:
                try:
                    if not interface.hold():
                        self.log("[mission 5] Panda hold unavailable: fresh joint state missing")
                except Exception as error:
                    self.log("[mission 5] Panda hold unavailable: " + str(error))
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission5))
