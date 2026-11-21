#!/usr/bin/env python3

from __future__ import annotations

import sys
import math
import time
from pathlib import Path
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator
from custom.vision import RGBD


class Mission6(Mission):
    MISSION = 6

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        start = self.begin_mission(6)
        deadline = time.monotonic()+235.
        observations = 0
        goal_stamp = None
        goal_frames = 0
        try:
            nav = Navigator(self, start)
            if not nav.route_to(12.,9.1,timeout=120.):
                return self.result(error='Could not enter soccer field')
            sensor = RGBD(self,nav)
            ball = None
            for yaw in (-math.pi*.8,-math.pi*.6,-math.pi,-math.pi/2):
                nav.face(yaw,timeout=12.)
                ball = sensor.observe('ball',seconds=1.5)
                if ball is not None and ball.valid():
                    break
            if ball is None or not ball.valid():
                return self.result(error='No stable elevated RGB-D soccer ball detection')
            self.log(f'[mission 6] observed ball ({ball.x:.2f},{ball.y:.2f}), range={ball.distance:.2f}')
            goal = np.array([9.25,1.4])  # published goal centre
            for attempt in range(5):
                if time.monotonic()>deadline:
                    break
                ball_xy = np.array([ball.x,ball.y])
                if not np.isfinite(ball_xy).all():
                    return self.result(error='Non-finite ball observation rejected')
                direction = goal-ball_xy
                direction /= max(np.linalg.norm(direction),1e-6)
                behind = ball_xy-direction*.70
                self.log(f'[mission 6] push attempt {attempt+1}, observed ball=({ball.x:.2f},{ball.y:.2f}), behind={behind.tolist()}')
                x,y,_ = nav.pose()
                # Walk around an observed ball before approaching from behind.
                if np.dot(np.array([x,y])-ball_xy,direction)>-.25:
                    side = np.array([-direction[1],direction[0]])
                    sign = 1 if np.dot(np.array([x,y])-ball_xy,side)>=0 else -1
                    detour = ball_xy+side*sign*.8-direction*.65
                    nav.goto(*detour,tolerance=.18,timeout=25.)
                if not nav.goto(*behind,tolerance=.15,timeout=35.):
                    break
                yaw = math.atan2(direction[1],direction[0])
                nav.face(yaw,timeout=10.)
                fresh = sensor.observe('ball',seconds=.8)
                if fresh is not None and np.hypot(fresh.x-ball.x,fresh.y-ball.y)<1.:
                    ball = fresh
                pushing_until = min(deadline,time.monotonic()+10.)
                while time.monotonic()<pushing_until:
                    detection = sensor.ball()
                    # Reject a switch from the tracked nearby ball to a white
                    # goal marking or another small elevated-looking patch.
                    if detection is not None and np.hypot(detection.x-ball.x,detection.y-ball.y)>1.:
                        detection = None
                    if detection is not None:
                        ball = detection
                        observations += 1
                        if detection.y<1.85 and 7.0<detection.x<11.5:
                            if sensor._stamp != goal_stamp:
                                goal_frames += 1
                                goal_stamp = sensor._stamp
                            if goal_frames>=3:
                                return self.result(ball_xy=[detection.x,detection.y],observations=observations)
                        else:
                            goal_frames = 0
                        x,y,heading = nav.pose()
                        bearing = math.atan2(detection.y-y,detection.x-x)
                        error = (bearing-heading+math.pi)%(2*math.pi)-math.pi
                    else:
                        heading = nav.pose()[2]
                        error = (yaw-heading+math.pi)%(2*math.pi)-math.pi
                    self.cmd_vel(.32 if abs(error)<.22 else .12,0.,float(np.clip(error,-.35,.35)))
                    time.sleep(.07)
                nav.stop()
                fresh = sensor.observe('ball',seconds=1.)
                if fresh is None or np.hypot(fresh.x-ball.x,fresh.y-ball.y)>1.:
                    break
                ball = fresh
            return self.result(observations=observations,error='Push budget exhausted or ball lost; referee decides success')
        except Exception as exc:
            return self.result(error=str(exc),observations=observations)
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission6))
