#!/usr/bin/env python3

from __future__ import annotations

import sys
import math
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from custom.sensor_mission import Mission
from mission_api import MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator


class Mission1(Mission):
    MISSION = 1

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        try:
            start = self.begin_mission(1)
            instructed = "goal" in self.params
            goal = self.params["goal"] if instructed else self.m1_goal
            if goal is None:
                return self.result(reason='No goal supplied by instruction')
            if (not isinstance(goal, (list, tuple)) or len(goal) not in ((2,) if instructed else (2, 3))
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in goal)):
                raise ValueError('Mission 1 requires finite goal coordinates [x,y]')
            x, y = map(float, goal[:2])
            if not (14. <= x <= 17. and 1. <= y <= 20.):
                raise ValueError('Mission 1 goal outside published corridor x=14..17, y=1..20')
            nav = Navigator(self, start)
            reached = nav.route_to(x, y, tolerance=.18, timeout=180)
            return self.result(controller_reached=reached, estimated_pose=nav.pose())
        except Exception as exc:
            return self.result(error=str(exc))
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission1))
