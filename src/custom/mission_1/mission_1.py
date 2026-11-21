#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator


class Mission1(Mission):
    MISSION = 1

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        start = self.begin_mission(1)
        goal = self.params.get("goal") or self.m1_goal
        try:
            if goal is None:
                return self.result(reason='No goal supplied by instruction')
            nav = Navigator(self, start)
            reached = nav.route_to(float(goal[0]), float(goal[1]), tolerance=.18, timeout=180)
            return self.result(controller_reached=reached, estimated_pose=nav.pose())
        except Exception as exc:
            return self.result(error=str(exc))
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission1))
