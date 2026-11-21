#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator
import time


class Mission3(Mission):
    MISSION = 3

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        start = self.begin_mission(3)
        try:
            nav = Navigator(self, start)
            deadline = time.monotonic() + 600
            # README publishes these gaps and the alternating 1.15 m floor strips.
            path = [(14.,20.125),(12.4,20.125),(6.075,20.125),
                    (6.075,18.9),(12.425,18.9),(12.425,17.2),
                    (6.075,17.2),(6.075,15.4),(12.425,15.4),
                    (12.425,13.4),(6.075,13.4),(6.075,11.375),
                    (13.65,11.375)]
            reached=True
            falls=self.gap_falls
            for x,y in path:
                if self.gap_falls != falls or time.monotonic() >= deadline:
                    reached=False
                    break
                if not nav.goto(x,y,tolerance=.12,timeout=min(85,deadline-time.monotonic()),speed=.28):
                    reached=False
                    break
            return self.result(controller_reached=reached,gap_falls=self.gap_falls)
        except Exception as exc:
            return self.result(error=str(exc),gap_falls=self.gap_falls)
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission3))
