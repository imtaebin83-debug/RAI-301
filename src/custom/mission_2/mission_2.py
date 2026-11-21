#!/usr/bin/env python3

from __future__ import annotations

import sys
import math
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator
from custom.vision import RGBD


class Mission2(Mission):
    MISSION = 2

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        start = self.begin_mission(2)
        color = self.params.get("color") or self.m2_color
        self.log(f"[mission 2] target cone: {color}")
        try:
            nav = Navigator(self, start)
            if color not in ('blue', 'green', 'orange'):
                return self.result(error='Missing valid cone color in planner parameters')
            # These are published doorway coordinates, not hidden cone positions.
            if not nav.route_to(15.5, 22.3, timeout=140.):
                return self.result(error='Could not reach cone observation point')
            nav.face(math.pi/2)
            sensor = RGBD(self, nav)
            detection = sensor.observe('cone', seconds=2.5, color=color)
            if detection is None:
                for yaw in (math.pi/2-.4, math.pi/2+.4, math.pi/2):
                    nav.face(yaw)
                    detection = sensor.observe('cone', seconds=1.2, color=color)
                    if detection:
                        break
            if detection is None:
                return self.result(error=f'No stable RGB-D detection of {color} cone')
            self.log(f'[mission 2] sensor cone estimate: ({detection.x:.2f}, {detection.y:.2f})')
            # Measured points lie on the cone's front surface. A small radius
            # correction precedes the published 0.75m door-side offset.
            target = (detection.x, detection.y-.67)
            reached = nav.goto(*target, tolerance=.12, timeout=55.)
            nav.face(math.pi/2, timeout=8.)
            return self.result(color=color, cone_xy=[detection.x,detection.y], reached=reached)
        except Exception as exc:
            return self.result(error=str(exc))
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission2))
