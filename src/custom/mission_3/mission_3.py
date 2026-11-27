#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from custom.sensor_mission import Mission
from mission_api import MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator


class Mission3(Mission):
    MISSION = 3

    # Public route from the launch point through the alternating floor strips.
    ENTRY = (14.0, 20.125)
    COURSE = (
        (12.4, 20.125),
        (9.25, 20.125),
        (6.075, 20.125),
        (6.075, 18.9),
        (9.25, 18.9),
        (12.425, 18.9),
        (12.425, 17.2),
        (9.25, 17.2),
        (6.075, 17.2),
        (6.075, 15.4),
        (9.25, 15.4),
        (12.425, 15.4),
        (12.425, 13.4),
        (9.25, 13.4),
        (6.075, 13.4),
        (6.075, 11.375),
        (9.25, 11.375),
        (13.65, 11.375),
    )

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        self.begin_mission(3)
        try:
            nav = Navigator(self, None)
            deadline = time.monotonic() + 600
            falls = self.gap_falls
            completed = 0
            last_target = None
            failure_reason = ""

            route = (self.ENTRY, *self.COURSE)
            for last_target in route:
                if self.gap_falls != falls:
                    failure_reason = "public respawn observed"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure_reason = "mission deadline reached"
                    break
                if not nav.goto(
                    last_target[0],
                    last_target[1],
                    tolerance=0.18,
                    timeout=min(85.0, remaining),
                    speed=0.22,
                ):
                    failure_reason = f"waypoint not reached ({last_target[0]:.2f}, {last_target[1]:.2f})"
                    break
                completed += 1

            reached = not failure_reason
            self.log(
                f"[mission3] route {'complete' if reached else 'stopped'}; "
                f"waypoints={completed}; last_target={last_target}; reason={failure_reason or 'none'}"
            )
            return self.result(
                controller_reached=reached,
                route_waypoints_completed=completed,
                last_target=last_target,
                failure_reason=failure_reason,
                gap_falls=self.gap_falls,
            )
        except Exception as exc:
            return self.result(error=str(exc), gap_falls=self.gap_falls)
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission3))
