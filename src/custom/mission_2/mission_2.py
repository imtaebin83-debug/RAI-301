#!/usr/bin/env python3

from __future__ import annotations

import sys
import math
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator
from custom.mission_2.perception import ConeSensor


class Mission2(Mission):
    MISSION = 2

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        deadline = time.monotonic() + 240.
        measurements = []

        def remaining(limit):
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError('Cone mission deadline reached')
            return min(value, limit)

        try:
            start = self.begin_mission(2, timeout=remaining(30.))
            color = self.params.get("color") or self.m2_color
            self.log(f"[mission 2] target cone: {color}")
            if color not in ('blue', 'green', 'orange'):
                return self.result(error='Missing valid cone color in planner parameters', reached=False)
            nav = Navigator(self, start)
            # These are published doorway coordinates, not hidden cone positions.
            if not nav.route_to(15.5, 22.3, timeout=remaining(140.)):
                return self.result(error='Could not reach cone observation point', reached=False)
            sensor = ConeSensor(self, nav)
            detection = None
            for yaw in (math.pi/2, math.pi/2-.4, math.pi/2+.4):
                if not nav.face(yaw, timeout=remaining(12.)):
                    raise RuntimeError('Could not face cone survey heading')
                detection = sensor.observe_cone(color, seconds=remaining(1.5))
                if detection is not None:
                    break
            if detection is None:
                return self.result(error=f'No stable RGB-D detection of {color} cone', reached=False)
            self.log(f'[mission 2] sensor cone estimate: ({detection.x:.2f}, {detection.y:.2f})')
            # Preserve the baseline's 8cm surface correction until measured
            # simulation data supports changing it. It is not a measured radius.
            surface_correction = .08
            for _ in range(7):
                target = (detection.x, detection.y + surface_correction - .75)
                px, py, _ = nav.pose()
                distance = math.hypot(target[0]-px, target[1]-py)
                measurements.append({'surface_xy': [detection.x, detection.y],
                                     'target_xy': list(target), 'error_m': distance})
                if distance <= .70:
                    # A saved close-range sensor capture shows the fixed camera
                    # can look above every cone near the stopping box. Finish
                    # this bounded leg using continuous LiDAR pose feedback and
                    # the just-measured target; do not require invisible pixels.
                    if not nav.goto(*target, tolerance=.08, timeout=remaining(14.), speed=.14):
                        raise RuntimeError('Final cone approach was not reached')
                    if not nav.face(math.pi/2, timeout=remaining(8.)):
                        raise RuntimeError('Final cone heading was not reached')
                    vx, vy, _ = nav.pose()
                    error = math.hypot(target[0]-vx, target[1]-vy)
                    if error < .12:
                        return self.result(color=color, reached=True,
                            cone_xy=[detection.x, detection.y], measurements=measurements,
                            surface_correction_m=surface_correction,
                            final_leg_m=distance, final_pose_error_m=error,
                            final_verification='continuous_sensor_pose_against_last_fresh_cone_target')
                    raise RuntimeError('Final rotation moved outside the estimated cone target')
                step = min(.55, distance)
                waypoint = (px+(target[0]-px)*step/distance,
                            py+(target[1]-py)*step/distance)
                if not nav.goto(*waypoint, tolerance=.08, timeout=remaining(14.), speed=.18):
                    raise RuntimeError('Cone approach waypoint was not reached')
                px, py, _ = nav.pose()
                if not nav.face(math.atan2(detection.y-py, detection.x-px),
                                timeout=remaining(8.)):
                    raise RuntimeError('Could not keep target cone in view')
                refreshed = sensor.observe_cone(color, remaining(1.2), previous=detection)
                if refreshed is None:
                    raise RuntimeError('Fresh target cone observations lost during approach')
                detection = refreshed
            raise RuntimeError('Cone approach did not converge within seven observations')
        except Exception as exc:
            return self.result(error=str(exc), reached=False, measurements=measurements)
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission2))
