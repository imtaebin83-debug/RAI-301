#!/usr/bin/env python3

from __future__ import annotations

import sys
import math
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from custom.sensor_mission import Mission
from mission_api import MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator
from custom.mission_2.perception import ConeSensor


class Mission2(Mission):
    MISSION = 2

    def __init__(self) -> None:
        super().__init__()

    def _check_respawn(self):
        """A public respawn rearranges targets; this attempt must not continue."""
        epoch = getattr(self, '_run_respawn_count', None)
        if epoch is not None and self.respawn_count != epoch:
            self.stop()
            raise RuntimeError('Public respawn invalidated mission observations; restart required')

    def _on_public_respawn(self, message):
        super()._on_public_respawn(message)
        # Also stop while the main thread is collecting cone observations.
        self.stop()

    def cmd_vel(self, vx=0., vy=0., wz=0.):
        # Serialize against the public notification so an old target cannot
        # issue another nonzero command after the respawn has been observed.
        with self._sensor_cv:
            if vx or vy or wz:
                self._check_respawn()
            super().cmd_vel(vx, vy, wz)

    def run(self) -> MissionResult:
        self._run_respawn_count = self.respawn_count
        deadline = time.monotonic() + 240.
        measurements = []
        final_stop = None

        def remaining(limit):
            self._check_respawn()
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError('Cone mission deadline reached')
            return min(value, limit)

        try:
            self.begin_mission(2, timeout=remaining(30.))
            color = self.params.get("color") or self.m2_color
            self.log(f"[mission 2] target cone: {color}")
            if color not in ('blue', 'green', 'orange'):
                return self.result(error='Missing valid cone color in planner parameters', reached=False)
            self._check_respawn()
            nav = Navigator(self, None)
            # These are published doorway coordinates, not hidden cone positions.
            if not nav.route_to(15.5, 22.3, timeout=remaining(140.)):
                raise RuntimeError('Could not reach cone observation point')
            sensor = ConeSensor(self, nav)
            detection = None
            for yaw in (math.pi/2, math.pi/2-.4, math.pi/2+.4):
                if not nav.face(yaw, timeout=remaining(12.)):
                    raise RuntimeError('Could not face cone survey heading')
                detection = sensor.observe_cone(color, seconds=remaining(3.0))
                self._check_respawn()
                if detection is not None:
                    break
            if detection is None:
                return self.result(error=f'No stable RGB-D detection of {color} cone', reached=False)
            self._check_respawn()
            self.log(f'[mission 2] sensor cone estimate: ({detection.x:.2f}, {detection.y:.2f})')
            # Preserve the baseline's 8cm surface correction until measured
            # simulation data supports changing it. It is not a measured radius.
            surface_correction = .08
            target = (detection.x, detection.y + surface_correction - .75)
            px, py, yaw = nav.pose()
            if not all(math.isfinite(value) for value in (px, py, yaw)):
                raise RuntimeError('Nonfinite sensor pose before cone approach')
            distance = math.hypot(target[0]-px, target[1]-py)
            measurements.append({'surface_xy': [detection.x, detection.y],
                                 'target_xy': list(target), 'error_m': distance})
            if distance > 3.5:
                raise RuntimeError('Measured cone approach exceeds observation-point range')
            # Live testing confirmed that the fixed camera loses the cone even
            # during approach. Navigate to the fresh measured target using the
            # shared continuous LiDAR pose, as the passing baseline does.
            if not nav.goto(*target, tolerance=.08, timeout=remaining(40.), speed=.18):
                raise RuntimeError('Final cone approach was not reached')
            # Mission 2 judges position only. An extra final rotation can move
            # the pelvis out of the reached stop box and trigger pose drift.
            # Hold the arrival heading, let walking settle, then verify position.
            self.stop()
            settle_seconds = remaining(.5)
            time.sleep(settle_seconds)
            self._check_respawn()
            vx, vy, yaw = nav.pose()
            if not all(math.isfinite(value) for value in (vx, vy, yaw)):
                raise RuntimeError('Nonfinite sensor pose at final cone stop')
            error = math.hypot(target[0]-vx, target[1]-vy)
            final_stop = {'estimated_pose': [vx, vy, yaw], 'target_xy': list(target),
                          'error_m': error, 'settle_seconds': settle_seconds}
            if error >= .15:
                raise RuntimeError('Settled stop moved outside the estimated cone target')
            self._check_respawn()
            return self.result(color=color, reached=True,
                cone_xy=[detection.x, detection.y], measurements=measurements,
                surface_correction_m=surface_correction,
                final_leg_m=distance, final_pose_error_m=error,
                final_stop=final_stop,
                final_verification='settled_sensor_pose_against_fresh_cone_target')
        except Exception as exc:
            invalidated = self.respawn_count != self._run_respawn_count
            return self.result(error=str(exc), reached=False,
                               measurements=[] if invalidated else measurements,
                               final_stop=None if invalidated else final_stop,
                               observations_invalidated=invalidated,
                               restart_required=invalidated,
                               respawns_during_run=self.respawn_count-self._run_respawn_count)
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission2))
