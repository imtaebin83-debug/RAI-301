#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator
from custom.vlm import assign_rooms, classify_room


class Mission4(Mission):
    MISSION = 4

    def _face_room(self, nav, deadline):
        while time.monotonic() < deadline:
            error = math.atan2(math.sin(-nav.pose()[2]), math.cos(-nav.pose()[2]))
            if abs(error) < 0.06:
                self.stop()
                return
            self.cmd_vel(0, 0, max(-0.4, min(0.4, error)))
            time.sleep(0.08)
        raise TimeoutError("could not face room before deadline")

    def _image(self, timeout=8.0):
        import numpy as np
        from PIL import Image as PILImage
        from sensor_msgs.msg import Image
        message = self.latest("/camera/color/image_raw", Image, timeout=timeout)
        if message is None or message.encoding.lower() not in ("rgb8", "bgr8"):
            raise RuntimeError("RGB camera unavailable or unsupported encoding")
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = rows[:, :message.width * 3].reshape(message.height, message.width, 3)
        if message.encoding.lower() == "bgr8":
            image = image[:, :, ::-1]
        return PILImage.fromarray(image.copy())

    def run(self) -> MissionResult:
        observations, answer = [], []
        assignment = None
        deadline = time.monotonic() + 480

        def remaining(limit):
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("room survey deadline reached")
            return min(limit, seconds)

        try:
            start = self.begin_mission(4, timeout=remaining(60))
            nav = Navigator(self, start)
            # README door geometry, never the hidden room assignment.
            for door_y in (17.5, 10.5, 3.59):
                if time.monotonic() > deadline - 10:
                    raise TimeoutError("room survey deadline reached")
                reached = nav.route_to(16.9, door_y, tolerance=0.22,
                    timeout=remaining(100))
                if reached is False:
                    raise RuntimeError("could not reach room observation waypoint")
                self._face_room(nav, min(deadline, time.monotonic() + 18))
                time.sleep(remaining(0.6))
                image = self._image(timeout=remaining(8))
                observation = classify_room(image, timeout=remaining(65))
                observations.append(observation)
                answer.append(observation["label"])
                self.log(f"[mission 4] VLM door y={door_y}: {observation}")
            answer, assignment = assign_rooms(observations)
            self.log(f"[mission 4] VLM joint assignment {answer}: {assignment}")
            classroom_y = (17.5, 10.5, 3.59)[answer.index("classroom")]
            reached = nav.route_to(16.9, classroom_y, tolerance=.2, timeout=remaining(100))
            if reached is False:
                raise RuntimeError("could not return to classroom doorway")
            reached = nav.goto(18.65, classroom_y, tolerance=.18, timeout=remaining(45))
            if reached is False:
                raise RuntimeError("could not enter the classroom")
            return self.result(rooms=answer, vlm_observations=observations, room_assignment=assignment)
        except Exception as exc:
            self.log(f"[mission 4] incomplete: {exc}")
            return self.result(rooms=answer, vlm_observations=observations,
                               room_assignment=assignment, error=str(exc))
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission4))
