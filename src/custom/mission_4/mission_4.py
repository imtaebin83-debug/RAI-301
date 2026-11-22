#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator
from custom.vlm import assign_rooms, classify_views, combine_views

DOOR_Y = (17.5, 10.5, 3.59)  # Published north-to-south door geometry.


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
        deadline = time.monotonic() + timeout
        message = self.latest("/camera/color/image_raw", Image, timeout=0.)
        previous = ((message.header.stamp.sec, message.header.stamp.nanosec)
                    if message is not None else None)
        fresh = 0
        while time.monotonic() < deadline:
            message = self.latest("/camera/color/image_raw", Image, timeout=0.)
            if message is not None:
                stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
                if previous is None or stamp > previous:
                    previous = stamp
                    fresh += 1
                    if fresh >= 2:
                        encoding = message.encoding.lower()
                        if encoding not in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
                            raise RuntimeError('Unsupported RGB camera encoding')
                        channels = 4 if 'a' in encoding else 3
                        image = np.ndarray((message.height, message.width, channels),
                            dtype=np.uint8, buffer=message.data,
                            strides=(message.step, channels, 1))[:, :, :3]
                        if encoding.startswith('bgr'):
                            image = image[:, :, ::-1]
                        self._last_image_stamp = list(stamp)
                        return PILImage.fromarray(image.copy())
            time.sleep(min(.04, max(0., deadline-time.monotonic())))
        raise TimeoutError('Fresh room camera frames unavailable after alignment')

    def _travel(self, nav, door_y, remaining):
        """Use a single corridor line and avoid repeated cross-corridor turns."""
        if not nav.route_to(16.0, door_y, tolerance=.18, timeout=remaining(100)):
            raise RuntimeError('Could not reach room observation waypoint')

    def run(self) -> MissionResult:
        observations, answer, view_metadata = [], [], []
        assignment = None
        deadline = time.monotonic() + 600
        started = time.monotonic()
        stage = 'begin'

        def remaining(limit):
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("room survey deadline reached")
            return min(limit, seconds)

        try:
            start = self.begin_mission(4, timeout=remaining(60))
            nav = Navigator(self, start)
            images = []
            for door_y in DOOR_Y:
                stage = f'survey door {door_y}'
                self._travel(nav, door_y, remaining)
                self._face_room(nav, min(deadline, time.monotonic() + 18))
                # The forward heading gives an actual sensor view, with no
                # assumed room labels. No cached pre-turn image is accepted.
                for yaw in (0.,):
                    if not nav.face(yaw, timeout=remaining(8)):
                        raise RuntimeError('Room observation heading was not reached')
                    self.stop()
                    time.sleep(remaining(.3))
                    image = self._image(timeout=remaining(5))
                    capture = os.getenv('HAIC_M4_CAPTURE_DIR')
                    if capture:
                        directory = Path(capture)
                        directory.mkdir(parents=True, exist_ok=True)
                        image.save(directory / f'view-{len(images)}.png')
                    images.append(image)
                    view_metadata.append({'door_y': door_y, 'stamp': self._last_image_stamp,
                                          'estimated_pose': list(nav.pose()),
                                          'elapsed_seconds': time.monotonic()-started})
            stage = 'VLM survey inference'
            raw = classify_views(images, timeout=remaining(240))
            observations = [combine_views([item]) for item in raw]
            # A small nonzero margin is a rejection threshold, not a calibrated
            # confidence probability. Record it for simulation calibration.
            answer, assignment = assign_rooms(observations, min_margin=.25)
            self.log(f"[mission 4] VLM joint assignment {answer}: {assignment}")
            classroom_y = DOOR_Y[answer.index("classroom")]
            stage = 'classroom return'
            self._travel(nav, classroom_y, remaining)
            self._face_room(nav, min(deadline, time.monotonic()+18))
            stage = 'classroom entry'
            # Aim well inside the published room, rather than stopping only
            # 0.4m beyond a doorway using a drifting global pose estimate.
            reached = nav.goto(20.0, classroom_y, tolerance=.15,
                               timeout=remaining(40), speed=.20)
            if not reached:
                raise RuntimeError("could not enter the classroom")
            return self.result(rooms=answer, vlm_observations=observations,
                               room_assignment=assignment, views=view_metadata,
                               elapsed_seconds=time.monotonic()-started)
        except Exception as exc:
            self.log(f"[mission 4] incomplete: {exc}")
            return self.result(rooms=answer, vlm_observations=observations,
                               room_assignment=assignment, views=view_metadata,
                               failed_stage=stage, error=str(exc),
                               elapsed_seconds=time.monotonic()-started)
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission4))
