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
from geometry_msgs.msg import PoseStamped

DOOR_Y = (17.5, 10.5, 3.59)  # Published north-to-south door geometry.


class Mission4(Mission):
    MISSION = 4

    def _navigate(self, nav, action, timeout):
        """Stop on a pose outage; retry only after sustained fresh poses, within budget.

        This does not reset or replace the shared estimator. A persistent loss
        remains a failed mission, and retries never extend the original leg.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return action(max(0., deadline - time.monotonic()))
            except RuntimeError as exc:
                self.stop()
                if not str(exc).startswith('Stale localization:'):
                    raise
                if self._recovery_attempts >= 3:
                    raise
                self._recovery_attempts += 1
                event = {'reason': str(exc), 'recovered': False}
                self._pose_recoveries.append(event)
                self.log('[mission 4] stopped: waiting up to 8s for fresh localization')
                started = time.monotonic()
                until = min(deadline, started + 8.)
                previous, advances = None, 0
                stable_since, last_advance = None, None
                while time.monotonic() < until:
                    self.stop()
                    now = time.monotonic()
                    if last_advance is not None and now-last_advance > .5:
                        stable_since, advances = None, 0
                    message = self.latest('/custom/estimated_pose', PoseStamped, timeout=0.)
                    if message is not None:
                        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
                        if previous is not None and stamp > previous:
                            advances += 1
                            last_advance = now
                            if stable_since is None:
                                stable_since = now
                        elif previous is not None and stamp < previous:
                            stable_since, advances = None, 0
                        previous = stamp
                        if advances >= 2 and now-stable_since >= 1.:
                            # Preserve Navigator's own freshness gate as well.
                            nav.pose()
                            event.update(recovered=True, wait_seconds=time.monotonic()-started)
                            self.log('[mission 4] fresh localization resumed; retrying remaining leg')
                            break
                    time.sleep(min(.05, max(0., until-time.monotonic())))
                else:
                    event['wait_seconds'] = time.monotonic()-started
                    raise RuntimeError('Localization did not recover while stopped') from exc
        self.stop()
        raise TimeoutError('Navigation deadline reached')

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
        if not self._navigate(nav, lambda timeout: nav.route_to(
                16.0, door_y, tolerance=.18, timeout=timeout), remaining(100)):
            raise RuntimeError('Could not reach room observation waypoint')

    def run(self) -> MissionResult:
        observations, answer, view_metadata = [], [], []
        vlm_retries = []
        assignment = None
        deadline = time.monotonic() + 600
        started = time.monotonic()
        stage = 'begin'
        self._recovery_attempts, self._pose_recoveries = 0, []

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
                self._navigate(nav, lambda timeout: self._face_room(
                    nav, time.monotonic()+timeout), remaining(18))
                # The forward heading gives an actual sensor view, with no
                # assumed room labels. No cached pre-turn image is accepted.
                for yaw in (0.,):
                    if not self._navigate(nav, lambda timeout: nav.face(yaw, timeout=timeout), remaining(8)):
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
            try:
                answer, assignment = assign_rooms(observations, min_margin=.25)
            except ValueError as exc:
                if str(exc) != 'VLM scores do not distinguish a room assignment':
                    raise
                stage = 'VLM crop retry'
                # Reuse actual captured pixels; this adds a second view of
                # the evidence, not an independent sensor observation.
                crops = []
                boxes = []
                for index, image in enumerate(images):
                    width, height = image.size
                    box = (width//10, 0, 9*width//10, 4*height//5)
                    crop = image.crop(box)
                    crops.append(crop)
                    boxes.append(list(box))
                    capture = os.getenv('HAIC_M4_CAPTURE_DIR')
                    if capture:
                        crop.save(Path(capture) / f'crop-{index}.png')
                vlm_retries.append({'reason': str(exc), 'crop_boxes': boxes,
                                    'source': 'same captured camera frames'})
                self.log('[mission 4] ambiguous VLM survey; scoring fixed room crops once')
                cropped = classify_views(crops, timeout=remaining(180))
                observations = [combine_views([full, crop]) for full, crop in zip(raw, cropped)]
                answer, assignment = assign_rooms(observations, min_margin=.25)
            self.log(f"[mission 4] VLM joint assignment {answer}: {assignment}")
            classroom_y = DOOR_Y[answer.index("classroom")]
            stage = 'classroom return'
            self._travel(nav, classroom_y, remaining)
            self._navigate(nav, lambda timeout: self._face_room(
                nav, time.monotonic()+timeout), remaining(18))
            stage = 'classroom entry'
            # Aim well inside the published room, rather than stopping only
            # 0.4m beyond a doorway using a drifting global pose estimate.
            reached = self._navigate(nav, lambda timeout: nav.goto(
                20.0, classroom_y, tolerance=.15, timeout=timeout, speed=.20), remaining(40))
            if not reached:
                raise RuntimeError("could not enter the classroom")
            return self.result(rooms=answer, vlm_observations=observations,
                               pose_recoveries=self._pose_recoveries,
                               vlm_retries=vlm_retries,
                               room_assignment=assignment, views=view_metadata,
                               elapsed_seconds=time.monotonic()-started)
        except Exception as exc:
            self.log(f"[mission 4] incomplete: {exc}")
            return self.result(rooms=answer, vlm_observations=observations,
                               pose_recoveries=self._pose_recoveries,
                               vlm_retries=vlm_retries,
                               room_assignment=assignment, views=view_metadata,
                               failed_stage=stage, error=str(exc),
                               elapsed_seconds=time.monotonic()-started)
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission4))
