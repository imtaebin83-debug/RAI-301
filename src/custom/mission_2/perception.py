"""Cone-only freshness and tracking; shared ball perception is unchanged."""
import math
import os
from pathlib import Path
import time

import numpy as np
from sensor_msgs.msg import CameraInfo, Image

from custom.vision import Detection, RGBD


def stamp(message):
    return (message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec)


def consensus(observations, previous=None):
    """Require three mutually consistent measurements, optionally of one track."""
    accepted = [item for item in observations if item.valid()]
    if previous is not None:
        accepted = [item for item in accepted
                    if math.hypot(item.x - previous.x, item.y - previous.y) < .45]
    if len(accepted) < 3:
        return None
    xy = np.array([(item.x, item.y) for item in accepted])
    center = np.median(xy, axis=0)
    keep = np.linalg.norm(xy - center, axis=1) <= .15
    if int(keep.sum()) < 3 or int(keep.sum()) < math.ceil(.7 * len(accepted)):
        return None
    selected = [item for item, good in zip(accepted, keep) if good]
    center = np.median(xy[keep], axis=0)
    last = selected[-1]
    return Detection(float(center[0]), float(center[1]),
                     float(np.median([item.distance for item in selected])),
                     last.pixel, last.area)


class _FrameSnapshot:
    def __init__(self):
        self.messages = {}

    def latest(self, topic, *args, **kwargs):
        return self.messages.get(topic)


class ConeSensor(RGBD):
    def __init__(self, mission, nav):
        super().__init__(mission, nav)
        self._snapshot = _FrameSnapshot()
        self._projection = RGBD(self._snapshot, nav)

    def observe_cone(self, color, seconds, previous=None):
        """Accept only new RGB/depth pairs received after this stopped survey."""
        self.mission.stop()
        deadline = time.monotonic() + seconds
        rgb = self.mission.latest('/camera/color/image_raw', Image, timeout=0.)
        depth = self.mission.latest('/camera/depth/image_raw', Image, timeout=0.)
        self.mission.latest('/camera/camera_info', CameraInfo, timeout=0.)
        last_rgb = stamp(rgb) if rgb is not None else -1
        last_depth = stamp(depth) if depth is not None else -1
        observations = []
        last_data = None
        while time.monotonic() < deadline:
            rgb = self.mission.latest('/camera/color/image_raw', Image, timeout=0.)
            depth = self.mission.latest('/camera/depth/image_raw', Image, timeout=0.)
            if rgb is not None and depth is not None:
                rgb_stamp, depth_stamp = stamp(rgb), stamp(depth)
                if (rgb_stamp > last_rgb and depth_stamp > last_depth
                        and abs(rgb_stamp - depth_stamp) <= 80_000_000
                        and rgb.encoding.lower() in ('rgb8', 'bgr8', 'rgba8', 'bgra8')
                        and depth.encoding.upper() in ('32FC1', '16UC1')):
                    last_rgb, last_depth = rgb_stamp, depth_stamp
                    info = self.mission.latest('/camera/camera_info', CameraInfo, timeout=0.)
                    # Project exactly the immutable messages that passed the
                    # pairing check. Re-reading live caches here used to race
                    # the next camera update and silently discard good frames.
                    self._snapshot.messages = {'/camera/color/image_raw': rgb,
                        '/camera/depth/image_raw': depth, '/camera/camera_info': info}
                    try:
                        data = self._projection.frame(timeout=0.)
                        if data is not None:
                            last_data = data
                            result = self.cone(color, data=data)
                            if result is not None:
                                observations.append(result)
                    except (ValueError, TypeError, BufferError):
                        pass
            time.sleep(min(.05, max(0., deadline - time.monotonic())))
        result = consensus(observations, previous)
        if result is None:
            self.mission.log(f'[cone] survey {color}: {len(observations)} candidates; '
                             f'xy={[(round(o.x, 2), round(o.y, 2)) for o in observations[-6:]]}')
            output = os.getenv('HAIC_M2_CAPTURE_DIR')
            if output and last_data is not None:
                from PIL import Image as PILImage
                directory = Path(output)
                directory.mkdir(parents=True, exist_ok=True)
                name = f'{color}-{time.time_ns()}'
                PILImage.fromarray(last_data[0]).save(directory / (name+'.png'))
                np.savez_compressed(directory / (name+'.npz'), rgb=last_data[0],
                    depth=last_data[1], world=last_data[2], height=last_data[3], valid=last_data[4])
        return result
