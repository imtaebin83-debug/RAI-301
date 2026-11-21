"""Cone-only freshness and tracking; shared ball perception is unchanged."""
import math
import time

import numpy as np
from sensor_msgs.msg import Image

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


class ConeSensor(RGBD):
    def observe_cone(self, color, seconds, previous=None):
        """Accept only new RGB/depth pairs received after this stopped survey."""
        self.mission.stop()
        deadline = time.monotonic() + seconds
        rgb = self.mission.latest('/camera/color/image_raw', Image, timeout=0.)
        depth = self.mission.latest('/camera/depth/image_raw', Image, timeout=0.)
        last_rgb = stamp(rgb) if rgb is not None else -1
        last_depth = stamp(depth) if depth is not None else -1
        observations = []
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
                    # Never reuse a projected frame captured at a prior waypoint.
                    self._stamp = None
                    self._cache = None
                    try:
                        data = super().frame(timeout=0.)
                        # frame() obtains immutable ROS messages again. Reject a
                        # racing depth update instead of attributing it to this pair.
                        if data is not None and self._stamp == (depth_stamp // 1_000_000_000,
                                                               depth_stamp % 1_000_000_000):
                            result = self.cone(color, data=data)
                            if result is not None:
                                observations.append(result)
                    except (ValueError, TypeError, BufferError):
                        pass
            time.sleep(min(.05, max(0., deadline - time.monotonic())))
        return consensus(observations, previous)
