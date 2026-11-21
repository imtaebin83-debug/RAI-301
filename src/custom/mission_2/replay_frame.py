"""CPU-only replay of capture.py's sensor bundle; never connects to ROS."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

from custom.vision import RGBD


def replay(directory):
    with np.load(directory / 'sensors.npz', allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    rgb, depth = arrays['rgb'], arrays['depth_m'].astype('<f4')
    height, width = depth.shape
    color, distance, info = Image(), Image(), CameraInfo()
    color.width = distance.width = info.width = width
    color.height = distance.height = info.height = height
    color.encoding, color.step, color.data = 'rgb8', width*3, rgb.tobytes()
    distance.encoding, distance.step, distance.data = '32FC1', width*4, depth.tobytes()
    info.k = arrays['camera_k'].astype(float).reshape(-1).tolist()
    color.header.stamp.sec = distance.header.stamp.sec = 1
    pose = Rotation.from_quat(arrays['estimated_quaternion_xyzw']).as_matrix()
    camera = Rotation.from_quat(arrays['camera_in_pelvis_quaternion_xyzw']).as_matrix()
    rotation = pose @ camera
    translation = pose @ arrays['camera_in_pelvis_translation'] + arrays['estimated_position']
    mission = SimpleNamespace(latest=lambda topic, *a, **k: {
        '/camera/color/image_raw': color, '/camera/depth/image_raw': distance,
        '/camera/camera_info': info}[topic])
    nav = SimpleNamespace(camera_to_world=lambda points, frame: points @ rotation.T + translation)
    sensor = RGBD(mission, nav)
    started = time.monotonic()
    data = sensor.frame(timeout=0.)
    seconds = time.monotonic() - started
    if data is None:
        raise RuntimeError('Could not project captured sensor bundle')
    detections = {}
    for color in ('blue', 'green', 'orange'):
        found = sensor.cone(color, data=data)
        detections[color] = asdict(found) if found is not None else None
    return {'kind': 'single_saved_frame_not_accuracy_or_mission_validation',
            'source': str(directory), 'projection_seconds': seconds,
            'detections': detections}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.capture_directory), indent=2))
