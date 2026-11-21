#!/usr/bin/env python3
"""Read-only RGB-D/estimated-pose capture for simulator diagnostics.

Run as: python custom/testing/capture.py /tmp/rai-capture --seconds 2
No robot commands, simulator status, scenario or ground-truth topics are used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


def stamp(header):
    return {'sec': header.stamp.sec, 'nanosec': header.stamp.nanosec,
            'frame_id': header.frame_id}


def image_array(message):
    encoding = message.encoding.lower()
    if encoding in ('32fc1', '16uc1'):
        item = 'f4' if encoding == '32fc1' else 'u2'
        dtype = np.dtype(('>' if message.is_bigendian else '<') + item)
        result = np.ndarray((message.height, message.width), dtype=dtype,
                            buffer=message.data, strides=(message.step, dtype.itemsize)).copy()
        return result.astype(np.float32) * (.001 if encoding == '16uc1' else 1.)
    if encoding not in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
        raise ValueError(f'Unsupported camera encoding: {encoding}')
    channels = 4 if 'a' in encoding else 3
    result = np.ndarray((message.height, message.width, channels), dtype=np.uint8,
                        buffer=message.data, strides=(message.step, channels, 1))[:,:,:3].copy()
    return result[:,:,::-1] if encoding.startswith('bgr') else result


class Capture(Node):
    def __init__(self):
        super().__init__('custom_sensor_capture')
        self.messages = {}
        self.counts = {}
        for name, topic, cls, qos in (
            ('rgb', '/camera/color/image_raw', Image, qos_profile_sensor_data),
            ('depth', '/camera/depth/image_raw', Image, qos_profile_sensor_data),
            ('camera_info', '/camera/camera_info', CameraInfo, qos_profile_sensor_data),
            ('estimated_pose', '/custom/estimated_pose', PoseStamped, 10),
        ):
            self.create_subscription(cls, topic, lambda msg, name=name: self.receive(name,msg), qos)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)

    def receive(self, name, message):
        self.messages[name] = message
        self.counts[name] = self.counts.get(name,0)+1

    def save(self, directory):
        directory.mkdir(parents=True,exist_ok=True)
        record = {'captured_wall_unix': time.time(), 'message_counts': self.counts,
                  'headers': {name:stamp(msg.header) for name,msg in self.messages.items()}}
        arrays = {}
        if 'rgb' in self.messages:
            arrays['rgb'] = image_array(self.messages['rgb'])
            PILImage.fromarray(arrays['rgb']).save(directory/'rgb.png')
        if 'depth' in self.messages:
            arrays['depth_m'] = image_array(self.messages['depth'])
            finite = arrays['depth_m'][np.isfinite(arrays['depth_m']) & (arrays['depth_m']>0)]
            record['depth_stats'] = {'finite_positive': int(finite.size),
                                     'min_m':float(finite.min()) if finite.size else None,
                                     'max_m':float(finite.max()) if finite.size else None}
        if 'camera_info' in self.messages:
            info = self.messages['camera_info']
            arrays['camera_k'] = np.asarray(info.k).reshape(3,3)
            record['camera_info'] = {'width':info.width,'height':info.height,
                                     'k':list(info.k),'d':list(info.d),
                                     'distortion_model':info.distortion_model}
            try:
                transform = self.tf.lookup_transform('pelvis',info.header.frame_id,Time()).transform
                translation = [transform.translation.x,transform.translation.y,transform.translation.z]
                quaternion = [transform.rotation.x,transform.rotation.y,transform.rotation.z,transform.rotation.w]
                arrays['camera_in_pelvis_translation'] = np.asarray(translation)
                arrays['camera_in_pelvis_quaternion_xyzw'] = np.asarray(quaternion)
                record['camera_in_pelvis'] = {'translation':translation,'quaternion_xyzw':quaternion}
            except Exception as error:
                record['tf_error'] = str(error)
        if 'estimated_pose' in self.messages:
            pose = self.messages['estimated_pose'].pose
            position = [pose.position.x,pose.position.y,pose.position.z]
            quaternion = [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
            arrays['estimated_position'] = np.asarray(position)
            arrays['estimated_quaternion_xyzw'] = np.asarray(quaternion)
            qx,qy,qz,qw = quaternion
            record['estimated_pose'] = {'position':position,'quaternion_xyzw':quaternion,
                'yaw_rad':math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))}
        np.savez_compressed(directory/'sensors.npz',**arrays)
        (directory/'snapshot.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--discovery-timeout',type=float,default=8.)
    args = parser.parse_args()
    if not 0<args.seconds<=30 or not 0<args.discovery_timeout<=60:
        parser.error('duration must be positive and bounded (30s capture, 60s discovery maximum)')
    rclpy.init()
    node = Capture()
    ready = None
    started = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.1)
            now = time.monotonic()
            if {'rgb','depth','camera_info'}.issubset(node.messages):
                ready = ready or now
            if ready is not None and now-ready>=args.seconds:
                break
            if now-started>=args.discovery_timeout+args.seconds:
                break
        node.save(args.output)
        return 0 if {'rgb','depth','camera_info','estimated_pose'}.issubset(node.messages) else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    raise SystemExit(main())
