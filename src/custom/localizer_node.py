"""Continuous sensor odometry, stamped with the scan that produced each pose."""
import json
import math
import os
import time
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from mission_api import MissionRunner, START_POSE_WORLD
from custom.localization import Localizer


def main():
    with MissionRunner('custom_lidar_localizer') as runner:
        runner.api.wait_for_sim(timeout=40)
        initial = [float(v) for v in os.getenv(
            'HAIC_INITIAL_POSE', ','.join(map(str, START_POSE_WORLD))).split(',')]
        if len(initial) != 3 or not all(math.isfinite(v) for v in initial):
            raise ValueError('HAIC_INITIAL_POSE requires finite x,y,yaw_deg')
        loc = Localizer(runner.api, {'pos': initial[:2], 'yaw_deg': initial[2], 'seq': -1})
        pub = runner.api.create_publisher(PoseStamped, '/custom/estimated_pose', 10)
        status_pub = runner.api.create_publisher(String, '/custom/localization_status', 10)
        tf = TransformBroadcaster(runner.api)
        lastlog = 0
        published_stamp = None
        traced_state = None
        trace_path = os.getenv('HAIC_LOCALIZATION_TRACE')
        trace = open(trace_path, 'a', buffering=1) if trace_path else None
        try:
            while runner._rclpy.ok():
                started = time.monotonic()
                loc.update()
                stamp = loc.measurement_stamp
                valid = not loc.stale
                age = time.monotonic() - loc._last_received if stamp is not None else None
                reason = loc.last_error
                if not valid and not reason:
                    reason = (f'no accepted LiDAR measurement for {age:.2f}s'
                              if age is not None else 'waiting for first accepted LiDAR measurement')
                status = {'version': 1, 'valid': valid, 'measurement_stamp': stamp,
                          'terminal': loc._recovery_blocked,
                          'backend': loc.backend, 'quality': loc.quality,
                          'reason': reason, 'measurement_age': age, 'rejected_scans': loc.rejected_scans,
                          'recovery_count': loc.recovery_count, 'metrics': loc.metrics}
                if trace:
                    trace_key = (stamp, valid, loc.last_error, loc.rejected_scans)
                    if trace_key != traced_state:
                        record = dict(status, update_seconds=time.monotonic() - started,
                                      wall_monotonic=time.monotonic())
                        if valid:
                            record['transform'] = loc.T_world_pelvis.tolist()
                        trace.write(json.dumps(record, allow_nan=False) + '\n')
                        traced_state = trace_key
                status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
                if valid and stamp != published_stamp:
                    T = loc.T_world_pelvis
                    q = Rotation.from_matrix(T[:3, :3]).as_quat()
                    pose = PoseStamped()
                    pose.header.frame_id = 'map'
                    pose.header.stamp.sec, pose.header.stamp.nanosec = stamp
                    p, r = pose.pose.position, pose.pose.orientation
                    p.x, p.y, p.z = map(float, T[:3, 3])
                    r.x, r.y, r.z, r.w = map(float, q)
                    pub.publish(pose)
                    t = TransformStamped()
                    t.header = pose.header
                    t.child_frame_id = 'pelvis'
                    t.transform.translation.x = p.x
                    t.transform.translation.y = p.y
                    t.transform.translation.z = p.z
                    t.transform.rotation = r
                    tf.sendTransform(t)
                    published_stamp = stamp
                elif not valid and time.monotonic() - lastlog > 5:
                    runner.api.log('[localizer] waiting: ' + reason)
                    lastlog = time.monotonic()
                time.sleep(.025)
        finally:
            if trace:
                trace.close()


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
