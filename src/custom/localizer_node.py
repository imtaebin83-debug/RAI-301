"""Continuous public-sensor odometry stamped with each accepted LiDAR scan."""
import json
import math
import os
import threading
import time
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from mission_api import START_POSE_WORLD
from custom.sensor_mission import Mission
from custom.localization import Localizer


def main():
    import rclpy
    from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import PointCloud2

    rclpy.init()
    node = Mission(node_name='custom_lidar_localizer')
    node.set_parameters([Parameter('use_sim_time', value=True)])
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        if node.latest('/lidar/points', PointCloud2, timeout=40.) is None:
            raise TimeoutError('No public LiDAR sensor within 40 seconds')
        initial = [float(v) for v in os.getenv(
            'HAIC_INITIAL_POSE', ','.join(map(str, START_POSE_WORLD))).split(',')]
        if len(initial) != 3 or not all(math.isfinite(v) for v in initial):
            raise ValueError('HAIC_INITIAL_POSE requires finite x,y,yaw_deg')
        loc = Localizer(node, {'pos': initial[:2], 'yaw_deg': initial[2], 'seq': -1})
        pub = node.create_publisher(PoseStamped, '/custom/estimated_pose', 10)
        status_pub = node.create_publisher(String, '/custom/localization_status', 10)
        tf = TransformBroadcaster(node)
        lastlog = 0
        published_stamp = None
        traced_state = None
        trace_path = os.getenv('HAIC_LOCALIZATION_TRACE')
        trace = open(trace_path, 'a', buffering=1) if trace_path else None
        try:
            while rclpy.ok():
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
                    node.log('[localizer] waiting: ' + reason)
                    lastlog = time.monotonic()
                time.sleep(.025)
        finally:
            if trace:
                trace.close()
    finally:
        executor.shutdown(timeout_sec=2.)
        thread.join(timeout=2.)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
