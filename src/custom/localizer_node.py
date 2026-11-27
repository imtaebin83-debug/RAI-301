"""Continuous LiDAR odometry; selected mission status is NOT a pose measurement."""
import math
import os
import time
import threading
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
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
    executor = MultiThreadedExecutor()
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
        # README public initial pose. Override only for explicit evaluation spawns.
        initial=[float(v) for v in os.getenv('HAIC_INITIAL_POSE',','.join(map(str,START_POSE_WORLD))).split(',')]
        loc=Localizer(node,{'pos':initial[:2], 'yaw_deg':initial[2]})
        pub=node.create_publisher(PoseStamped,'/custom/estimated_pose',10)
        tf=TransformBroadcaster(node)
        lastlog=0
        while rclpy.ok():
            loc.update()
            if not loc.stale:
                T=loc.T_world_pelvis
                q=Rotation.from_matrix(T[:3,:3]).as_quat()
                pose=PoseStamped()
                pose.header.frame_id='map'
                pose.header.stamp=node.get_clock().now().to_msg()
                p=pose.pose.position
                p.x,p.y,p.z=map(float,T[:3,3])
                r=pose.pose.orientation
                r.x,r.y,r.z,r.w=map(float,q)
                pub.publish(pose)
                t=TransformStamped()
                t.header=pose.header
                t.child_frame_id='pelvis'
                t.transform.translation.x=p.x
                t.transform.translation.y=p.y
                t.transform.translation.z=p.z
                t.transform.rotation=r
                tf.sendTransform(t)
            elif time.monotonic()-lastlog>5:
                node.log('[localizer] waiting: '+loc.last_error)
                lastlog=time.monotonic()
            time.sleep(.045)
    finally:
        executor.shutdown(timeout_sec=2.)
        thread.join(timeout=2.)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
