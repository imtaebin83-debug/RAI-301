"""Continuous LiDAR odometry; selected mission status is NOT a pose measurement."""
import math
import os
import time
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
from tf2_ros import TransformBroadcaster
from mission_api import MissionRunner, START_POSE_WORLD
from custom.localization import Localizer


def main():
    with MissionRunner('custom_lidar_localizer') as runner:
        runner.api.wait_for_sim(timeout=40)
        # README public initial pose. Override only for explicit evaluation spawns.
        initial=[float(v) for v in os.getenv('HAIC_INITIAL_POSE',','.join(map(str,START_POSE_WORLD))).split(',')]
        loc=Localizer(runner.api,{'pos':initial[:2], 'yaw_deg':initial[2], 'seq':-1})
        pub=runner.api.create_publisher(PoseStamped,'/custom/estimated_pose',10)
        tf=TransformBroadcaster(runner.api)
        lastlog=0
        while runner._rclpy.ok():
            loc.update()
            if not loc.stale:
                T=loc.T_world_pelvis
                q=Rotation.from_matrix(T[:3,:3]).as_quat()
                pose=PoseStamped()
                pose.header.frame_id='map'
                pose.header.stamp=runner.api.get_clock().now().to_msg()
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
                runner.api.log('[localizer] waiting: '+loc.last_error)
                lastlog=time.monotonic()
            time.sleep(.045)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
