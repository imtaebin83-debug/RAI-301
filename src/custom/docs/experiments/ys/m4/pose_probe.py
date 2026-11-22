"""Read-only evaluator diagnostic; never publishes commands or reads truth."""
import json
import sys
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: pose_probe.py OUTPUT.jsonl DURATION_SECONDS')
    duration = float(sys.argv[2])
    if not math.isfinite(duration) or not 0 < duration <= 900:
        raise SystemExit('Duration must be in (0,900] seconds')
    rclpy.init()
    node = Node('ys_m4_pose_diagnostic')
    started = time.monotonic()
    last, counts, largest = {}, {}, {}
    output = open(sys.argv[1], 'w', buffering=1)
    def observe(topic, message):
        now = time.monotonic()
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if topic in last:
            gap = now-last[topic][0]
            largest[topic] = max(largest.get(topic, 0.), gap)
            if gap > .5:
                record = dict(topic=topic, elapsed=now-started, receive_gap=gap,
                              stamp_delta=stamp-last[topic][1])
                if topic == 'pose':
                    p=message.pose.position
                    record['estimated_xy']=[p.x,p.y]
                output.write(json.dumps(record)+'\n')
        last[topic]=(now, stamp)
        counts[topic]=counts.get(topic, 0)+1
    node.create_subscription(PoseStamped, '/custom/estimated_pose', lambda m: observe('pose',m), qos_profile_sensor_data)
    node.create_subscription(PointCloud2, '/lidar/points', lambda m: observe('lidar',m), qos_profile_sensor_data)
    try:
        while rclpy.ok() and time.monotonic()-started < duration:
            rclpy.spin_once(node, timeout_sec=.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        output.write(json.dumps(dict(counts=counts,largest_receive_gaps=largest))+'\n')
        output.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
