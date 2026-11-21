"""Shared sensor-based navigation; coordinates describe the public building only."""
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ContinuousPose:
    def __init__(self,mission):
        self.mission=mission
        self._stamp=None
        self._received=0
        self.T_world_pelvis=np.eye(4)
        self.last_error='bringup localizer has not published a pose'

    @property
    def stale(self):
        return time.monotonic()-self._received>1.2

    def update(self):
        msg=self.mission.latest('/custom/estimated_pose',PoseStamped,timeout=0.)
        if msg is not None:
            stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
            if stamp!=self._stamp:
                p,q=msg.pose.position,msg.pose.orientation
                self.T_world_pelvis[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
                self.T_world_pelvis[:3,3]=[p.x,p.y,p.z]
                self._received=time.monotonic()
                self._stamp=stamp
        T=self.T_world_pelvis
        return float(T[0,3]),float(T[1,3]),math.atan2(T[1,0],T[0,0])


class Navigator:
    def __init__(self, mission, start):
        self.mission = mission
        self.localizer = ContinuousPose(mission)
        self._last_log = 0
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            self.localizer.update()
            if not self.localizer.stale:
                break
            time.sleep(.05)
        if self.localizer.stale:
            raise RuntimeError('Localization unavailable: '+self.localizer.last_error)

    def pose(self):
        result=self.localizer.update()
        if self.localizer.stale:
            self.stop()
            raise RuntimeError('Stale localization: '+self.localizer.last_error)
        return result

    def stop(self):
        self.mission.stop()

    def camera_to_world(self, points, frame):
        self.pose()
        t = self.mission._tf().lookup_transform('pelvis', frame, Time()).transform
        q, p = t.rotation, t.translation
        T = np.eye(4)
        T[:3,:3] = Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        T[:3,3] = [p.x,p.y,p.z]
        T = self.localizer.T_world_pelvis @ T
        return np.asarray(points) @ T[:3,:3].T + T[:3,3]

    def face(self, yaw, timeout=15):
        if not math.isfinite(yaw):
            self.stop()
            raise ValueError('Nonfinite heading rejected')
        deadline = time.monotonic()+timeout
        try:
            while time.monotonic()<deadline:
                x,y,a = self.pose()
                err = wrap(yaw-a)
                if abs(err)<.07:
                    return True
                self.mission.cmd_vel(0,0,float(np.clip(1.2*err,-.45,.45)))
                time.sleep(.06)
            return False
        finally:
            self.stop()

    def goto(self, x, y, tolerance=.20, timeout=100, speed=.38):
        if not all(math.isfinite(v) for v in (x,y,tolerance,timeout,speed)):
            self.stop()
            raise ValueError('Nonfinite navigation target rejected')
        deadline = time.monotonic()+timeout
        stable = 0
        falls=self.mission.gap_falls
        try:
            while time.monotonic()<deadline:
                if self.mission.gap_falls!=falls:
                    return False
                px,py,yaw = self.pose()
                dx,dy = x-px,y-py
                distance = math.hypot(dx,dy)
                if distance<tolerance:
                    stable += 1
                    self.stop()
                    if stable>=4:
                        return True
                else:
                    stable=0
                    heading=wrap(math.atan2(dy,dx)-yaw)
                    # Turn in place before translation; use body-frame final correction.
                    if abs(heading)>.55 and distance>.45:
                        vx=vy=0.
                        wz=float(np.clip(1.2*heading,-.5,.5))
                    else:
                        vx=float(np.clip(.9*(math.cos(yaw)*dx+math.sin(yaw)*dy),-.16,speed))
                        vy=float(np.clip(.9*(-math.sin(yaw)*dx+math.cos(yaw)*dy),-.20,.20))
                        wz=float(np.clip(.8*heading,-.35,.35)) if distance>.45 else 0.
                    self.mission.cmd_vel(vx,vy,wz)
                now=time.monotonic()
                if now-self._last_log>5:
                    self.mission.log(f'[nav] pose=({px:.2f},{py:.2f},{yaw:.2f}) target=({x:.2f},{y:.2f}) error={distance:.2f}')
                    self._last_log=now
                time.sleep(.06)
            self.mission.log(f'[nav] timeout target=({x},{y})')
            return False
        finally:
            self.stop()

    def route_to(self, x, y, tolerance=.2, timeout=180, speed=.38):
        """Connect rooms through their public doors and the central corridor."""
        deadline=time.monotonic()+timeout
        px,py,_=self.pose()
        waypoints=[]
        if px>18:
            door=min([17.5,10.5,3.59],key=lambda v:abs(v-py))
            waypoints.extend([(19.,door),(16.,door)])
        elif px<13.0 and py<10.5:
            waypoints.extend([(12.,9.1),(14.,9.1)])
        elif py>21:
            waypoints.extend([(15.5,22.),(15.5,20.3)])
        if x>18:
            door=min([17.5,10.5,3.59],key=lambda v:abs(v-y))
            waypoints.extend([(16.,door),(19.,door)])
        elif x<13.0 and y<10.5:
            waypoints.extend([(14.,9.1),(12.,9.1)])
        elif y>21:
            waypoints.extend([(15.5,20.3),(15.5,22.)])
        waypoints.append((x,y))
        for gx,gy in waypoints:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not self.goto(gx,gy,tolerance=tolerance,timeout=remaining,speed=speed):
                return False
        return True
