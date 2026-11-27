"""Shared sensor-based navigation; coordinates describe the public building only."""
import math
import json
import time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ContinuousPose:
    def __init__(self,mission):
        self.mission=mission
        self._stamp=None
        self._received=0
        self.T_world_pelvis=np.eye(4)
        self.last_error='bringup localizer has not published a pose'
        self._status = None
        self._status_message = None
        self._status_received = 0.0
        self._source_age = 0.0

    @property
    def terminal(self):
        return bool(self._status and self._status.get('terminal'))

    @property
    def stale(self):
        now = time.monotonic()
        return (now - self._received > 1.2 or self._source_age > 1.0 or self._source_age < -.25
                or (self._status is not None and
                    (not self._status.get('valid') or now - self._status_received > 1.2)))

    def update(self):
        status = self.mission.latest('/custom/localization_status', String, timeout=0.)
        if status is not None and status is not self._status_message:
            self._status_message = status
            try:
                value = json.loads(status.data)
                if (not isinstance(value, dict) or value.get('version') != 1
                        or type(value.get('valid')) is not bool):
                    raise ValueError('invalid localization health message')
                self._status = value
                self._status_received = time.monotonic()
                self.last_error = value.get('reason') or 'waiting for a fresh sensor pose'
            except (ValueError, TypeError) as error:
                self._status = {'valid': False, 'terminal': True}
                self.last_error = str(error)
        msg=self.mission.latest('/custom/estimated_pose',PoseStamped,timeout=0.)
        if msg is not None:
            stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
            if self._stamp is not None and stamp < self._stamp:
                self.last_error = 'out-of-order estimated pose'
                return self._pose()
            if stamp!=self._stamp:
                p,q=msg.pose.position,msg.pose.orientation
                values = [p.x,p.y,p.z,q.x,q.y,q.z,q.w]
                if (msg.header.frame_id.lstrip('/') != 'map' or not np.isfinite(values).all()
                        or abs(np.linalg.norm(values[3:]) - 1.0) > .05):
                    self.last_error = 'invalid estimated pose frame, coordinates or quaternion'
                    self._received = 0.0
                    return self._pose()
                self.T_world_pelvis[:3,:3]=Rotation.from_quat(values[3:]).as_matrix()
                self.T_world_pelvis[:3,3]=[p.x,p.y,p.z]
                self._received=time.monotonic()
                self._stamp=stamp
        # Use the simulator sensor clock only when the new status protocol
        # identifies scan-stamped poses. Older localizer publishers remain
        # compatible through the monotonic receipt watchdog above.
        if self._status is not None and self._stamp is not None:
            clock = self.mission.latest('/clock', Clock, timeout=0.)
            if clock is not None:
                self._source_age = (clock.clock.sec - self._stamp[0]
                                   + (clock.clock.nanosec - self._stamp[1]) * 1e-9)
                if self._source_age > 1.0:
                    self.last_error = f'localization measurement is {self._source_age:.2f}s behind sensor clock'
                elif self._source_age < -.25:
                    self.last_error = 'localization timestamp is ahead of sensor clock'
        return self._pose()

    def _pose(self):
        T=self.T_world_pelvis
        return float(T[0,3]),float(T[1,3]),math.atan2(T[1,0],T[0,0])


class Navigator:
    def __init__(self, mission, start):
        self.mission = mission
        self.localizer = ContinuousPose(mission)
        self._last_log = 0
        self.last_failure = ''
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            self.localizer.update()
            if not self.localizer.stale:
                break
            time.sleep(.05)
        if self.localizer.stale:
            self.stop()
            raise RuntimeError('Localization unavailable: '+self.localizer.last_error)

    def pose(self):
        result=self.localizer.update()
        if self.localizer.stale:
            self.stop()
            raise RuntimeError('Stale localization: '+self.localizer.last_error)
        return result

    def stop(self):
        self.mission.stop()

    def _observed_pose(self, deadline):
        """Stop on invalid estimates; wait a bounded time for sensor recovery."""
        until = min(deadline, time.monotonic() + .8)
        while True:
            try:
                return self.pose()
            except RuntimeError:
                self.stop()
                if self.localizer.terminal or time.monotonic() >= until:
                    raise
                time.sleep(.04)

    def camera_to_world(self, points, frame):
        try:
            self.pose()
            stamp = self.localizer._stamp
            at = (Time(seconds=stamp[0], nanoseconds=stamp[1])
                  if self.localizer._status is not None else Time())
            t = self.mission._tf().lookup_transform(
                'pelvis', frame, at, timeout=Duration(seconds=.05)).transform
            q, p = t.rotation, t.translation
            T = np.eye(4)
            T[:3,:3] = Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
            T[:3,3] = [p.x,p.y,p.z]
            T = self.localizer.T_world_pelvis @ T
            return np.asarray(points) @ T[:3,:3].T + T[:3,3]
        except Exception:
            self.stop()
            raise

    def face(self, yaw, timeout=15):
        if not all(math.isfinite(v) for v in (yaw, timeout)) or timeout <= 0:
            self.stop()
            raise ValueError('Nonfinite heading rejected')
        deadline = time.monotonic()+timeout
        try:
            while time.monotonic()<deadline:
                x,y,a = self._observed_pose(deadline)
                err = wrap(yaw-a)
                if abs(err)<.07:
                    return True
                self.mission.cmd_vel(0,0,float(np.clip(1.2*err,-.45,.45)))
                time.sleep(.06)
            return False
        finally:
            self.stop()

    def goto(self, x, y, tolerance=.20, timeout=100, speed=.38):
        if (not all(math.isfinite(v) for v in (x,y,tolerance,timeout,speed))
                or min(tolerance, timeout, speed) <= 0):
            self.stop()
            raise ValueError('Nonfinite navigation target rejected')
        deadline = time.monotonic()+timeout
        stable = 0
        stable_stamp = None
        history = deque()
        recovery_attempts = 0
        self.last_failure = ''
        falls=self.mission.gap_falls
        try:
            while time.monotonic()<deadline:
                if self.mission.gap_falls!=falls:
                    return False
                px,py,yaw = self._observed_pose(deadline)
                dx,dy = x-px,y-py
                distance = math.hypot(dx,dy)
                if distance<tolerance:
                    if self.localizer._stamp != stable_stamp:
                        stable += 1
                        stable_stamp = self.localizer._stamp
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
                history.append((now, distance, abs(wrap(math.atan2(dy,dx) - yaw))))
                while len(history) > 1 and history[1][0] < now - 8.0:
                    history.popleft()
                if (distance >= tolerance and now - history[0][0] >= 8.0
                        and history[0][1] - distance < .08
                        and history[0][2] - history[-1][2] < .08):
                    self.stop()
                    if recovery_attempts >= 2:
                        self.mission.log('[nav] stalled after two sensor reobservations')
                        self.last_failure = 'stalled'
                        return False
                    recovery_attempts += 1
                    self.mission.log(f'[nav] stalled; sensor reobservation {recovery_attempts}/2')
                    previous = self.localizer._stamp
                    until = min(deadline, time.monotonic() + .8)
                    while time.monotonic() < until:
                        self._observed_pose(until)
                        if self.localizer._stamp != previous:
                            break
                        time.sleep(.04)
                    history.clear()
                if now-self._last_log>5:
                    self.mission.log(f'[nav] pose=({px:.2f},{py:.2f},{yaw:.2f}) target=({x:.2f},{y:.2f}) error={distance:.2f}')
                    self._last_log=now
                time.sleep(.06)
            self.mission.log(f'[nav] timeout target=({x},{y})')
            self.last_failure = 'timeout'
            return False
        finally:
            self.stop()

    @staticmethod
    def _route_points(px, py, x, y):
        """Public door topology; never assumes a waypoint has been reached."""
        doors = (17.5, 10.5, 3.59)
        def region(a, b):
            if a > 18.:
                return ('east', min(doors, key=lambda door: abs(door - b)))
            if a < 13. and b < 10.5:
                return ('soccer', 9.1)
            if b > 21.:
                return ('cones', 21.)
            return ('corridor', 0.)
        if region(px, py) == region(x, y):
            return [(x, y)]
        waypoints=[]
        if px>18:
            door=min(doors,key=lambda v:abs(v-py))
            waypoints.extend([(19.,door),(16.,door)])
        elif px<13.0 and py<10.5:
            waypoints.extend([(12.,9.1),(14.,9.1)])
        elif py>21:
            waypoints.extend([(15.5,22.),(15.5,20.3)])
        if x>18:
            door=min(doors,key=lambda v:abs(v-y))
            waypoints.extend([(16.,door),(19.,door)])
        elif x<13.0 and y<10.5:
            waypoints.extend([(14.,9.1),(12.,9.1)])
        elif y>21:
            waypoints.extend([(15.5,20.3),(15.5,22.)])
        waypoints.append((x,y))
        return [p for i, p in enumerate(waypoints) if i == 0 or p != waypoints[i - 1]]

    def route_to(self, x, y, tolerance=.2, timeout=180, speed=.38):
        """Connect rooms through public doors, with one bounded stall replan."""
        if (not all(math.isfinite(v) for v in (x, y, tolerance, timeout, speed))
                or min(tolerance, timeout, speed) <= 0):
            self.stop()
            raise ValueError('Invalid navigation target or limits')
        deadline = time.monotonic() + timeout
        try:
            for attempt in range(2):
                px, py, _ = self._observed_pose(deadline)
                for gx, gy in self._route_points(px, py, x, y):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    if not self.goto(gx, gy, tolerance=tolerance, timeout=remaining, speed=speed):
                        if self.last_failure != 'stalled' or attempt == 1:
                            return False
                        self.mission.log('[nav] replanning once from fresh sensor pose after stall')
                        break
                else:
                    return True
            return False
        finally:
            self.stop()
