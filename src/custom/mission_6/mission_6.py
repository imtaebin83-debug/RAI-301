#!/usr/bin/env python3
"""Mission 6: find the soccer ball with RGB-D and dribble it into the goal.

The head camera cannot see the floor within ~1.9 m of the pelvis, so the
controller never relies on seeing the ball while touching it. Each cycle:
observe the ball from a stand-off point behind it (on the goal line of
action), re-centre, walk straight through the ball toward the goal, stop and
re-observe. Only the published pitch/goal geometry and live sensor
observations are used; the referee alone decides success.
"""

from __future__ import annotations

import sys
import math
import time
from pathlib import Path
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from custom.sensor_mission import Mission
from mission_api import MissionResult, run_standalone  # noqa: E402
from custom.navigation import Navigator, wrap  # noqa: E402
from custom.vision import RGBD, BALL_RADIUS  # noqa: E402

# Public pitch geometry (README coordinate table).
GOAL_CENTRE = np.array([9.25, 1.4])
GOAL_LINE_Y = 2.13
POST_X = (6.82, 11.68)
ENTRY = (12.0, 9.1)
ROBOT_X = (6.15, 12.35)     # pelvis >= 0.65 m from the side walls (contact tilts the head camera)
ROBOT_Y_MAX = 9.8           # >= 0.7 m from the north wall
ROBOT_Y_FIELD = 2.55        # pelvis stays outside the goal unless between posts
ROBOT_Y_MOUTH = 1.75
VIEW_RANGE = 2.3            # preferred stand-off: whole ball in view
MIN_VIEW = 1.6              # nearest stand-off (near walls)
BLIND_STANDOFF = 2.0        # closer than this the ball is below the image even while stepping
ALIGN_TOLERANCE = .05       # lateral ball offset accepted before dribbling (m)
PUSH_THROUGH = 1.1          # walk this far past the observed ball centre
DRIBBLE_SEGMENT = 1.2       # walk this far past the ball centre, then re-observe
PUSH_SPEED = .5             # goal-ward kick: the ball rolls ~4 m, clear of the blind zone
NUDGE_SPEED = .25           # off-wall push: a full kick rolled the ball into a side wall
NUDGE_THROUGH = .6
FINISH_RANGE = 3.0          # ball this close to the goal line: dribble all the way
BUDGET = 300.
TURN_RATE = .25            # rad/s cap for in-place turns (localizer-friendly)
CREEP_VX = -.05            # backward creep keeps the torso leaned forward (forward creep does not)


def in_goal(xy) -> bool:
    return (POST_X[0]+BALL_RADIUS < xy[0] < POST_X[1]-BALL_RADIUS
            and .66 < xy[1] < GOAL_LINE_Y-BALL_RADIUS-.06)


def unit(v):
    n = float(np.linalg.norm(v))
    return v/n if n > 1e-6 else np.array([0., -1.])


def clamp_robot(p):
    return np.array([float(np.clip(p[0], *ROBOT_X)), float(np.clip(p[1], ROBOT_Y_FIELD, ROBOT_Y_MAX))])


class Mission6(Mission):
    MISSION = 6

    def __init__(self) -> None:
        super().__init__()

    def _check_run(self):
        if self.respawn_count != self._run_respawn_count:
            self.stop()
            raise RuntimeError('Public respawn invalidated ball observations; restart required')
        if time.monotonic() >= self.deadline:
            self.stop()
            raise TimeoutError('Soccer mission deadline reached')

    def _on_public_respawn(self, message):
        super()._on_public_respawn(message)
        self.stop()

    def cmd_vel(self, vx=0., vy=0., wz=0.):
        with self._sensor_cv:
            if vx or vy or wz:
                self._check_run()
            super().cmd_vel(vx, vy, wz)

    # ------------------------------------------------------------ locomotion
    def _pose(self, wait=4.):
        """Pose that tolerates short localizer publishing gaps (0.4-1.2 s were
        measured): stop and wait instead of aborting the manoeuvre."""
        self._check_run()
        try:
            return self.nav.pose()
        except RuntimeError as exc:
            if 'Stale localization' not in str(exc):
                raise
            self.stats['localization_waits'] += 1
            self.nav.stop()
            until = min(self.deadline, time.monotonic()+wait)
            while time.monotonic() < until:
                self._check_run()
                time.sleep(.1)
                self.nav.localizer.update()
                if not self.nav.localizer.stale:
                    return self.nav.pose()
            raise

    def _face(self, yaw, timeout=12.):
        """Slow in-place turn. Fast turns produced rejected LiDAR registrations
        ("sensor reanchor") in the shared localizer, losing up to ~1 m."""
        if not math.isfinite(yaw):
            raise ValueError('Nonfinite heading rejected')
        deadline = min(self.deadline, time.monotonic()+timeout)
        try:
            while time.monotonic() < deadline:
                err = wrap(yaw-self._pose()[2])
                if abs(err) < .06:
                    return True
                self.cmd_vel(0., 0., float(np.clip(1.0*err, -TURN_RATE, TURN_RATE)))
                time.sleep(.05)
            return False
        finally:
            self.nav.stop()

    def _goto(self, x, y, tolerance=.2, timeout=40., speed=.38):
        """Use the tested shared vector controller, not the old clipping copy."""
        self._check_run()
        return self.nav.goto(x, y, tolerance=tolerance,
                             timeout=min(timeout, self.deadline-time.monotonic()), speed=speed)

    # ------------------------------------------------------------------ vision
    def _observe(self, seconds=3., prior=None, gate=None, min_frames=2, settle=.6, enough=4):
        """Median of consistent ball detections over a bounded window.

        Frames taken while the body is still turning are skipped: the image
        and the latest pose estimate are not time-matched, so heading motion
        projects a distant ball to the wrong world position.

        The robot creeps backwards meanwhile: a G1 standing still straightens
        its torso and the camera's lowest image row moves from ~1.8 m to
        3-4.5 m ahead, hiding a ball at the 2.3 m stand-off distance. While
        stepping, the torso stays leaned forward.
        """
        heading = self._pose()[2]
        creep_vx = CREEP_VX

        def creep():
            herr = wrap(heading-self._pose()[2])
            self.cmd_vel(creep_vx, 0., float(np.clip(1.0*herr, -.15, .15)))

        settle_until = min(self.deadline, time.monotonic()+settle)
        while time.monotonic() < settle_until:
            creep()
            time.sleep(.05)
        deadline = min(self.deadline, time.monotonic()+seconds)
        points, rel, last_stamp = [], [], None
        previous_yaw = self._pose()[2]
        frames = raw = 0
        while time.monotonic() < deadline:
            creep()
            det = self.sensor.ball(prior=prior, gate=gate)
            px, py, yaw = self._pose()
            steady = abs(wrap(yaw-previous_yaw)) < .05
            previous_yaw = yaw
            if self.sensor._stamp != last_stamp:
                frames += 1
                raw += det is not None
                if det is not None and steady:
                    points.append((det.x, det.y))
                    # Ball in the pelvis frame: world projection and its inverse
                    # use the same pose estimate, so absolute pose error cancels.
                    dx, dy = det.x-px, det.y-py
                    rel.append((math.cos(yaw)*dx+math.sin(yaw)*dy, -math.sin(yaw)*dx+math.cos(yaw)*dy))
                last_stamp = self.sensor._stamp
                if len(points) >= enough:
                    break
            time.sleep(.05)
        self.nav.stop()
        if raw:
            self.log(f'[mission 6] observe: frames={frames} detections={raw} steady={len(points)}')
        if len(points) < min_frames:
            return None
        pts = np.asarray(points)
        centre = np.median(pts, axis=0)
        keep = np.linalg.norm(pts-centre, axis=1) < .25
        near = pts[keep]
        self.last_rel = np.median(np.asarray(rel)[keep], axis=0) if keep.any() else None
        if len(near) < min_frames:
            self.log(f'[mission 6] inconsistent ball frames n={len(pts)} spread={np.ptp(pts, axis=0).round(2).tolist()}')
            return None
        self.stats['observed_frames'] += len(near)
        return np.median(near, axis=0)

    def _scan(self, headings, prior=None, gate=None):
        for yaw in headings:
            if self._expired():
                return None
            self._face(yaw, timeout=8.)
            ball = self._observe(3., prior, gate)
            if ball is not None:
                return ball
        return None

    def _acquire(self):
        """First detection: scan from the entry, then from public vantage points."""
        x, y, yaw = self._pose()
        ball = self._observe(3.)           # look first without turning
        if ball is not None:
            return ball
        # Centre on the public random-ball area, then sweep in half-FOV steps.
        centre = math.atan2(7.0-y, 9.25-x)
        offsets = (0., -.45, .45, -.9, .9, -1.35, 1.35, math.pi)
        ball = self._scan([centre+o for o in offsets])
        if ball is not None:
            return ball
        for vx, vy in ((12.2, 4.2), (9.25, 3.3), (6.5, 4.2), (6.5, 9.4), (12.1, 9.4)):
            if self._expired():
                return None
            self.log(f'[mission 6] search vantage ({vx},{vy})')
            if not self._goto(vx, vy, tolerance=.3, timeout=35.):
                continue
            centre = math.atan2(7.0-vy, 9.25-vx)
            ball = self._scan([centre+o for o in (0., -.45, .45, -.9, .9, -1.35, 1.35)])
            if ball is not None:
                return ball
        return None

    # ---------------------------------------------------------------- planning
    def _plan(self, ball):
        """Push direction and stand-off point behind the ball.

        Every push rolls the ball ~4 m (low rolling friction), so the goal
        direction is kept whenever possible: near a wall the stand-off
        distance shrinks first (down to MIN_VIEW, where the top of the ball
        is still visible while stepping). Only then is the direction rotated,
        toward the pitch centre. Returns (direction, stand, toward_goal).
        """
        goal_dir = unit(GOAL_CENTRE-ball)
        base = math.atan2(goal_dir[1], goal_dir[0])
        toward_centre = 1 if np.cross(goal_dir, np.array([9.25, 5.25])-ball) > 0 else -1
        for k in (0, toward_centre, -toward_centre, 2*toward_centre, -2*toward_centre,
                  3*toward_centre, -3*toward_centre, 4*toward_centre, -4*toward_centre):
            angle = base+math.radians(15*k)
            direction = np.array([math.cos(angle), math.sin(angle)])
            for distance in (VIEW_RANGE, 2.0, 1.8, MIN_VIEW):
                stand = ball-direction*distance
                if np.allclose(clamp_robot(stand), stand, atol=1e-6):
                    return direction, stand, k == 0
        return goal_dir, clamp_robot(ball-goal_dir*VIEW_RANGE), True

    def _go_behind(self, ball, direction, stand):
        """Walk to the stand-off point without crossing the ball."""
        x, y, _ = self._pose()
        me = np.array([x, y])
        ahead = float(np.dot(me-ball, direction))   # > 0: on the goal side of the ball
        lateral = abs(float(np.cross(direction, me-ball)))
        if ahead > 0 or (ahead > -.6 and lateral < 1.):
            side = np.array([-direction[1], direction[0]])
            options = [ball+s*side*1.1-direction*.4 for s in (1, -1)]
            options = [p for p in options if np.allclose(clamp_robot(p), p, atol=1e-6)] or \
                [clamp_robot(p) for p in options]
            detour = min(options, key=lambda p: np.linalg.norm(p-me))
            self.log(f'[mission 6] detour around ball via ({detour[0]:.2f},{detour[1]:.2f})')
            self._goto(*detour, tolerance=.25, timeout=30.)
        return self._goto(*stand, tolerance=.10, timeout=40.)

    # ----------------------------------------------------------------- control
    def _push(self, direction, length, speed=None):
        """Walk along the line through the observed ball centre (body-frame
        cmd_vel) for ``length`` m from the current position.

        The ball sits in the camera blind zone during contact, so this loop is
        pure line following on the pose estimate. Returns True when stopped by
        the goal-mouth limit (the ball, if still in front, is at the goal).
        """
        start = np.array(self._pose()[:2])
        heading = math.atan2(direction[1], direction[0])
        deadline = min(self.deadline, time.monotonic()+length/.18+8.)
        at_goal = False
        try:
            while time.monotonic() < deadline:
                x, y, yaw = self._pose()
                p = np.array([x, y])
                if float(np.dot(p-start, direction)) >= length:
                    break
                mouth = POST_X[0]+.45 < x < POST_X[1]-.45
                if y < (ROBOT_Y_MOUTH if mouth else ROBOT_Y_FIELD) or not (ROBOT_X[0] < x < ROBOT_X[1]):
                    at_goal = mouth or y < ROBOT_Y_FIELD+.05
                    self.log(f'[mission 6] push stopped at pitch/goal limit ({x:.2f},{y:.2f})')
                    break
                cross = float(np.cross(direction, p-start))  # ball is on this line after alignment
                herr = wrap(heading-yaw)
                vx = (speed or PUSH_SPEED) if abs(herr) < .25 else .1
                # cross > 0: pelvis left of the line; body +y is left while
                # the heading follows the line, so steer right.
                vy = float(np.clip(-1.5*cross, -.2, .2))
                self.cmd_vel(vx, vy, float(np.clip(1.5*herr, -.4, .4)))
                time.sleep(.05)
        finally:
            self.nav.stop()
        self.stats['pushes'] += 1
        return at_goal

    def _back_up(self, distance, seconds=16.):
        """Walk backwards holding the heading (open-loop backing drifted 60 deg)."""
        x0, y0, heading = self._pose()
        start = np.array([x0, y0])
        until = min(self.deadline, time.monotonic()+seconds)
        try:
            while time.monotonic() < until:
                x, y, yaw = self._pose()
                if np.linalg.norm(np.array([x, y])-start) >= distance:
                    break
                self.cmd_vel(-.16, 0., float(np.clip(1.2*wrap(heading-yaw), -.25, .25)))
                time.sleep(.05)
        finally:
            self.nav.stop()

    def _after_push(self, ball, direction, at_goal):
        """Find the ball again after a blind dribble.

        After contact the ball usually rests 0.5-0.8 m in front of the pelvis,
        i.e. inside the blind zone, so back away before looking.
        """
        guess = ball+direction*1.5
        fresh = self._observe(3., prior=guess, gate=4.5, settle=.6)
        if fresh is None:
            # The ball can deflect sideways or behind off a leg: look around
            # here (it is >2 m away unless it rests at the feet) before backing.
            yaw = self._pose()[2]
            fresh = self._scan([yaw+.9, yaw-.9, yaw+1.8, yaw-1.8, yaw+math.pi], prior=guess, gate=5.)
        if fresh is None:
            self._face(math.atan2(direction[1], direction[0]), timeout=10.)
            self._back_up(2.5, seconds=22.)
            here = np.array(self._pose()[:2])
            look = math.atan2(*(guess-here)[::-1])
            fresh = self._scan([look, look-.45, look+.45, look-.9, look+.9, look-1.35, look+1.35],
                               prior=guess, gate=5.)
        if fresh is None and at_goal:
            self.stats['unseen_after_goal_push'] += 1
            return None
        if fresh is None:
            self.stats['reacquisitions'] += 1
            fresh = self._acquire()
        return fresh

    def _expired(self):
        return time.monotonic() > self.deadline

    def _align(self, ball, direction):
        """Strafe until the ball is straight ahead in the pelvis frame.

        Uses only the camera's ball offset relative to the body, so the
        (sometimes 0.3 m) absolute pose error does not matter. Returns the
        refreshed world estimate and the distance ahead, or (None, None).
        """
        heading = math.atan2(direction[1], direction[0])
        for attempt in range(6):
            fresh = self._observe(3., prior=ball, gate=1.0, settle=.4)
            if fresh is None or self.last_rel is None:
                return None, None
            ball = fresh
            ahead, lateral = (float(v) for v in self.last_rel)
            self.log(f'[mission 6] align {attempt}: ball {ahead:.2f} m ahead, {lateral:+.2f} m left')
            if abs(lateral) < ALIGN_TOLERANCE:
                return ball, ahead
            # The walking policy barely responds below ~0.1 m/s.
            vy = math.copysign(min(.2, max(.12, abs(lateral))), lateral)
            vx = 0.
            until = min(self.deadline, time.monotonic()+min(4., max(.5, 1.2*abs(lateral)/abs(vy))))
            try:
                while time.monotonic() < until:
                    herr = wrap(heading-self._pose()[2])
                    self.cmd_vel(vx, vy if abs(lateral) >= ALIGN_TOLERANCE else 0., float(np.clip(1.0*herr, -.2, .2)))
                    time.sleep(.05)
            finally:
                self.nav.stop()
            self._face(heading, timeout=6.)
        return ball, max(float(self.last_rel[0]), 0.) if self.last_rel is not None else None

    def _viewpoint(self, ball, direction):
        """A pitch point VIEW_RANGE from the ball, as close to 'behind' as walls allow."""
        behind = math.atan2(-direction[1], -direction[0])
        for k in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
            angle = behind+math.radians(15*k)
            p = ball+VIEW_RANGE*np.array([math.cos(angle), math.sin(angle)])
            if np.allclose(clamp_robot(p), p, atol=1e-6):
                return p
        return None

    def _cycle(self, ball):
        """Stand off, align on the camera, dribble; returns the new estimate or None."""
        direction, stand, toward_goal = self._plan(ball)
        self.log(f'[mission 6] cycle {self.stats["cycles"]}: ball=({ball[0]:.2f},{ball[1]:.2f}) '
                 f'stand=({stand[0]:.2f},{stand[1]:.2f}) aim={math.degrees(math.atan2(direction[1], direction[0])):.0f}deg '
                 f'{"goal" if toward_goal else "off-wall"}')
        if np.linalg.norm(stand-ball) < BLIND_STANDOFF:
            return self._wall_kick(ball, direction, stand)
        if not self._go_behind(ball, direction, stand):
            self.log('[mission 6] stand-off point not reached')
        here = np.array(self._pose()[:2])
        self._face(math.atan2(direction[1], direction[0]), timeout=10.)
        fresh, ahead = self._align(ball, direction)
        if fresh is None:
            here = np.array(self._pose()[:2])
            look = math.atan2(*(ball-here)[::-1])
            fresh = self._scan([look, look-.45, look+.45], prior=ball, gate=2.)
            if fresh is None:
                self.stats['reacquisitions'] += 1
                self.log('[mission 6] ball not visible from the stand-off point; full reacquisition')
                return self._acquire()
            return fresh            # re-plan from the new estimate
        ball = fresh
        to_line = float(ball[1]-GOAL_LINE_Y)
        if toward_goal:
            extra, speed = (12. if to_line < FINISH_RANGE else DRIBBLE_SEGMENT), PUSH_SPEED
        else:
            extra, speed = NUDGE_THROUGH, NUDGE_SPEED      # free it from the wall gently
        self.log(f'[mission 6] {"kick" if toward_goal else "nudge"} {ahead+extra:.1f} m at {speed:.2f} m/s, '
                 f'ball ({ball[0]:.2f},{ball[1]:.2f}) {ahead:.2f} m ahead')
        return self._after_push(ball, direction, self._push(direction, ahead+extra, speed))

    def _wall_kick(self, ball, direction, stand):
        """Ball near a wall: the stand-off point is inside the camera blind
        zone. Observe from a diagonal view point, then walk the short way to
        the stand-off point and kick on that fresh observation."""
        self.stats['blind_pushes'] += 1
        view = self._viewpoint(ball, direction)
        if view is not None:
            self.log(f'[mission 6] wall case: view from ({view[0]:.2f},{view[1]:.2f})')
            self._go_behind(ball, direction, view)
            here = np.array(self._pose()[:2])
            self._face(math.atan2(*(ball-here)[::-1]), timeout=10.)
            fresh = self._observe(3., prior=ball, gate=1.2)
            if fresh is None:
                self.stats['reacquisitions'] += 1
                return self._acquire()
            ball = fresh
            direction, stand, _ = self._plan(ball)
        self._go_behind(ball, direction, stand)
        self._face(math.atan2(direction[1], direction[0]), timeout=10.)
        here = np.array(self._pose()[:2])
        cross = float(np.cross(direction, here-ball))
        if abs(cross) > .06:      # put the pelvis back on the ball line
            self._goto(*(ball-direction*float(np.dot(ball-here, direction))), tolerance=.06, timeout=10.)
            self._face(math.atan2(direction[1], direction[0]), timeout=8.)
        here = np.array(self._pose()[:2])
        ahead = max(float(np.dot(ball-here, direction)), 0.)
        self.log(f'[mission 6] wall kick {ahead+DRIBBLE_SEGMENT:.1f} m, ball ({ball[0]:.2f},{ball[1]:.2f}) {ahead:.2f} m ahead (blind)')
        return self._after_push(ball, direction, self._push(direction, ahead+DRIBBLE_SEGMENT))

    def _wait_for_localization(self, seconds=8.):
        self.nav.stop()
        until = min(self.deadline, time.monotonic()+seconds)
        while time.monotonic() < until:
            self.nav.localizer.update()
            if not self.nav.localizer.stale:
                return True
            time.sleep(.1)
        return False

    # --------------------------------------------------------------------- run
    def run(self) -> MissionResult:
        self._run_respawn_count = self.respawn_count
        self.deadline = time.monotonic()+BUDGET
        self.last_rel = None
        self.stats = {'observed_frames': 0, 'pushes': 0, 'blind_pushes': 0, 'reacquisitions': 0,
                      'cycles': 0, 'localization_waits': 0, 'unseen_after_goal_push': 0}
        ball = None
        try:
            self.begin_mission(6, timeout=15.)
            self._check_run()
            self.nav = Navigator(self, None)
            if not self.nav.route_to(*ENTRY, timeout=120.):
                return self.result(error='Could not enter soccer field', **self.stats)
            self.sensor = RGBD(self, self.nav)
            while not self._expired():
                if ball is not None and in_goal(ball):
                    self.log(f'[mission 6] ball observed inside goal ({ball[0]:.2f},{ball[1]:.2f})')
                    return self.result(ball_xy=np.round(ball, 3).tolist(), **self.stats)
                try:
                    if ball is None:
                        fresh = self._acquire()
                        if fresh is None:
                            return self.result(error='Ball not found from any vantage point', **self.stats)
                        self.log(f'[mission 6] acquired ball ({fresh[0]:.2f},{fresh[1]:.2f})')
                        ball = fresh
                        continue
                    self.stats['cycles'] += 1
                    fresh = self._cycle(ball)
                except RuntimeError as exc:
                    if 'Stale localization' not in str(exc):
                        raise
                    self.stats['localization_waits'] += 1
                    self.log(f'[mission 6] {exc}; waiting for the localizer')
                    if not self._wait_for_localization():
                        return self.result(error=str(exc), **self.stats)
                    continue
                if fresh is None:
                    if self.stats['unseen_after_goal_push']:
                        self.log('[mission 6] ball not visible after a dribble into the goal mouth; ending')
                        return self.result(note='ball not re-observed after goal-mouth dribble', **self.stats)
                    return self.result(error='Ball lost', **self.stats)
                ball = fresh
            return self.result(error='Time budget exhausted; referee decides success',
                               ball_xy=None if ball is None else np.round(ball, 3).tolist(), **self.stats)
        except Exception as exc:
            return self.result(error=f'{type(exc).__name__}: {exc}',
                               restart_required=self.respawn_count != self._run_respawn_count,
                               **getattr(self, 'stats', {}))
        finally:
            self.stop()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission6))
