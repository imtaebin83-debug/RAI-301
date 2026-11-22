"""Task-parameterized keypose policy learned from measured successful episodes.

The inductive bias is an object-relative SE(3) trajectory with a finite-state
servo. Phase order, endpoint poses, gripper actions and phase durations come
from the training split. This is not the raw-image CNN and never calls the
demonstration teacher's planner or step function.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .perception import CameraModel, measure_scene

FORMAT = "m5-task-parameterized-keyposes-v1"


def yaw_rotation(yaw):
    return Rotation.from_euler("z", float(yaw)).as_matrix()


def average_pencil_yaw(yaws):
    """Align an undirected PCA axis to its first observed branch before mean.

    Without a visible eraser, PCA may return either theta or theta+pi. Keep
    the first grasp branch, rather than cancelling opposite vectors or changing
    every existing grasp to a newly chosen global branch.
    """
    values = np.asarray(yaws, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid pencil axis measurements')
    axis_confidence = abs(np.exp(2j*values).mean())
    if axis_confidence < .95:
        raise ValueError('Pencil axis is not temporally stable')
    aligned = values + np.where(np.cos(values-values[0]) < 0, np.pi, 0.)
    return math.atan2(np.sin(aligned).mean(), np.cos(aligned).mean())


def materialize(phase, measurement):
    """A learned pose in an observed object's yaw-canonical frame -> base pose."""
    ref = np.asarray(measurement[phase["reference"] + "_xy"], dtype=float)
    rotation = yaw_rotation(measurement["pencil_yaw"])
    target = np.eye(4)
    target[:3, 3] = np.r_[ref, 0.] + rotation @ np.asarray(phase["offset_xyz"], float)
    target[:3, :3] = rotation @ Rotation.from_quat(phase["quaternion_xyzw"]).as_matrix()
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite learned keypose")
    return target


def servo_action(state, target, grip, controller):
    state = np.asarray(state, dtype=float)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 8-D robot proprioception")
    error = target[:3, 3] - state[:3]
    angular = Rotation.from_matrix(target[:3, :3] @ Rotation.from_rotvec(state[3:6]).as_matrix().T).as_rotvec()
    translation = error * controller["position_gain"]
    translation *= min(1., controller["max_position_step_m"] / max(np.linalg.norm(translation), 1e-12))
    turn = angular * controller["orientation_gain"]
    turn *= min(1., controller["max_rotation_step_rad"] / max(np.linalg.norm(turn), 1e-12))
    action = np.r_[translation / .05, turn / .5, -1. if grip < 0 else 1.]
    return np.clip(action, -1., 1.).astype(np.float32), float(np.linalg.norm(error)), float(np.linalg.norm(angular))


def validate_model(model):
    if model.get("format") != FORMAT or not model.get("phases"):
        raise ValueError("Not a learned task-parameterized keypose checkpoint")
    if not model.get("training", {}).get("episode_sha256"):
        raise ValueError("Checkpoint lacks training demonstration provenance")
    names = []
    for phase in model["phases"]:
        if phase.get("reference") not in ("pencil", "holder"):
            raise ValueError("Keypose reference must be a sensor-observed object")
        values = [*phase["offset_xyz"], *phase["quaternion_xyzw"], phase["grip"], phase["minimum_seconds"]]
        if not np.isfinite(values).all() or phase["minimum_seconds"] < 0:
            raise ValueError("Invalid learned keypose values")
        if len(phase["offset_xyz"]) != 3 or len(phase["quaternion_xyzw"]) != 4:
            raise ValueError("Invalid learned keypose shape")
        names.append(phase["name"])
    if len(set(names)) != len(names):
        raise ValueError("Duplicate learned phases")
    required = ("position_gain", "orientation_gain", "max_position_step_m", "max_rotation_step_rad",
                "position_tolerance_m", "orientation_tolerance_rad", "stage_timeout_scale", "stage_timeout_min")
    c = model["controller"]
    if any(not math.isfinite(float(c[k])) or float(c[k]) <= 0 for k in required):
        raise ValueError("Invalid finite servo configuration")
    return model


class Policy:
    policy_name = "task-parameterized-keypose-bc"
    def __init__(self, checkpoint, device="cpu", camera=None):
        if device != "cpu":
            raise ValueError("Keypose policy is CPU-only")
        self.model = validate_model(json.loads(Path(checkpoint).read_text()))
        self.fps = float(self.model["fps"])
        if not 1 <= self.fps <= 30:
            raise ValueError("Unsupported checkpoint fps")
        calibration = self.model["camera"]
        self.camera = camera or CameraModel.from_pose(calibration["k"], calibration["position"],
                                                      calibration["quaternion_wxyz"])
        self.controller = self.model["controller"]
        self.reset()

    def reset(self):
        self.done = False
        self.failed = None
        self.stage = "observe"
        self.debug = {"policy_kind": self.policy_name, "phase_source": "training_split_measured_robot_states"}
        self._measurements = []
        self._scene = None
        self._index = 0
        self._start = self._stage_started = None
        self._last_measure_time = None
        self._last_hit_time = None
        self._hits = 0
        self._grip = -1.
        self._grasp_anchor = None
        self._grasp_closed_at = None
        self._grasp_align_hits = 0
        self._last_align_time = None
        self._grasp_stable_since = None
        self._grasp_width = None

    def _closing_phase(self):
        """A learned open-to-closed transition near the preceding keypose."""
        if self._index == 0:
            return False
        previous, current = self.model['phases'][self._index-1:self._index+1]
        return previous['grip'] < 0 < current['grip']

    def _hold(self):
        return {"actions": np.asarray([[0, 0, 0, 0, 0, 0, self._grip]], dtype=np.float32)}

    def act(self, observation):
        if self.done or self.failed:
            return self._hold()
        now = float(observation.get("time", time.monotonic()))
        if not math.isfinite(now):
            raise ValueError("Invalid observation timestamp")
        if self._start is None:
            self._start = self._stage_started = now
        if self._scene is None:
            try:
                measurement = measure_scene(observation["image"], self.camera).as_dict()
                if now != self._last_measure_time:
                    self._measurements.append(measurement)
                    self._measurements = self._measurements[-4:]
                    self._last_measure_time = now
                if len(self._measurements) >= 3:
                    p = np.asarray([m["pencil_xy"] for m in self._measurements])
                    h = np.asarray([m["holder_xy"] for m in self._measurements])
                    yaw = np.asarray([m["pencil_yaw"] for m in self._measurements])
                    if max(p.std(0).max(), h.std(0).max()) < .008:
                        self._scene = {"pencil_xy": np.median(p, axis=0), "holder_xy": np.median(h, axis=0),
                                       "pencil_yaw": average_pencil_yaw(yaw)}
                        self.stage = self.model["phases"][0]["name"]
                        self._stage_started = now
            except (ValueError, np.linalg.LinAlgError) as error:
                self.debug["vision_error"] = str(error)
            if self._scene is None and now - self._start > 8:
                self.failed = "Initial sensor-only object localization timed out"
            return self._hold()
        phase = self.model["phases"][self._index]
        learned_target = materialize(phase, self._scene)
        target = learned_target
        self._grip = float(phase["grip"])
        closing = self._closing_phase()
        elapsed = now - self._stage_started
        timeout = max(self.controller["stage_timeout_min"],
                      phase["minimum_seconds"] * self.controller["stage_timeout_scale"])
        aligning = closing and self._grasp_anchor is None
        if aligning:
            # Reach the learned grasp endpoint while the hand is still open.
            # The preceding phase's 6-mm arrival tolerance must not silently
            # become a higher grasp height. Require two distinct observations.
            self._grip = float(self.model['phases'][self._index-1]['grip'])
            if elapsed >= min(5., timeout):
                self.failed = 'Learned-keypose open-hand alignment timed out: ' + self.stage
                self.debug.update(stage=self.stage, elapsed=elapsed, grasp_mode='align_open',
                                  contact_hold=False)
                return self._hold()
            _, alignment_distance, alignment_angular = servo_action(
                observation['state'], learned_target, self._grip, self.controller)
            aligned = (alignment_distance < min(.003, self.controller['position_tolerance_m'])
                       and alignment_angular < self.controller['orientation_tolerance_rad'])
            if now != self._last_align_time:
                self._grasp_align_hits = self._grasp_align_hits + 1 if aligned else 0
                self._last_align_time = now
            if self._grasp_align_hits >= 2:
                self._grasp_anchor = np.eye(4)
                self._grasp_anchor[:3, 3] = observation['state'][:3]
                self._grasp_anchor[:3, :3] = Rotation.from_rotvec(observation['state'][3:6]).as_matrix()
                self._grasp_closed_at = now
                self._hits = 0
                self._last_hit_time = None
                self._grip = float(phase['grip'])
                aligning = False
        if closing and not aligning:
            # Once squeezing starts, hold this measured endpoint instead of
            # chasing contact deflection while an object rests on the table.
            target = self._grasp_anchor
        action, distance, angular = servo_action(observation["state"], target, self._grip, self.controller)
        close_elapsed = (now-self._grasp_closed_at if closing and not aligning else 0.)
        self.debug.update(stage=self.stage, learned_target_xyz=learned_target[:3, 3].tolist(),
                          command_target_xyz=target[:3, 3].tolist(),
                          position_error_xyz=(target[:3,3]-observation['state'][:3]).tolist(),
                          position_error=distance, orientation_error=angular, elapsed=elapsed,
                          grasp_mode=('align_open' if aligning else 'contact_hold') if closing else None,
                          close_elapsed=close_elapsed, alignment_hits=self._grasp_align_hits)
        width = float(np.abs(np.asarray(observation['state'])[6:8]).sum())
        self.debug.update(finger_width=width, contact_hold=closing and not aligning)
        if elapsed > timeout:
            self.failed = "Learned-keypose stage timed out: " + self.stage
            return self._hold()
        if aligning:
            return {"actions": action[None]}
        hit = distance < self.controller["position_tolerance_m"] and angular < self.controller["orientation_tolerance_rad"]
        if closing:
            # Reject empty closure and a still-wide/open hand. A nonzero gap is
            # evidence from joint sensors, not an oracle that an object is held.
            grasped = .008 <= width <= .065
            stable_width = self._grasp_width is not None and abs(width-self._grasp_width) <= .001
            self._grasp_width = width
            if grasped and hit and stable_width:
                if self._grasp_stable_since is None:
                    self._grasp_stable_since = now
            else:
                self._grasp_stable_since = None
            hit = grasped and hit and self._grasp_stable_since is not None and now-self._grasp_stable_since >= .3
            if close_elapsed >= max(2., phase['minimum_seconds']) and width < .008:
                self.failed = 'Empty grasp: fingers closed without a supported object'
                return self._hold()
        if now != self._last_hit_time:
            self._hits = self._hits + 1 if hit else 0
            self._last_hit_time = now
        dwell = close_elapsed if closing else elapsed
        if self._hits >= 2 and dwell >= phase["minimum_seconds"]:
            self._index += 1
            self._hits = 0
            self._grasp_anchor = None
            self._grasp_closed_at = None
            self._grasp_align_hits = 0
            self._last_align_time = None
            self._grasp_stable_since = None
            self._grasp_width = None
            self._stage_started = now
            if self._index >= len(self.model["phases"]):
                self.done, self.stage = True, "done"
            else:
                self.stage = self.model["phases"][self._index]["name"]
        return {"actions": action[None]}
