"""Sensor-only geometric demonstration teacher (not the submitted learned policy).

Reads only RGB images and robot proprioception. Geometry is fixed object size,
camera calibration and table height, never simulator object poses or scenarios.
The collector must have the unmodified referee verify completed demonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


from .perception import CameraModel, SceneMeasurement, measure_scene


@dataclass
class TeacherConfig:
    table_z: float = 0.
    grasp_z: float = .018
    approach_z: float = .20
    lift_z: float = .42
    carry_z: float = .40
    insert_z: float = .265
    position_tolerance: float = .006
    orientation_tolerance: float = .07
    max_position_step: float = .015
    max_rotation_step: float = .10
    position_gain: float = .65
    orientation_gain: float = .65
    stage_timeout: float = 18.
    total_timeout: float = 130.
    verify_grasp: bool = True


class VisionTeacher:
    """Finite state sensor teacher emitting normalized Panda-base actions.

    stage/done/failed/debug are collector-facing. A done state is not success;
    only an external unmodified referee may label a collected episode successful.
    """
    def __init__(self, camera=None, config=None, clock=time.monotonic):
        self.camera = camera or CameraModel.panda_agentview_default()
        self.config = config or TeacherConfig()
        self.clock = clock
        self.stage = 'observe'
        self.done = False
        self.failed = None
        self.debug = {}
        self.initial = None
        self._started = self._stage_started = clock()
        self._plan = []
        self._index = -1
        self._grip = -1.
        self._measurements = []

    def reset(self):
        self.__init__(self.camera, self.config, self.clock)

    def _advance(self):
        self._index += 1
        self._stage_started = self.clock()
        if self._index >= len(self._plan):
            self.stage, self.done = 'done', True
        else:
            self.stage = self._plan[self._index][0]

    def _make_plan(self, m):
        cfg = self.config
        # Finger joints close along grip-site X (finger bodies have Rz(+90)).
        # Align the pencil with EEF Y, approach with EEF Z pointing downward.
        grasp = Rotation.from_euler('z', m.pencil_yaw + math.pi/2).as_matrix() @ Rotation.from_euler('x', math.pi).as_matrix()
        vertical = grasp @ Rotation.from_euler('x', math.pi/2).as_matrix()
        p, c, z = m.pencil_xy, m.holder_xy, cfg.table_z
        self._plan = [
            ('approach', np.r_[p, z+cfg.approach_z], grasp, -1., 0.),
            ('descend', np.r_[p, z+cfg.grasp_z], grasp, -1., 0.),
            ('close', np.r_[p, z+cfg.grasp_z], grasp, 1., 1.0),
            ('lift', np.r_[p, z+cfg.lift_z], grasp, 1., 0.),
            ('upright', np.r_[p, z+cfg.lift_z], vertical, 1., 0.),
            ('carry', np.r_[c, z+cfg.carry_z], vertical, 1., 0.),
            ('insert', np.r_[c, z+cfg.insert_z], vertical, 1., 0.),
            ('release', np.r_[c, z+cfg.insert_z], vertical, -1., 1.1),
            ('retreat', np.r_[c, z+cfg.lift_z], vertical, -1., 0.),
        ]
        self._advance()

    def step(self, observation):
        hold = np.r_[np.zeros(6), self._grip].astype(np.float32)
        if self.done or self.failed:
            return hold
        now = self.clock()
        if now-self._started > self.config.total_timeout:
            self.failed = 'teacher total timeout'
            return hold
        state = np.asarray(observation['state'], dtype=float)
        if state.shape != (8,) or not np.isfinite(state).all():
            self.failed = 'invalid robot proprioception'
            return hold
        if self.stage == 'observe':
            try:
                m = measure_scene(observation['image'], self.camera, self.config.table_z)
                self._measurements.append(m)
                self._measurements = self._measurements[-4:]
                self.debug = m.as_dict()
                if len(self._measurements) >= 3:
                    p = np.array([row.pencil_xy for row in self._measurements])
                    c = np.array([row.holder_xy for row in self._measurements])
                    if max(p.std(0).max(), c.std(0).max()) < .008:
                        m.pencil_xy = np.median(p, axis=0)
                        m.holder_xy = np.median(c, axis=0)
                        self.initial = m.as_dict()
                        self._make_plan(m)
            except (ValueError, np.linalg.LinAlgError) as error:
                self.debug = {'vision_error': str(error)}
            if now-self._stage_started > 8. and self.stage == 'observe':
                self.failed = 'initial visual localization timeout'
            return hold
        if now-self._stage_started > self.config.stage_timeout:
            self.failed = 'teacher stage timeout: '+self.stage
            return hold
        stage, target, rotation, grip, dwell = self._plan[self._index]
        self._grip = grip
        delta = target-state[:3]
        current = Rotation.from_rotvec(state[3:6]).as_matrix()
        angular = Rotation.from_matrix(rotation @ current.T).as_rotvec()
        self.debug.update({'stage': stage, 'target_xyz': target.tolist(),
                           'position_error': float(np.linalg.norm(delta)),
                           'orientation_error': float(np.linalg.norm(angular)),
                           'finger_width': float(np.abs(state[6:]).sum())})
        reached = np.linalg.norm(delta)<self.config.position_tolerance and np.linalg.norm(angular)<self.config.orientation_tolerance
        if reached and now-self._stage_started >= dwell:
            if stage == 'close' and self.config.verify_grasp and np.abs(state[6:]).sum()<.008:
                self.failed = 'empty grasp: fingers closed without an object'
                return np.r_[np.zeros(6), self._grip].astype(np.float32)
            self._advance()
        translation = delta*self.config.position_gain
        translation *= min(1., self.config.max_position_step/max(np.linalg.norm(translation),1e-12))
        angular *= self.config.orientation_gain
        angular *= min(1., self.config.max_rotation_step/max(np.linalg.norm(angular),1e-12))
        return np.r_[translation/.05, angular/.5, grip].clip(-1.,1.).astype(np.float32)
