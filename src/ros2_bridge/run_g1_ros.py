"""Run the val_861 mission scene with a sensor G1 and publish it to ROS 2."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import math
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(__file__).resolve().parent
ENV_ROOT = SRC_ROOT / "environment"
DEFAULT_SCENE = ENV_ROOT / "test_zone" / "world" / "layout.json"
DEFAULT_POLICY = ENV_ROOT / "robots" / "control" / "g1" / "walking" / "g1_29dof_velocity.onnx"
DEFAULT_ASSETS = ENV_ROOT / "models" / "assets" / "molmo"
DEFAULT_PANDA_DATASET = SRC_ROOT.parent / "datasets" / "val861_panda_base_v4"

G1_RESPAWN_XY = (16.0, 11.0)

G1_MISSION_SPAWNS = {
    2: ("cone-room door", (15.50, 20.15), 90.0),
    3: ("cliff-course entrance", (13.75, 20.125), 180.0),
    5: ("Panda desk east side", (22.25, 19.55), 180.0),
    6: ("soccer-room door", (13.75, 9.10), 180.0),
}

FALL_SOURCE = "fell over"
START_LABEL = "G1 start (16, 11)"
# Fallen = pelvis below FALL_ROOT_Z or tilted past FALL_TILT_DEG for FALL_HOLD_S of sim time.
FALL_ROOT_Z = 0.30
FALL_TILT_DEG = 60.0
FALL_HOLD_S = 5.0

from ros2_bridge.tools.scenario import (  # noqa: E402
    DEFAULT_BALL_REGION,
    DEFAULT_M1_GOAL_REGION,
    DEFAULT_M1_GOAL_YAW_DEG,
    Scenario,
    quat_yaw_deg,
    sample_point,
)
from ros2_bridge.tools.sensors import load_sensors, sensors_signature  # noqa: E402

ROOM_SLOT_CENTERS = (17.5, 10.5, 3.5)
ROOM_SEMANTICS = {
    "classroom": {"room_id": "room6", "center_y": 17.5},
    "medical": {"room_id": "room7", "center_y": 10.5},
    "cafeteria": {"room_id": "room5", "center_y": 3.5},
}


def soften_classic_visuals(model) -> None:
    import mujoco
    import numpy as np

    if getattr(mujoco, "mjRENDERER", "classic") == "filament":
        return
    model.vis.headlight.ambient[:] = (0.32, 0.32, 0.32)
    model.vis.headlight.diffuse[:] = (0.58, 0.58, 0.58)
    model.vis.headlight.specular[:] = (0.04, 0.04, 0.04)
    if model.nlight:
        model.light_diffuse[:] *= 0.78
        model.light_specular[:] *= 0.10
    if model.nmat:
        model.mat_specular[:] = np.minimum(model.mat_specular, 0.12)
        model.mat_shininess[:] = np.minimum(model.mat_shininess, 0.22)


def set_black_skybox(model) -> int:
    import mujoco

    changed = 0
    for texture_id in range(model.ntex):
        if model.tex_type[texture_id] != mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            continue
        start = int(model.tex_adr[texture_id])
        count = (
            int(model.tex_width[texture_id])
            * int(model.tex_height[texture_id])
            * int(model.tex_nchannel[texture_id])
        )
        model.tex_data[start:start + count] = 0
        changed += 1
    model.vis.rgba.haze[:] = (0.0, 0.0, 0.0, 1.0)
    model.vis.rgba.fog[:] = (0.0, 0.0, 0.0, 1.0)
    return changed


def release_compile_memory() -> None:
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def set_scene_overview_camera(camera, model, data, scene_ngeom: int) -> None:
    import mujoco
    import numpy as np

    count = min(int(scene_ngeom), model.ngeom)
    positions = np.asarray(data.geom_xpos[:count], dtype=float)
    non_plane = np.asarray(model.geom_type[:count]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
    positions = positions[non_plane & np.all(np.isfinite(positions), axis=1)]
    if not positions.size:
        return
    low = np.percentile(positions, 1.0, axis=0)
    high = np.percentile(positions, 99.0, axis=0)
    span_xy = float(max(high[0] - low[0], high[1] - low[1]))
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.trackbodyid = -1
    camera.lookat[:] = ((low + high) * 0.5)
    camera.lookat[2] = max(0.5, float(low[2] + 0.8))
    camera.distance = max(12.0, span_xy * 1.65)
    camera.azimuth = 90.0
    camera.elevation = -68.0


def set_panda_table_camera(camera, panda_xy: Sequence[float] = (21.18, 20.05)) -> None:
    import mujoco

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.trackbodyid = -1
    camera.lookat[:] = (float(panda_xy[0]), float(panda_xy[1]) - 0.43, 0.76)
    camera.distance = 2.25
    camera.azimuth = 180.0
    camera.elevation = -55.0


def set_g1_focus_camera(camera, model, data, prefix: str) -> None:
    import mujoco

    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis"
    )
    if body_id < 0:
        candidates = [
            index
            for index in range(model.nbody)
            if (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, index
            ) or "").startswith(prefix)
            and (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, index
            ) or "").endswith("pelvis")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"could not resolve one G1 pelvis under prefix {prefix!r}"
            )
        body_id = candidates[0]

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.trackbodyid = -1
    camera.lookat[:] = data.xpos[body_id]
    camera.lookat[2] += 0.45
    camera.distance = 5.0
    camera.azimuth = 135.0
    camera.elevation = -50.0


def set_g1_chase_camera(camera, model, data, prefix: str) -> None:
    import mujoco

    root_id = _prefixed_root_body(model, prefix)
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "head_camera_mount")
    if head_id < 0:
        head_id = root_id
    heading = math.degrees(math.atan2(data.xmat[root_id, 3], data.xmat[root_id, 0]))
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[head_id]
    camera.distance = 3.5
    camera.azimuth = heading + 225.0
    camera.elevation = -18.0


def _prefixed_root_body(model, prefix: str) -> int:
    import mujoco

    matches = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not name.startswith(prefix):
            continue
        parent_id = int(model.body_parentid[body_id])
        parent_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or ""
        )
        if not parent_name.startswith(prefix):
            matches.append(body_id)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one root body under {prefix!r}, found {len(matches)}"
        )
    return matches[0]


def _free_joint_for_body(model, body_id: int) -> int:
    import mujoco

    start = int(model.body_jntadr[body_id])
    count = int(model.body_jntnum[body_id])
    for joint_id in range(start, start + count):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id
    return -1


class SoccerMissionReset:
    def __init__(self, model, data) -> None:
        self.model = model
        self.goal_body = _prefixed_root_body(
            model, "editor/soccer_goal_model_38f1faa2/"
        )
        self.ball_body = _prefixed_root_body(model, "editor/soccer_ball_center/")
        self.goal_pos = model.body_pos[self.goal_body].copy()
        self.goal_quat = model.body_quat[self.goal_body].copy()
        self.ball_joint = _free_joint_for_body(model, self.ball_body)
        if self.ball_joint < 0:
            raise RuntimeError("Mission 6 soccer ball must retain its free joint")
        self.ball_qadr = int(model.jnt_qposadr[self.ball_joint])
        self.ball_dadr = int(model.jnt_dofadr[self.ball_joint])
        self.ball_qpos = data.qpos[self.ball_qadr:self.ball_qadr + 7].copy()
        self.ball_start = self.ball_qpos.copy()
        self._goal_bounds: tuple[float, ...] | None = None
        import mujoco

        self.ball_geom = next(
            (g for g in range(model.ngeom)
             if int(model.geom_bodyid[g]) == self.ball_body and int(model.geom_contype[g]) != 0),
            -1,
        )
        self.ring_geom = next(
            (g for g in range(model.ngeom)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(
                 "soccer_pitch_markings/ball_position_ring")),
            -1,
        )

    @property
    def home_xy(self) -> tuple[float, float]:
        return float(self.ball_qpos[0]), float(self.ball_qpos[1])

    def set_home(self, data, x: float, y: float) -> None:
        import mujoco

        self.ball_qpos[0], self.ball_qpos[1] = float(x), float(y)
        if self.ring_geom >= 0:
            body = int(self.model.geom_bodyid[self.ring_geom])
            rotation = data.xmat[body].reshape(3, 3)
            local = self.model.geom_pos[self.ring_geom].copy()
            world_z = data.xpos[body][2] + (rotation @ local)[2]
            local_xy = rotation.T @ (np.array([x, y, world_z], dtype=float) - data.xpos[body])
            self.model.geom_pos[self.ring_geom] = [local_xy[0], local_xy[1], local[2]]
        self.reset(data)
        mujoco.mj_forward(self.model, data)

    def set_friction(self, friction) -> None:
        if self.ball_geom < 0:
            raise RuntimeError("Mission 6 soccer ball has no collision geom")
        self.model.geom_friction[self.ball_geom] = [float(v) for v in friction]
        self.model.geom_priority[self.ball_geom] = 1
        # Rolling friction requires condim 6; condim 3 leaves the ball rolling indefinitely.
        self.model.geom_condim[self.ball_geom] = 6

    def goal_bounds(self, data) -> tuple[float, float, float, float, float]:
        if self._goal_bounds is None:
            from ros2_bridge.tools.mission_judge import goal_mouth

            bounds = goal_mouth(self.model, data, self.goal_body)
            if bounds is None:
                raise RuntimeError("Mission 6 soccer goal has no post/crossbar collision geoms")
            self._goal_bounds = bounds
        return self._goal_bounds

    def reset(self, data) -> None:
        import mujoco

        self.model.body_pos[self.goal_body] = self.goal_pos
        self.model.body_quat[self.goal_body] = self.goal_quat
        data.qpos[self.ball_qadr:self.ball_qadr + 7] = self.ball_qpos
        data.qvel[self.ball_dadr:self.ball_dadr + 6] = 0.0
        data.qacc[self.ball_dadr:self.ball_dadr + 6] = 0.0
        data.qfrc_applied[self.ball_dadr:self.ball_dadr + 6] = 0.0
        data.xfrc_applied[self.ball_body] = 0.0
        data.xfrc_applied[self.goal_body] = 0.0
        self.ball_start = self.ball_qpos.copy()
        mujoco.mj_forward(self.model, data)

    def metrics(self, data) -> dict[str, float]:
        position = data.qpos[self.ball_qadr:self.ball_qadr + 3]
        initial = self.ball_start[:3]
        velocity = data.qvel[self.ball_dadr:self.ball_dadr + 3]
        xmin, xmax, ymin, ymax, zmax = self.goal_bounds(data)
        in_goal = bool(
            xmin <= position[0] <= xmax
            and ymin <= position[1] <= ymax
            and position[2] <= zmax
        )
        return {
            "horizontal_displacement_m": float(
                np.linalg.norm(position[:2] - initial[:2])
            ),
            "speed_mps": float(np.linalg.norm(velocity)),
            "x_m": float(position[0]),
            "y_m": float(position[1]),
            "z_m": float(position[2]),
            "in_goal": in_goal,
            "goal_bounds": [xmin, xmax, ymin, ymax, zmax],
        }


class ConePermutationRuntime:
    CONES = ("blue", "green", "orange")

    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data
        self.rng = np.random.default_rng()
        self.current_order = self.CONES
        self._bodies: dict[str, int] = {}
        self._pose0: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._slot_x: list[float] = []
        for color in self.CONES:
            body_id = _prefixed_root_body(model, f"editor/extension_cone_{color}/")
            if _free_joint_for_body(model, body_id) >= 0:
                raise RuntimeError(f"mission-2 cone must be fixed: {color}")
            self._bodies[color] = body_id
            self._pose0[color] = (
                model.body_pos[body_id].copy(), model.body_quat[body_id].copy()
            )
            self._slot_x.append(float(model.body_pos[body_id, 0]))
        self._slot_x.sort()

    def randomize(self) -> tuple[str, ...]:
        choices = [
            order for order in itertools.permutations(self.CONES)
            if order != self.current_order
        ]
        return self.apply(choices[int(self.rng.integers(len(choices)))])

    def apply(self, order) -> tuple[str, ...]:
        import mujoco

        order = tuple(order)
        if sorted(order) != sorted(self.CONES):
            raise ValueError(f"cone order must list {self.CONES} once each, got {order}")
        for slot_index, color in enumerate(order):
            body_id = self._bodies[color]
            position, quaternion = self._pose0[color]
            moved = position.copy()
            moved[0] = self._slot_x[slot_index]
            self.model.body_pos[body_id] = moved
            self.model.body_quat[body_id] = quaternion
        mujoco.mj_forward(self.model, self.data)
        self.current_order = tuple(order)
        return self.current_order


class RoomPermutationRuntime:
    def __init__(self, model, data, layout_path: Path, assets_dir: Path, task: dict):
        import mujoco

        self.model = model
        self.data = data
        self.task = task
        self.rng = np.random.default_rng()
        self.current_order = tuple(ROOM_SEMANTICS)
        self.original_order = self.current_order
        self._static_bodies: dict[str, set[int]] = {
            semantic: set() for semantic in ROOM_SEMANTICS
        }
        self._dynamic_bodies: dict[str, list[tuple[int, int, int, np.ndarray]]] = {
            semantic: [] for semantic in ROOM_SEMANTICS
        }
        self._body_pose0: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        layout = json.loads(layout_path.read_text())
        for item in layout.get("objects", []):
            if not item.get("active", True) or not item.get("pos"):
                continue
            x, y = map(float, item["pos"][:2])
            if not 17.85 <= x <= 23.15:
                continue
            semantic = min(
                ROOM_SEMANTICS,
                key=lambda name: abs(y - ROOM_SEMANTICS[name]["center_y"]),
            )
            if abs(y - ROOM_SEMANTICS[semantic]["center_y"]) > 3.51:
                continue
            prefix = f"editor/{item['id']}/"
            body_id = _prefixed_root_body(model, prefix)
            self._register_body(semantic, body_id, data)

        pencil_body = _prefixed_root_body(model, "editor/panda_task_random_pencil/")
        self._register_body("classroom", pencil_body, data)

        panda_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "panda/robot0_base"
        )
        if panda_body < 0:
            raise RuntimeError("Panda base is missing from room permutation model")
        self.panda_body = int(panda_body)
        self._register_body("classroom", self.panda_body, data)

        zone_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "panda_task_zone_visuals"
        )
        if zone_body >= 0:
            self._register_body("classroom", int(zone_body), data)

        description_path = (
            assets_dir / "scenes" / "holodeck-objaverse-val" / "val_861.json"
        )
        description = json.loads(description_path.read_text())
        room_index = {
            room["id"]: index for index, room in enumerate(description["rooms"])
        }
        for semantic, record in ROOM_SEMANTICS.items():
            index = room_index[record["room_id"]]
            for body_name in (f"room_{index}", f"editor_floor_collision_{index}"):
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                if body_id < 0:
                    raise RuntimeError(
                        f"room permutation floor body is missing: {body_name}"
                    )
                self._register_body(semantic, int(body_id), data)

        self._task0 = copy.deepcopy(task)
        self._task_site0: dict[int, np.ndarray] = {}
        for name in ("panda_task_pencil_spawn", "panda_task_cup_drop_target"):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                self._task_site0[int(site_id)] = model.site_pos[site_id].copy()

        wall_colors = layout["structure"]["room_wall_finishes"]
        self._wall_rgba = {
            semantic: np.asarray(wall_colors[record["room_id"]]["rgba"], dtype=float)
            for semantic, record in ROOM_SEMANTICS.items()
        }
        slot_wall_geoms: list[list[int]] = [[] for _ in ROOM_SLOT_CENTERS]
        for geom_id in range(model.ngeom):
            body_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom_id]),
            ) or ""
            if not body_name.startswith(
                ("wall_", "editor_wall_", "editor_room_finish_")
            ):
                continue
            x, y = map(float, data.geom_xpos[geom_id, :2])
            if not 17.85 <= x <= 23.15 or not -0.02 <= y <= 21.02:
                continue
            # Assign each finish skin to its nearest room to keep shared-wall sides separate.
            slot_index = int(np.argmin(np.abs(np.asarray(ROOM_SLOT_CENTERS) - y)))
            slot_wall_geoms[slot_index].append(geom_id)
        self._slot_wall_geoms = [
            np.asarray(ids, dtype=np.int32) for ids in slot_wall_geoms
        ]

        counts = [len(self._static_bodies[name]) for name in ROOM_SEMANTICS]
        if min(counts) < 3 or any(ids.size == 0 for ids in self._slot_wall_geoms):
            raise RuntimeError(
                "incomplete semantic-room bundle: "
                f"static bodies={counts}, wall geoms="
                f"{[ids.size for ids in self._slot_wall_geoms]}"
            )

    def _register_body(self, semantic: str, body_id: int, data) -> None:
        free_joint = _free_joint_for_body(self.model, body_id)
        if free_joint >= 0:
            qadr = int(self.model.jnt_qposadr[free_joint])
            dadr = int(self.model.jnt_dofadr[free_joint])
            self._dynamic_bodies[semantic].append(
                (body_id, qadr, dadr, data.qpos[qadr:qadr + 7].copy())
            )
            return
        if int(self.model.body_parentid[body_id]) != 0:
            raise RuntimeError(
                "room bundle root must be attached to the world: "
                f"body_id={body_id}"
            )
        self._static_bodies[semantic].add(body_id)
        self._body_pose0.setdefault(
            body_id,
            (
                self.model.body_pos[body_id].copy(),
                self.model.body_quat[body_id].copy(),
            ),
        )

    @staticmethod
    def _translated_xy(value, delta_y: float) -> np.ndarray:
        result = np.asarray(value, dtype=float).copy()
        result[1] += delta_y
        return result

    def _translate_task_frame(self, delta_y: float) -> None:
        for key in ("desk_center", "base_xy", "pencil_xy", "cup_xy"):
            if key in self._task0:
                self.task[key] = self._translated_xy(self._task0[key], delta_y)
        if "desk_bounds" in self._task0:
            x0, x1, y0, y1 = self._task0["desk_bounds"]
            self.task["desk_bounds"] = (x0, x1, y0 + delta_y, y1 + delta_y)
        for key in ("pencil_sector", "cup_sector"):
            if key in self._task0:
                self.task[key] = copy.deepcopy(self._task0[key])
                self.task[key]["center"] = self._translated_xy(
                    self._task0[key]["center"], delta_y
                )
        for site_id, position in self._task_site0.items():
            moved = position.copy()
            moved[1] += delta_y
            self.model.site_pos[site_id] = moved

    def randomize(self) -> tuple[tuple[str, ...], np.ndarray, float]:
        choices = [
            order
            for order in itertools.permutations(ROOM_SEMANTICS)
            if order != self.current_order
        ]
        return self.apply(choices[int(self.rng.integers(len(choices)))])

    def apply(self, order) -> tuple[tuple[str, ...], np.ndarray, float]:
        import mujoco

        order = tuple(order)
        if sorted(order) != sorted(ROOM_SEMANTICS):
            raise ValueError(f"room order must list {tuple(ROOM_SEMANTICS)} once each, got {order}")
        target_center = {
            semantic: ROOM_SLOT_CENTERS[index]
            for index, semantic in enumerate(order)
        }
        for semantic, record in ROOM_SEMANTICS.items():
            delta_y = target_center[semantic] - float(record["center_y"])
            for body_id in self._static_bodies[semantic]:
                position, quaternion = self._body_pose0[body_id]
                moved = position.copy()
                moved[1] += delta_y
                self.model.body_pos[body_id] = moved
                self.model.body_quat[body_id] = quaternion
            for body_id, qadr, dadr, qpos0 in self._dynamic_bodies[semantic]:
                moved = qpos0.copy()
                moved[1] += delta_y
                self.data.qpos[qadr:qadr + 7] = moved
                self.data.qvel[dadr:dadr + 6] = 0.0
                self.data.qacc[dadr:dadr + 6] = 0.0
                self.data.qfrc_applied[dadr:dadr + 6] = 0.0
                self.data.xfrc_applied[body_id] = 0.0
            if semantic == "classroom":
                self._translate_task_frame(delta_y)

        for slot_index, semantic in enumerate(order):
            geom_ids = self._slot_wall_geoms[slot_index]
            self.model.geom_matid[geom_ids] = -1
            self.model.geom_rgba[geom_ids] = self._wall_rgba[semantic]

        mujoco.mj_forward(self.model, self.data)
        self.current_order = tuple(order)
        panda_position = self.data.xpos[self.panda_body].copy()
        classroom_delta_y = (
            target_center["classroom"] - ROOM_SEMANTICS["classroom"]["center_y"]
        )
        return self.current_order, panda_position, classroom_delta_y

    def rebase_task_objects(self, body_ids) -> None:
        delta_y = (
            ROOM_SLOT_CENTERS[self.current_order.index("classroom")]
            - ROOM_SEMANTICS["classroom"]["center_y"]
        )
        dynamic = self._dynamic_bodies["classroom"]
        for index, (body_id, qadr, dadr, _qpos0) in enumerate(dynamic):
            if body_id in body_ids:
                qpos0 = self.data.qpos[qadr:qadr + 7].copy()
                qpos0[1] -= delta_y
                dynamic[index] = (body_id, qadr, dadr, qpos0)
        for body_id in body_ids:
            if body_id in self._body_pose0:
                position = self.model.body_pos[body_id].copy()
                position[1] -= delta_y
                self._body_pose0[body_id] = (position, self.model.body_quat[body_id].copy())
        for key in ("pencil_xy", "cup_xy"):
            self._task0[key] = self._translated_xy(self.task[key], -delta_y)
        for site_id in self._task_site0:
            position = self.model.site_pos[site_id].copy()
            position[1] -= delta_y
            self._task_site0[site_id] = position

    @property
    def panda_position(self) -> np.ndarray:
        return self.data.xpos[self.panda_body].copy()


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class MissionStatusPublisher:
    def __init__(self, node, mission_topic: str) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._String = String
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._mission = node.create_publisher(String, mission_topic, qos)
        self._seq = 0
        self.common: dict = {}

    def _send(self, publisher, payload: dict) -> None:
        self._seq += 1
        message = self._String()
        message.data = json.dumps(
            {"seq": self._seq, "wall_time": time.time(), **self.common, **payload},
            default=_json_default,
        )
        publisher.publish(message)

    def mission(self, event: str, index: int, label: str, pos, source: str, sim_time: float,
                yaw: float | None = None, extra: dict | None = None) -> None:
        self._send(self._mission, {
            "event": event, "mission": int(index), "label": label,
            **({} if pos is None else {"pos": [float(v) for v in pos]}),
            "source": source, "sim_time": float(sim_time),
            **({} if yaw is None else {"yaw_deg": math.degrees(float(yaw))}),
            **(extra or {}),
        })


def _load_combined_builder():
    from ros2_bridge.tools import val861_scene

    return val861_scene


def hold_pose(model, data, prefix: str = "") -> int:
    import mujoco

    n = 0
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint = int(model.actuator_trnid[actuator, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if prefix and not name.startswith(prefix):
            continue
        data.ctrl[actuator] = data.qpos[model.jnt_qposadr[joint]]
        n += 1
    return n


def ensure_sensor_robot(assets_dir: Path, regenerate: bool = False,
                        sensors: Optional[dict] = None) -> Path:
    from ros2_bridge.tools.g1_sensor_rig import (
        EXTERIOR_MARKER,
        PLACEMENT_MARKER,
        build_sensor_xml,
    )
    from ros2_bridge.tools.interactive_sim import ROBOT_XML, resolve_robot_xml

    sensors = sensors or load_sensors(None)
    placement = sensors_signature(sensors)
    source = resolve_robot_xml("g1", assets_dir)
    # Keep the generated MJCF beside g1_dex.xml so meshdir="assets/" resolves.
    output = Path(ROBOT_XML["g1_sensors"])
    current = output.read_text(errors="ignore") if output.exists() else ""
    if regenerate or EXTERIOR_MARKER not in current or PLACEMENT_MARKER + placement not in current:
        output.parent.mkdir(parents=True, exist_ok=True)
        camera, lidar = sensors["g1"]["camera"], sensors["g1"]["lidar"]
        build_sensor_xml(source, output, lidar="OS1-32", pitch_deg=camera["pitch_deg"],
                         camera_forward_m=camera["forward_m"], camera_up_m=camera["up_m"],
                         lidar_up_m=lidar["up_m"], placement=placement)
    return output


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE,
                    help="MJCF or editor layout JSON (default: current edited val_861)")
    ap.add_argument("--assets-dir", type=Path,
                    default=Path(os.environ.get("MLSPACES_ASSETS_DIR", DEFAULT_ASSETS)))
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--regenerate-sensors", action="store_true")
    ap.add_argument(
        "--shadow-size",
        type=int,
        default=0,
        help="shadow-map resolution applied to the compiled scene (0 = keep the scene's "
             "value, 16384). Every GL context (viewer, G1 camera renderer, Panda camera "
             "renderer) allocates this map, so 4096 cuts GPU memory roughly in half on an "
             "8 GB card at the cost of softer shadow edges",
    )
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until viewer closes/Ctrl-C")
    ap.add_argument(
        "--fall-respawn", action=argparse.BooleanOptionalAction, default=True,
        help="respawn G1 at world (16, 11), yaw 0 when it has fallen over or into a gap "
             f"(pelvis below {FALL_ROOT_Z} m or tilted past {FALL_TILT_DEG:g} deg for "
             f"{FALL_HOLD_S:g} s; mission status: event respawned, source 'fell over')",
    )
    ap.add_argument(
        "--scenario", type=Path, default=None, metavar="FILE",
        help="scenario file (ros2_bridge/tools/scenario.py): mission 1 goal, mission 2 "
             "colour, cone and room order, ball, pencil/holder; default: the scene's stock "
             "layout.  Key R / /g1/randomize draw a new one; every mission status carries "
             "the current one as 'environment'",
    )
    ap.add_argument(
        "--hide-targets", action="store_true",
        help="leave mission 1's goal and mission 2's colour out of the mission status (main.py "
             "--llm-web: the missions get them from the instruction); still judged and drawn",
    )
    ap.add_argument(
        "--scenario-out", type=Path, default=None, metavar="FILE",
        help="write the current scenario to this YAML at start and after every randomization",
    )
    ap.add_argument(
        "--sensors", type=Path, default=None, metavar="FILE",
        help="custom/robot.yaml sensors: G1 camera tilt/offset, lidar model/height, Panda cameras",
    )
    ap.add_argument(
        "--m1-goal-region", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
        default=list(DEFAULT_M1_GOAL_REGION),
        help="where a randomized mission 1 goal may land (world box)",
    )
    ap.add_argument("--m1-goal-yaw", type=float, default=DEFAULT_M1_GOAL_YAW_DEG,
                    help="yaw_deg of a randomized mission 1 goal")
    ap.add_argument(
        "--ball-region", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
        default=list(DEFAULT_BALL_REGION),
        help="where a randomized soccer ball may land (world box, inside the pitch)",
    )
    ap.add_argument(
        "--ball-friction", type=float, nargs=3, metavar=("SLIDE", "SPIN", "ROLL"),
        default=None,
        help="the soccer ball's friction coefficients (default: the ball model's); it wins "
             "over the floor's",
    )
    ap.add_argument(
        "--judge-regions", action=argparse.BooleanOptionalAction, default=True,
        help="draw the referee's success regions (key J) and the random spawn regions + "
             "Panda task zones (key K) in the viewer at start",
    )
    ap.add_argument(
        "--m4-room", choices=("classroom", "medical", "cafeteria"), default="classroom",
        help="mission 4 referee: the room the delivered object belongs in",
    )
    ap.add_argument("--judge-goal-radius", type=float, default=0.5,
                    help="mission 1 referee: pelvis within this many metres of the goal")
    ap.add_argument("--judge-cone-box", type=float, default=0.5,
                    help="mission 2 referee: side (m) of the square in front of the target cone")
    ap.add_argument("--judge-cone-offset", type=float, default=0.75,
                    help="mission 2 referee: the square's centre this far (m) in front of the cone")
    ap.add_argument(
        "--g1-mission-end-topic", default="/g1/mission_end",
        help="std_msgs/Int32 topic: mission N ended; the referee judges G1's true pose "
             "(mission status: event judged, field judge)",
    )
    ap.add_argument("--walk", action=argparse.BooleanOptionalAction, default=True,
                    help="enable the learned controller (default: enabled, zero command)")
    ap.add_argument("--view", action=argparse.BooleanOptionalAction, default=True,
                    help="show the interactive MuJoCo window")
    ap.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True,
                    help="pace headless simulation to wall time")
    ap.add_argument("--pos", type=float, nargs="+", default=None, metavar="XYZ",
                    help="spawn x y [z]; omit for automatic free-space placement")
    ap.add_argument("--spawn", default=None, metavar='"X Y [YAW_DEG]"',
                    help='spawn pose in one string, e.g. "15.5 21.5 90" (sets --pos and '
                         '--yaw-deg); "auto" keeps the default placement')
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="spawn heading (0: the reported start-pose frame's axes stay parallel "
                         "to the building)")
    ap.add_argument("--panda", action=argparse.BooleanOptionalAction, default=True,
                    help="spawn Panda and its exact task zones in layout scenes")
    ap.add_argument("--panda-x", type=float, default=21.18)
    ap.add_argument("--panda-y", type=float, default=20.05)
    ap.add_argument("--panda-z", type=float, default=0.8016)
    ap.add_argument("--panda-yaw-deg", type=float, default=-90.0)
    ap.add_argument("--task-seed", type=int, default=None)
    ap.add_argument("--task-zones", action=argparse.BooleanOptionalAction, default=True,
                    help="show the exact Panda pencil/cup spawn sectors")
    ap.add_argument(
        "--panda-mode",
        choices=("ros", "collect"),
        default="ros",
        help="Panda controller: ROS command topics, or TeleDex demonstration recording",
    )
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_PANDA_DATASET)
    ap.add_argument(
        "--task-instruction",
        default="Pick up the wooden pencil and place it inside the gray square pencil holder.",
    )
    ap.add_argument("--teledex-port", type=int, default=8888)
    ap.add_argument("--panda-rate", type=float, default=10.0,
                    help="Panda recording rate in --panda-mode collect (default: LIBERO 10 Hz)")
    ap.add_argument("--panda-position-scale", type=float, default=1.6)
    ap.add_argument(
        "--panda-yaw-scale",
        type=float,
        default=1.0,
        help="TeleDex display-up yaw gain; use -1 only to invert (default: 1)",
    )
    ap.add_argument(
        "--panda-pitch-scale",
        type=float,
        default=1.0,
        help="TeleDex phone top-edge tilt gain; use -1 to invert, 0 to disable (default: 1)",
    )
    ap.add_argument(
        "--panda-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show exterior/wrist collection cameras in an OpenCV window",
    )
    ap.add_argument("--auto-focus-panda-distance", type=float, default=1.8,
                    help="switch to the Panda desk camera once G1 is within this many metres (default: 1.8); 0 disables")
    ap.add_argument("--cmd-vel-topic", "--cmd-vel", dest="cmd_vel_topic", default="/cmd_vel")
    ap.add_argument(
        "--nav",
        action="store_true",
        help="Nav2 on ground truth: --footprint, odom TF, realtime, a 0.8 s twist "
             "window and a 0.06 idle deadband (cannot be combined with --mola)",
    )
    ap.add_argument("--footprint", action="store_true",
                    help="publish odom -> base_footprint -> pelvis instead of odom -> pelvis, "
                         "so Nav2 gets a base that stays flat while the pelvis bobs")
    ap.add_argument("--odom-hz", type=float, default=50.0,
                    help="ground-truth /odom rate; 0 disables it (forced by --mola)")
    ap.add_argument("--odom-tf", action=argparse.BooleanOptionalAction, default=True,
                    help="broadcast the ground-truth odom TF along with /odom")
    ap.add_argument("--odom-twist-window", type=float, default=None,
                    help="average the /odom twist over this many sim seconds (one gait "
                         "cycle is 0.8); the pose stays exact. Default: 0.8 with --nav, "
                         "else 0 (raw)")
    ap.add_argument("--cmd-bias", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("VX", "VY", "WZ"),
                    help="feed-forward calibration added to /cmd_vel after the clamp "
                         "(and to the watchdog's stop) to cancel residual creep")
    ap.add_argument("--idle-hold", action=argparse.BooleanOptionalAction, default=True,
                    help="hold the G1 base still while the command is inside the idle "
                         "deadband instead of letting the gait shuffle in place")
    ap.add_argument("--idle-threshold", type=float, default=None,
                    help="command norm treated as idle by --idle-hold; must exceed any "
                         "--cmd-bias (default: 0.06 with --nav, else 0.001)")
    ap.add_argument(
        "--mola",
        action="store_true",
        help="run under custom/g1_navigation: disable ground-truth /odom and its "
             "odom -> pelvis TF (MOLA and the frame splitter own that chain)",
    )
    ap.add_argument(
        "--ground-truth",
        action="store_true",
        help="publish ground truth: /odom (--odom-hz) with its odom TF and the --nav chain "
             "(implied).  Off by default -- without it the simulator publishes "
             "no ground-truth pose at all and odometry has to come from MOLA",
    )
    ap.add_argument(
        "--auto-initial-pose",
        action="store_true",
        help="seed AMCL with G1's true pose: one PoseWithCovarianceStamped on "
             "--initial-pose-topic when AMCL subscribes, after every respawn and for "
             "each std_msgs/Empty on --localize-topic (a one-shot 2D Pose Estimate, "
             "not an odometry stream)",
    )
    ap.add_argument("--initial-pose-topic", default="/initialpose")
    ap.add_argument("--localize-topic", default="/g1/localize",
                    help="std_msgs/Empty: send the --auto-initial-pose seed now; answered "
                         "with mission_status event 'initial_pose'")
    ap.add_argument(
        "--g1-mission-begin-topic",
        default="/g1/mission_begin",
        help="std_msgs/Int32: make mission N the active one (its resets, e.g. the soccer "
             "ball, and the rescue point) WITHOUT moving G1; answered with mission_status "
             "event 'selected'",
    )
    ap.add_argument(
        "--g1-respawn-topic",
        default="/g1/respawn",
        help="std_msgs/Empty topic that restores G1 to the active mission spawn",
    )
    ap.add_argument(
        "--respawned-topic", default="/g1/respawned",
        help="std_msgs/Empty: published once every time G1 is respawned at the start point "
             "(16, 11), yaw 0 (fall, gap, key B)",
    )
    ap.add_argument(
        "--g1-randomize-topic",
        default="/g1/randomize",
        help="std_msgs/Empty topic: shuffle room order and cone colours (viewer key R)",
    )
    ap.add_argument("--mission-status-topic", default="/g1/mission_status",
                    help="latched std_msgs/String JSON: last G1 respawn")
    ap.add_argument("--joint-command-topic", default="/g1/joint_command",
                    help="sensor_msgs/JointState (name + position, rad): G1 joint targets; "
                         "while they arrive the walking policy is suspended")
    ap.add_argument("--joint-gains-topic", default="/g1/joint_gains",
                    help="sensor_msgs/JointState (name, position = kp, velocity = kd): PD gains "
                         "used while --joint-command-topic drives G1")
    ap.add_argument("--joint-command-timeout", type=float, default=0.2,
                    help="sim seconds without a joint command before the walking policy resumes")
    ap.add_argument("--cmd-timeout", "--cmd-vel-timeout", dest="cmd_timeout",
                    type=float, default=0.5,
                    help="wall seconds without a Twist before the command is zeroed")
    ap.add_argument("--max-vx", type=float, default=0.6)
    ap.add_argument("--max-vy", type=float, default=0.4)
    ap.add_argument("--max-wz", type=float, default=0.7)
    ap.add_argument("--max-vel", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"),
                    help="shorthand for --max-vx --max-vy --max-wz")
    ap.add_argument("--camera-hz", type=float, default=None,
                    help="G1 D435 RGB-D rate; default 10 with viewer, 20 headless")
    ap.add_argument(
        "--panda-camera-hz",
        type=float,
        default=10.0,
        help="Panda agentview/wrist ROS image rate in ros mode (default: 10, also with viewer)",
    )
    ap.add_argument("--lidar-hz", type=float, default=10.0)
    ap.add_argument("--lidar-threads", type=int, default=1, metavar="N",
                    help="threads the LiDAR raycast is split across (0 = cores-2, capped "
                         "at 8). Each worker casts through its own mjData")
    ap.add_argument("--rtf", type=float, default=1.0, metavar="FACTOR",
                    help="target real-time factor for the pacing (2.0 = sim time twice as "
                         "fast as wall time). Sensor rates are in sim time, so lower this "
                         "if MOLA or Nav2 cannot keep up")
    ap.add_argument("--imu-hz", type=float, default=100.0)
    ap.add_argument(
        "--unitree-odom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="publish Unitree SDK2 rt/odommodestate and rt/lf/odommodestate",
    )
    ap.add_argument("--unitree-odom-hz", type=float, default=500.0)
    ap.add_argument("--unitree-odom-lf-hz", type=float, default=20.0)
    ap.add_argument(
        "--unitree-dds-domain",
        type=int,
        default=int(os.environ.get("ROS_DOMAIN_ID", "0")),
        help="CycloneDDS domain for native Unitree topics (default: ROS_DOMAIN_ID or 0)",
    )
    ap.add_argument(
        "--unitree-dds-interface",
        default=None,
        help="optional NIC name for Unitree DDS; omit for automatic selection",
    )
    ap.add_argument("--lidar-columns", type=int, default=512,
                    help="512 is a real OS1 mode and keeps interactive simulation responsive")
    args = ap.parse_args(argv)
    if args.spawn is not None and args.spawn.strip().lower() not in ("", "auto"):
        values = [float(token) for token in args.spawn.replace(",", " ").split()]
        if len(values) not in (2, 3):
            ap.error(f'--spawn takes "X Y [YAW_DEG]", got {args.spawn!r}')
        args.pos = values[:2]
        if len(values) == 3:
            args.yaw_deg = values[2]
    if args.max_vel is not None:
        args.max_vx, args.max_vy, args.max_wz = args.max_vel
    if args.nav and args.mola:
        ap.error("--nav publishes ground-truth odometry; use --mola for MOLA instead")
    if args.ground_truth and args.mola:
        ap.error("--mola hands odometry to MOLA; drop --ground-truth")
    if args.ground_truth and not args.mola:
        args.nav = True
    if not args.ground_truth:
        for flag, used in (("--nav", args.nav), ("--footprint", args.footprint),
                           ("--unitree-odom", args.unitree_odom)):
            if used:
                ap.error(f"{flag} publishes ground truth, which is off by default; "
                         "add --ground-truth")
        args.odom_hz = 0.0
        args.odom_tf = False
    if args.nav:
        args.footprint = args.odom_tf = args.realtime = True
    if args.mola:
        args.footprint = False
        args.odom_hz = 0.0
    if args.odom_twist_window is None:
        args.odom_twist_window = 0.8 if args.nav else 0.0
    if args.idle_threshold is None:
        args.idle_threshold = 0.06 if args.nav else 1e-3
    if args.footprint and args.odom_hz <= 0:
        ap.error("--footprint needs ground-truth odometry (--odom-hz > 0, no --mola)")
    if args.odom_hz < 0 or args.odom_twist_window < 0 or args.idle_threshold < 0:
        ap.error("--odom-hz, --odom-twist-window and --idle-threshold must be non-negative")
    if args.idle_hold and float(np.linalg.norm(args.cmd_bias)) >= args.idle_threshold > 0:
        ap.error("--cmd-bias would keep G1 out of the idle deadband; raise --idle-threshold")
    if args.rtf <= 0:
        ap.error("--rtf must be positive")
    if args.lidar_threads < 0:
        ap.error("--lidar-threads must be non-negative")
    if args.pos is not None and len(args.pos) not in (2, 3):
        ap.error("--pos requires x y or x y z")
    if min(args.max_vx, args.max_vy, args.max_wz) <= 0:
        ap.error("velocity limits must be positive")
    if args.unitree_odom and min(args.unitree_odom_hz, args.unitree_odom_lf_hz) <= 0:
        ap.error("Unitree odometry rates must be positive")
    if args.panda_mode != "ros" and not args.panda:
        ap.error("--panda-mode collect requires --panda")
    if args.panda_rate <= 0:
        ap.error("--panda-rate must be positive")
    if not all(
        math.isfinite(value)
        for value in (args.panda_yaw_scale, args.panda_pitch_scale)
    ):
        ap.error("Panda yaw/pitch parameters must be finite")
    if args.panda_camera_hz < 0:
        ap.error("--panda-camera-hz must be non-negative")
    if args.camera_hz is None:
        args.camera_hz = 10.0 if args.view else 20.0
    if args.camera_hz < 0:
        ap.error("--camera-hz must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arguments(argv)
    args.scene = args.scene.expanduser().resolve()
    args.assets_dir = args.assets_dir.expanduser().resolve()
    args.policy = args.policy.expanduser().resolve()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    os.environ["MLSPACES_ASSETS_DIR"] = str(args.assets_dir)
    os.environ["ROS_LOCALHOST_ONLY"] = os.environ.get("HAIC_LOCALHOST_ONLY", "1")

    if not args.scene.exists():
        raise FileNotFoundError(f"scene/layout not found: {args.scene}")
    if not args.assets_dir.exists():
        raise FileNotFoundError(f"MolmoSpaces assets not found: {args.assets_dir}")
    sensors = load_sensors(args.sensors)
    args.lidar = "OS1-32"
    ensure_sensor_robot(args.assets_dir, args.regenerate_sensors, sensors)
    scenario = Scenario.load(args.scenario) if args.scenario else Scenario()
    if args.walk and not args.policy.exists():
        raise FileNotFoundError(
            f"walking policy not found: {args.policy}\n"
            f"Run: bash {SRC_ROOT / 'setup' / 'setup.sh'}"
        )

    import rclpy

    from ros2_bridge.tools.interactive_sim import InteractiveSim
    from ros2_bridge.tools.ros_publisher import SimPublisher

    task = None
    panda_prefix = None
    if args.panda:
        if args.scene.suffix.lower() != ".json":
            raise ValueError("--panda task composition currently requires an editor layout JSON")
        combined = _load_combined_builder()
        if args.pos is None:
            x, y, z = 16.0, 11.0, 0.0
        else:
            x, y = args.pos[:2]
            z = args.pos[2] if len(args.pos) == 3 else 0.0
        combined_args = SimpleNamespace(
            assets_dir=args.assets_dir,
            layout=args.scene,
            catalog=combined.DEFAULT_CATALOG,
            g1_x=float(x), g1_y=float(y), g1_z=float(z),
            g1_yaw=float(args.yaw_deg), g1_maze_start=False,
            panda_x=float(args.panda_x), panda_y=float(args.panda_y),
            panda_z=float(args.panda_z), panda_yaw=float(args.panda_yaw_deg),
            original_g1_hands=False, pin_g1=False, demo=False,
            task_seed=args.task_seed, no_task_randomization=False,
            show_task_zones=bool(args.task_zones), task_map=None, maze_map=None,
            panda_learning_cameras=args.panda_mode != "ros",
            panda_cameras=sensors["panda"],
            view="overview", check_only=True,
        )
        model, data, _removed, task = combined.build_combined_model(combined_args)

        class CompiledSim:
            def __init__(self, model, data):
                self.model, self.data = model, data

            def step(self):
                import mujoco
                mujoco.mj_step(self.model, self.data)

        sim = CompiledSim(model, data)
        import mujoco
        roots = [
            joint for joint in range(model.njnt)
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "").startswith("g1/")
        ]
        if len(roots) != 1:
            raise RuntimeError(f"expected one G1 free joint, found {len(roots)}")
        root_qadr = int(model.jnt_qposadr[roots[0]])
        spawn = SimpleNamespace(prefix="g1/", pos=data.qpos[root_qadr:root_qadr + 3].copy())
        spawn_pos = tuple(spawn.pos)
        floor_z = 0.0
        panda_prefix = "panda/"
        scene_ngeom = model.ngeom
    else:
        sim = InteractiveSim(args.scene, assets_dir=args.assets_dir, static_scene=True)
        scene_ngeom = sim.model.ngeom
        if args.pos is None:
            x, y, floor_z = sim.find_free_pos()
            spawn_pos = (x, y)
        else:
            x, y = args.pos[:2]
            floor_z = 0.0
            spawn_pos = tuple(args.pos)
        yaw = math.radians(args.yaw_deg)
        spawn = sim.spawn_robot(
            "g1_sensors", pos=spawn_pos, yaw=yaw, floor_z=floor_z,
            keyframe=None if args.walk else "stand",
        )
    yaw = math.radians(args.yaw_deg)
    soften_classic_visuals(sim.model)
    set_black_skybox(sim.model)
    if args.shadow_size > 0:
        # Set before any GL context is created from this model.
        sim.model.vis.quality.shadowsize = int(args.shadow_size)
    release_compile_memory()

    panda_controller = None
    if panda_prefix:
        from ros2_bridge.tools.panda_controller import PandaController

        panda_controller = PandaController(sim.model, sim.data, prefix=panda_prefix)
    panda_home = None if panda_controller is None else (
        sim.data.qpos[panda_controller.qpos_adrs].copy(),
        sim.data.qpos[panda_controller.finger_qpos_adrs].copy(),
    )

    walker = None
    joint_command_switch = None
    # Recover only before the first movement command; later falls count as failures.
    g1_motion_command_seen = {"value": False}
    if args.walk:
        if args.policy.suffix.lower() == ".onnx":
            from ros2_bridge.tools.g1_walk_policy_29dof import G1WalkPolicy29DoF

            policy_class = G1WalkPolicy29DoF
            policy_label = "official Unitree 29-DoF ONNX"
        else:
            from ros2_bridge.tools.g1_walk_policy import G1WalkPolicy

            policy_class = G1WalkPolicy
            policy_label = "legacy Unitree 12-DoF TorchScript"

        if getattr(policy_class, "__name__", "") == "G1WalkPolicy29DoF":
            from ros2_bridge.tools.g1_joint_command import G1JointCommandSwitch29DoF

            walker = G1JointCommandSwitch29DoF(
                sim.model, sim.data, args.policy, prefix=spawn.prefix,
                timeout=args.joint_command_timeout,
            )
            joint_command_switch = walker
        else:
            walker = policy_class(
                sim.model, sim.data, args.policy, prefix=spawn.prefix,
                cmd=(0.0, 0.0, 0.0),
            )
        walker.reset(pos=spawn.pos, yaw=yaw)
        velocity_policy = getattr(walker, "walk", walker)
        if hasattr(velocity_policy, "stationary_idle"):
            velocity_policy.stationary_idle = bool(args.idle_hold)
            velocity_policy.command_deadband = float(args.idle_threshold)
        walker.install()

        def command_g1(vx: float, vy: float, wz: float) -> None:
            if np.linalg.norm((vx, vy, wz)) > 1e-3:
                g1_motion_command_seen["value"] = True
            walker.set_command(vx, vy, wz)

        command_callback = command_g1
        state_text = (
            f"{policy_label} READY; standing until {args.cmd_vel_topic} arrives"
        )
    else:
        held = hold_pose(sim.model, sim.data, spawn.prefix)
        command_callback = None
        state_text = f"policy disabled; holding {held} joints"

    # A full mj_resetData would also reset Panda and randomized task objects.
    import mujoco

    g1_joint_ids = [
        joint_id for joint_id in range(sim.model.njnt)
        if (mujoco.mj_id2name(
            sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        ) or "").startswith(spawn.prefix)
    ]
    qpos_width = {
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    dof_width = {
        mujoco.mjtJoint.mjJNT_FREE: 6,
        mujoco.mjtJoint.mjJNT_BALL: 3,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    g1_qpos_indices = np.asarray([
        index
        for joint_id in g1_joint_ids
        for index in range(
            int(sim.model.jnt_qposadr[joint_id]),
            int(sim.model.jnt_qposadr[joint_id])
            + qpos_width[mujoco.mjtJoint(sim.model.jnt_type[joint_id])],
        )
    ], dtype=int)
    g1_dof_indices = np.asarray([
        index
        for joint_id in g1_joint_ids
        for index in range(
            int(sim.model.jnt_dofadr[joint_id]),
            int(sim.model.jnt_dofadr[joint_id])
            + dof_width[mujoco.mjtJoint(sim.model.jnt_type[joint_id])],
        )
    ], dtype=int)
    g1_actuator_ids = np.asarray([
        actuator_id for actuator_id in range(sim.model.nu)
        if sim.model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
        and (mujoco.mj_id2name(
            sim.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(sim.model.actuator_trnid[actuator_id, 0]),
        ) or "").startswith(spawn.prefix)
    ], dtype=int)
    g1_body_ids = np.asarray([
        body_id for body_id in range(sim.model.nbody)
        if (mujoco.mj_id2name(
            sim.model, mujoco.mjtObj.mjOBJ_BODY, body_id
        ) or "").startswith(spawn.prefix)
    ], dtype=int)
    initial_g1_qpos = sim.data.qpos[g1_qpos_indices].copy()
    initial_g1_ctrl = sim.data.ctrl[g1_actuator_ids].copy()
    respawn_pos = np.asarray(spawn.pos, dtype=float).copy()
    mission_spawns: dict[int, tuple[str, np.ndarray, float]] = {
        1: ("launch point", respawn_pos.copy(), yaw),
        4: ("three-room hallway view", respawn_pos.copy(), 0.0),
    }
    for mission_index, (label, xy, yaw_deg) in G1_MISSION_SPAWNS.items():
        mission_spawns[mission_index] = (
            label,
            np.asarray((xy[0], xy[1], respawn_pos[2]), dtype=float),
            math.radians(yaw_deg),
        )
    active_spawn = {"index": 1}
    respawn_requests: deque[tuple[int | None, str]] = deque(maxlen=8)
    scenario_randomize_requests: deque[str] = deque(maxlen=2)

    def respawn_g1(mission_index: int | None = None, pose=None) -> tuple[int, str, np.ndarray]:
        rescue = mission_index is None and pose is None
        if mission_index is None:
            mission_index = active_spawn["index"]
        if mission_index not in mission_spawns:
            raise ValueError(f"unknown G1 mission spawn: {mission_index}")
        label, target_pos, target_yaw = mission_spawns[mission_index]
        if rescue:
            label = START_LABEL
            target_pos = np.asarray((*G1_RESPAWN_XY, respawn_pos[2]), dtype=float)
            target_yaw = 0.0
        if pose is not None:
            label = "placed"
            target_pos = np.asarray((pose[0], pose[1], target_pos[2]), dtype=float)
            target_yaw = float(pose[2])
        active_spawn["index"] = mission_index
        if walker is not None:
            walker.set_command(0.0, 0.0, 0.0)
            walker.reset(pos=target_pos, yaw=target_yaw)
            g1_motion_command_seen["value"] = False
        else:
            sim.data.qpos[g1_qpos_indices] = initial_g1_qpos
            free_joint_ids = [
                joint_id for joint_id in g1_joint_ids
                if sim.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            ]
            if len(free_joint_ids) != 1:
                raise RuntimeError(
                    f"expected one G1 free joint for respawn, found {len(free_joint_ids)}"
                )
            root_qadr = int(sim.model.jnt_qposadr[free_joint_ids[0]])
            sim.data.qpos[root_qadr:root_qadr + 3] = target_pos
            sim.data.qpos[root_qadr + 3:root_qadr + 7] = (
                math.cos(target_yaw / 2.0), 0.0, 0.0,
                math.sin(target_yaw / 2.0),
            )
            sim.data.qvel[g1_dof_indices] = 0.0
            sim.data.ctrl[g1_actuator_ids] = initial_g1_ctrl
        sim.data.qvel[g1_dof_indices] = 0.0
        sim.data.qacc[g1_dof_indices] = 0.0
        sim.data.qfrc_applied[g1_dof_indices] = 0.0
        if g1_body_ids.size:
            sim.data.xfrc_applied[g1_body_ids] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        return mission_index, label, target_pos

    g1_root_body_id = (
        int(sim.model.jnt_bodyid[walker.root_jid]) if walker is not None else -1
    )

    def recover_unready_g1() -> tuple[float, float] | None:
        if walker is None or g1_motion_command_seen["value"]:
            return None
        root_z = float(sim.data.qpos[walker.root_qadr + 2])
        up_z = float(sim.data.xmat[g1_root_body_id].reshape(3, 3)[2, 2])
        if (
            np.isfinite(root_z)
            and np.isfinite(up_z)
            and root_z >= float(respawn_pos[2]) - 0.20
            and up_z >= math.cos(math.radians(45.0))
        ):
            return None
        failed_pose = (root_z, up_z)
        respawn_g1()
        return failed_pose

    fall_since: dict[str, float | None] = {"value": None}

    def check_fall() -> tuple[float, float] | None:
        if walker is None or not args.fall_respawn or respawn_requests:
            return None
        root_z = float(sim.data.qpos[walker.root_qadr + 2])
        up_z = float(sim.data.xmat[g1_root_body_id].reshape(3, 3)[2, 2])
        down = (not np.isfinite(root_z) or not np.isfinite(up_z) or root_z < FALL_ROOT_Z
                or up_z < math.cos(math.radians(FALL_TILT_DEG)))
        now = float(sim.data.time)
        if not down:
            fall_since["value"] = None
            return None
        if fall_since["value"] is None or now < fall_since["value"]:
            fall_since["value"] = now
            return None
        if now - fall_since["value"] < FALL_HOLD_S:
            return None
        fall_since["value"] = None
        respawn_requests.append((None, FALL_SOURCE))
        return root_z, up_z

    panda_runtime = None
    if args.panda_mode != "ros":
        from ros2_bridge.tools.panda_learning_runtime import PandaLearningRuntime

        panda_runtime = PandaLearningRuntime(
            sim.model,
            sim.data,
            task,
            dataset_dir=args.dataset_dir,
            instruction=args.task_instruction,
            rate_hz=args.panda_rate,
            teledex_port=args.teledex_port,
            position_scale=args.panda_position_scale,
            yaw_scale=args.panda_yaw_scale,
            pitch_scale=args.panda_pitch_scale,
            preview=args.panda_preview,
            controller=panda_controller,
        )

    soccer_mission = None
    room_permutation = None
    cone_permutation = None
    if args.scene.suffix.lower() == ".json":
        soccer_mission = SoccerMissionReset(sim.model, sim.data)
        cone_permutation = ConePermutationRuntime(sim.model, sim.data)
        if task is not None and panda_prefix:
            room_permutation = RoomPermutationRuntime(
                sim.model, sim.data, args.scene, args.assets_dir, task
            )

    from ros2_bridge.tools.mission_judge import MissionJudge

    if soccer_mission is not None and args.ball_friction is not None:
        soccer_mission.set_friction(args.ball_friction)
    RANDOM_RGBA = (0.35, 0.55, 1.0, 0.10)
    mission_judge = MissionJudge(
        sim.model, sim.data, g1_prefix=spawn.prefix, m2_color=scenario.m2_color,
        m1_goal=scenario.m1_goal, show_regions=args.judge_regions, m4_room=args.m4_room,
        regions=[(args.m1_goal_region, RANDOM_RGBA)]
        + ([(args.ball_region, RANDOM_RGBA)] if soccer_mission is not None else []),
        hide_targets=args.hide_targets,
        goal_radius=args.judge_goal_radius,
        cone_box=args.judge_cone_box, cone_offset=args.judge_cone_offset,
        room_centers=lambda: {
            name: ROOM_SLOT_CENTERS[slot] for slot, name in enumerate(
                room_permutation.current_order if room_permutation is not None
                else ROOM_SEMANTICS)
        },
        soccer=soccer_mission,
    )

    try:
        stop_requested = False

        def request_stop(_signum, _frame) -> None:
            nonlocal stop_requested
            stop_requested = True

        # Raising KeyboardInterrupt inside mj_step callbacks causes a MuJoCo error.
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        with SimPublisher(
            sim.model,
            prefix=spawn.prefix,
            camera_hz=args.camera_hz,
            lidar_hz=args.lidar_hz,
            lidar=args.lidar,
            lidar_threads=args.lidar_threads,
            imu_hz=args.imu_hz,
            odom_hz=args.odom_hz,
            publish_tf=args.odom_tf,
            footprint_frame="base_footprint" if args.footprint else None,
            odom_twist_window=args.odom_twist_window,
            unitree_odom=args.unitree_odom,
            unitree_odom_hz=args.unitree_odom_hz,
            unitree_odom_lf_hz=args.unitree_odom_lf_hz,
            unitree_dds_domain=args.unitree_dds_domain,
            unitree_dds_interface=args.unitree_dds_interface,
            lidar_columns=args.lidar_columns,
            cmd_vel_callback=command_callback,
            cmd_vel_topic=args.cmd_vel_topic,
            cmd_timeout=args.cmd_timeout,
            cmd_limits=(args.max_vx, args.max_vy, args.max_wz),
            cmd_bias=args.cmd_bias,
            extra_joint_prefixes={"panda": panda_prefix} if panda_prefix else None,
            extra_joint_commanders={"panda": panda_controller} if panda_controller else None,
            extra_rgb_cameras={
                "panda/agentview": (
                    "panda/agentview", "panda/agentview", 256, 256
                ),
                "panda/wrist_camera": (
                    "panda/robot0_eye_in_hand",
                    "panda/robot0_eye_in_hand",
                    256,
                    256,
                ),
            } if panda_prefix else None,
            extra_rgb_hz=(
                args.panda_camera_hz
                if panda_prefix and panda_runtime is None
                else 0.0
            ),
        ) as pub:
            from std_msgs.msg import Empty

            def receive_g1_respawn(_message: Empty) -> None:
                # Mutate simulation state only in the simulation loop under the viewer lock.
                respawn_requests.append((None, "ROS"))

            g1_respawn_subscription = pub.node.create_subscription(
                Empty, args.g1_respawn_topic, receive_g1_respawn, 10
            )
            pub.node.get_logger().info(
                f"G1 rescue: keys 1-6 in viewer or publish std_msgs/Empty to "
                f"{args.g1_respawn_topic}"
            )
            joint_command_subscription = None
            if joint_command_switch is not None:
                from sensor_msgs.msg import JointState as _JointCommand

                def receive_joint_command(message) -> None:
                    try:
                        joint_command_switch.command(message.name, message.position)
                    except ValueError as error:
                        pub.node.get_logger().warning(
                            f"ignored {args.joint_command_topic}: {error}",
                            throttle_duration_sec=2.0)
                        return
                    g1_motion_command_seen["value"] = True

                def receive_joint_gains(message) -> None:
                    try:
                        joint_command_switch.set_gains(message.name, message.position, message.velocity)
                    except ValueError as error:
                        pub.node.get_logger().warning(f"ignored {args.joint_gains_topic}: {error}")

                joint_command_subscription = [
                    pub.node.create_subscription(
                        _JointCommand, args.joint_command_topic, receive_joint_command, 10),
                    pub.node.create_subscription(
                        _JointCommand, args.joint_gains_topic, receive_joint_gains, 10),
                ]
                pub.node.get_logger().info(
                    f"G1 joint control: sensor_msgs/JointState on {args.joint_command_topic} "
                    f"suspends the walking policy; it resumes {args.joint_command_timeout:g} s "
                    f"(sim) after the last command; gains on {args.joint_gains_topic}"
                )
            from std_msgs.msg import Int32

            status = MissionStatusPublisher(pub.node, args.mission_status_topic)

            status.common = mission_judge.public

            judge_requests: deque[int] = deque(maxlen=8)
            mission_end_subscription = pub.node.create_subscription(
                Int32, args.g1_mission_end_topic,
                lambda message: judge_requests.append(int(message.data)), 10,
            )
            region_toggle_requests: deque[str] = deque(maxlen=2)
            random_toggle_requests: deque[str] = deque(maxlen=2)
            pub.node.get_logger().info(
                f"referee: std_msgs/Int32 {args.g1_mission_end_topic} judges mission N on "
                f"ground truth; mission 4 room {args.m4_room}; scenario "
                + (str(args.scenario) if args.scenario else "stock layout")
            )

            zone_body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "panda_task_zone_visuals")
            zone_geoms = np.flatnonzero(sim.model.geom_bodyid == zone_body) if zone_body >= 0 else np.array([], int)
            zone_alpha = sim.model.geom_rgba[zone_geoms, 3].copy()
            zone_sites = [sid for name in ("panda_task_pencil_spawn", "panda_task_cup_drop_target")
                          if (sid := mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, name)) >= 0]
            site_alpha = {sid: float(sim.model.site_rgba[sid, 3]) for sid in zone_sites}

            def show_task_zones(visible: bool) -> None:
                sim.model.geom_rgba[zone_geoms, 3] = zone_alpha if visible else 0.0
                for sid in zone_sites:
                    sim.model.site_rgba[sid, 3] = site_alpha[sid] if visible else 0.0

            show_task_zones(mission_judge.show_random)

            def process_judge_requests() -> None:
                while region_toggle_requests:
                    region_toggle_requests.popleft()
                    mission_judge.show_regions = not mission_judge.show_regions
                    pub.node.get_logger().info(
                        "referee success regions " + ("shown" if mission_judge.show_regions else "hidden"))
                while random_toggle_requests:
                    random_toggle_requests.popleft()
                    mission_judge.show_random = not mission_judge.show_random
                    show_task_zones(mission_judge.show_random)
                    pub.node.get_logger().info(
                        "random spawn regions (mission 1 goal, ball, Panda task zones) "
                        + ("shown" if mission_judge.show_random else "hidden"))
                while judge_requests:
                    index = judge_requests.popleft()
                    verdict = mission_judge.judge(index, sim.data)
                    label = mission_spawns[index][0] if index in mission_spawns else f"mission {index}"
                    status.mission("judged", index, label, mission_judge.pelvis(sim.data), "referee",
                                   float(sim.data.time), extra={"judge": verdict})
                    outcome = {True: "PASS", False: "FAIL"}.get(verdict["success"], "N/A")
                    pub.node.get_logger().info(
                        f"referee: mission {index} {outcome} -- {verdict['summary']}")

            mission_begin_requests: deque[tuple[int, str]] = deque(maxlen=4)

            def receive_mission_begin(message: Int32) -> None:
                index = int(message.data)
                if index in mission_spawns:
                    mission_begin_requests.append((index, f"ROS begin {index}"))
                else:
                    pub.node.get_logger().warning(
                        f"ignored mission index {index}; valid: {sorted(mission_spawns)}"
                    )

            mission_begin_subscription = pub.node.create_subscription(
                Int32, args.g1_mission_begin_topic, receive_mission_begin, 10
            )

            def process_mission_begin_requests() -> None:
                process_judge_requests()
                while mission_begin_requests:
                    index, source = mission_begin_requests.popleft()
                    active_spawn["index"] = index
                    mission_judge.mission_started(index)
                    if index == 6 and soccer_mission is not None:
                        soccer_mission.reset(sim.data)
                    label, point, point_yaw = mission_spawns[index]
                    # No coordinates: mission 5's point follows the classroom and
                    # would tell which room that is (mission 4's answer).
                    status.mission("selected", index, label, None, source, float(sim.data.time))
                    pub.node.get_logger().info(
                        f"mission {index} ({label}) selected from {source}; G1 not moved"
                        + ("; soccer ball restored" if index == 6 and soccer_mission is not None else "")
                    )

            from geometry_msgs.msg import PoseWithCovarianceStamped
            from ros2_bridge.tools.ros_publisher import to_ros_time

            initial_pose_pub = (
                pub.node.create_publisher(PoseWithCovarianceStamped, args.initial_pose_topic, 10)
                if args.auto_initial_pose else None
            )
            respawned_pub = pub.node.create_publisher(Empty, args.respawned_topic, 10)
            localize_requests: deque[str] = deque(maxlen=4)
            # AMCL ignores seeds before activation; retry until explicitly localized.
            amcl_watch = {"subscribers": 0, "checked": 0.0, "due": []}

            def receive_localize(_message: Empty) -> None:
                if initial_pose_pub is None:
                    pub.node.get_logger().warning(
                        f"ignored {args.localize_topic}: start the simulator with --auto-initial-pose"
                    )
                    return
                localize_requests.append("ROS")
                # Cancel startup seeds after explicit localization because G1 may be moving.
                amcl_watch["due"].clear()

            localize_subscription = pub.node.create_subscription(
                Empty, args.localize_topic, receive_localize, 10
            )

            g1_body_set = set(int(b) for b in g1_body_ids)
            g1_free_qadr = next(
                (int(sim.model.jnt_qposadr[j]) for j in range(sim.model.njnt)
                 if sim.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                 and int(sim.model.jnt_bodyid[j]) in g1_body_set),
                None,
            )

            def g1_planar_pose() -> tuple[float, float, float]:
                qadr = walker.root_qadr if walker is not None else g1_free_qadr
                w, qx, qy, qz = (float(v) for v in sim.data.qpos[qadr + 3:qadr + 7])
                yaw_now = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                return float(sim.data.qpos[qadr]), float(sim.data.qpos[qadr + 1]), yaw_now

            def process_localize_requests() -> None:
                if initial_pose_pub is None:
                    return
                now = time.monotonic()
                if now - amcl_watch["checked"] >= 1.0:
                    amcl_watch["checked"] = now
                    count = initial_pose_pub.get_subscription_count()
                    if count > amcl_watch["subscribers"]:
                        amcl_watch["due"] = [now + 3.0, now + 8.0]
                    amcl_watch["subscribers"] = count
                if amcl_watch["due"] and now >= amcl_watch["due"][0]:
                    amcl_watch["due"].pop(0)
                    localize_requests.append("AMCL start")
                if not localize_requests:
                    return
                source = localize_requests.pop()
                localize_requests.clear()
                x, y, pose_yaw = g1_planar_pose()
                message = PoseWithCovarianceStamped()
                message.header.frame_id = "map"
                message.header.stamp = to_ros_time(float(sim.data.time))
                message.pose.pose.position.x, message.pose.pose.position.y = x, y
                message.pose.pose.orientation.z = math.sin(pose_yaw / 2.0)
                message.pose.pose.orientation.w = math.cos(pose_yaw / 2.0)
                message.pose.covariance[0] = message.pose.covariance[7] = 0.25
                message.pose.covariance[35] = 0.06853891945200942
                initial_pose_pub.publish(message)
                index = active_spawn["index"]
                status.mission("initial_pose", index, mission_spawns[index][0], (x, y), source,
                               float(sim.data.time), yaw=pose_yaw)
                pub.node.get_logger().info(
                    f"initial pose ({source}): ({x:.2f}, {y:.2f}) yaw "
                    f"{math.degrees(pose_yaw):+.1f} deg on {args.initial_pose_topic}"
                )

            def receive_randomize(_message: Empty) -> None:
                scenario_randomize_requests.append("ROS")

            randomize_subscription = pub.node.create_subscription(
                Empty, args.g1_randomize_topic, receive_randomize, 10
            )
            pub.node.get_logger().info(
                f"Mission API: std_msgs/Int32 {args.g1_mission_begin_topic} (1-6, G1 not moved), "
                f"std_msgs/Empty {args.g1_randomize_topic}; status on "
                f"{args.mission_status_topic}"
            )
            pub.node.get_logger().info(
                "ground truth: "
                + (("/odom + odom TF" if args.odom_tf else "/odom (no TF)")
                   + " published (--ground-truth)" if args.odom_hz > 0
                   else "NOT published (no /odom, no odom TF; add --ground-truth to enable)")
                + "; initial pose: "
                + (f"G1's true pose on {args.initial_pose_topic} when AMCL comes up, after "
                   f"respawns and on {args.localize_topic}" if initial_pose_pub is not None
                   else "manual (RViz 2D Pose Estimate)")
            )
            # The Panda base's world pose is NOT published: the desk moves with the
            # classroom, so it would tell which room that is (mission 4's answer).

            def perform_mission_respawn(
                checkpoint: int | None, source: str
            ) -> tuple[int, str, np.ndarray]:
                if isinstance(checkpoint, tuple):
                    return respawn_g1(None, pose=checkpoint)
                selected = active_spawn["index"] if checkpoint is None else checkpoint
                mission_judge.mission_started(checkpoint)
                if checkpoint is None:
                    # A respawn at the start point (fall, gap, key B): cone order, room order
                    # and ball are drawn again.
                    room_order, cone_order = randomize_scenario(source, respawn=True)
                    pub.node.get_logger().info(
                        f"Scenario after {source}: {current_scenario().summary()}")
                elif selected == 6 and soccer_mission is not None:
                    soccer_mission.reset(sim.data)
                index, label, target_pos = respawn_g1(checkpoint)
                return index, label, target_pos

            env_rng = np.random.default_rng()

            def rooms_moved(moved_panda, classroom_delta_y: float) -> None:
                label, point5, point5_yaw = mission_spawns[5]
                moved_point5 = point5.copy()
                moved_point5[1] = 19.55 + classroom_delta_y
                mission_spawns[5] = (label, moved_point5, point5_yaw)
                if panda_runtime is not None:
                    panda_runtime.workspace_relocated()
                elif panda_controller is not None:
                    if panda_controller.mode == "osc":
                        panda_controller.set_eef_target(
                            panda_controller.site_transform(sim.data)
                        )
                    else:
                        panda_controller.set_joint_target(
                            sim.data.qpos[panda_controller.qpos_adrs]
                        )

            def randomize_scenario(source: str, respawn: bool = False) -> tuple[
                tuple[str, ...] | None, tuple[str, ...] | None
            ]:
                """``respawn``: only the cone order, room order and ball are drawn again (the mission 1
                goal, mission 2 colour and pencil/holder placement stay)."""

                room_order = None
                cone_order = None
                if room_permutation is not None:
                    room_order, moved_panda, classroom_delta_y = room_permutation.randomize()
                    rooms_moved(moved_panda, classroom_delta_y)
                if cone_permutation is not None:
                    cone_order = cone_permutation.randomize()
                from ros2_bridge.tools.mission_judge import COLORS

                if not respawn:
                    mission_judge.set_m2_color(COLORS[int(env_rng.integers(len(COLORS)))])
                    gx, gy = sample_point(env_rng, args.m1_goal_region)
                    mission_judge.set_m1_goal_world(gx, gy, args.m1_goal_yaw)
                if soccer_mission is not None:
                    soccer_mission.set_home(sim.data, *sample_point(env_rng, args.ball_region))
                if task is not None and not respawn:
                    reset_panda_task(source)
                write_scenario(source)
                return room_order, cone_order

            def task_bodies() -> tuple[int, int]:
                return (
                    _prefixed_root_body(sim.model, "editor/panda_task_random_pencil/"),
                    _prefixed_root_body(sim.model, "editor/school_room7_empty_desk_pencil_cup/"),
                )

            def current_scenario() -> Scenario:
                env = Scenario()
                yaw = mission_judge.goal_yaw if mission_judge.goal_yaw is not None else args.m1_goal_yaw
                env.m1_goal = [*mission_judge.goal_xy, float(yaw)]
                env.m2_color = mission_judge.m2_color
                if cone_permutation is not None:
                    env.cone_order = list(cone_permutation.current_order)
                if room_permutation is not None:
                    env.room_order = list(room_permutation.current_order)
                if soccer_mission is not None:
                    env.ball_xy = list(soccer_mission.home_xy)
                if task is not None:
                    pencil_body, cup_body = task_bodies()
                    joint = int(sim.model.body_jntadr[pencil_body])
                    qadr = int(sim.model.jnt_qposadr[joint])
                    pencil = sim.data.qpos[qadr:qadr + 7]
                    env.panda_task = {
                        "pencil_xy": [float(pencil[0]), float(pencil[1])],
                        "pencil_yaw_deg": quat_yaw_deg(pencil[3:7]),
                        "cup_xy": [float(v) for v in sim.model.body_pos[cup_body][:2]],
                        "cup_yaw_deg": quat_yaw_deg(sim.model.body_quat[cup_body]),
                    }
                return env

            def apply_scenario(env: Scenario, source: str) -> None:
                if room_permutation is not None and tuple(env.room_order) != room_permutation.current_order:
                    _order, moved_panda, classroom_delta_y = room_permutation.apply(env.room_order)
                    rooms_moved(moved_panda, classroom_delta_y)
                if cone_permutation is not None:
                    cone_permutation.apply(env.cone_order)
                mission_judge.set_m2_color(env.m2_color)
                mission_judge.set_m1_goal_world(*env.m1_goal)
                if soccer_mission is not None:
                    soccer_mission.set_home(sim.data, *env.ball_xy)
                if task is not None and env.panda_task:
                    reset_panda_task(source, placement=env.panda_task)
                write_scenario(source)

            def write_scenario(source: str) -> None:
                if args.scenario_out is not None:
                    current_scenario().save(args.scenario_out, f"from the simulator after {source}")

            task_rng = np.random.default_rng()

            def reset_panda_task(source: str, placement: dict | None = None) -> bool:
                if task is None or panda_controller is None:
                    pub.node.get_logger().warning(
                        f"{source}: this scene has no Panda pencil/holder task"
                    )
                    return False
                controller = panda_controller
                arm_home, finger_home = panda_home
                sim.data.qpos[controller.qpos_adrs] = arm_home
                sim.data.qpos[controller.finger_qpos_adrs] = finger_home
                sim.data.qvel[controller.dof_adrs] = 0.0
                sim.data.qvel[sim.model.jnt_dofadr[controller.finger_joint_ids]] = 0.0
                bodies = task_bodies()
                if placement is None:
                    combined.respawn_panda_task(sim.model, sim.data, task, *bodies, task_rng)
                else:
                    combined.place_panda_task(
                        sim.model, sim.data, task, *bodies, placement["pencil_xy"],
                        placement["pencil_yaw_deg"], placement["cup_xy"], placement["cup_yaw_deg"],
                    )
                if room_permutation is not None:
                    room_permutation.rebase_task_objects(bodies)
                controller.nullspace_target = arm_home.copy()
                controller.set_gripper_opening(0.04)
                if controller.mode == "osc":
                    controller.set_eef_target(controller.site_transform(sim.data))
                else:
                    controller.set_joint_target(arm_home)
                if panda_runtime is not None:
                    panda_runtime.workspace_relocated()
                pencil, cup = task["pencil_xy"], task["cup_xy"]
                pub.node.get_logger().info(
                    f"Panda task ({source}): arm home, pencil ({pencil[0]:.3f}, "
                    f"{pencil[1]:.3f}), holder ({cup[0]:.3f}, {cup[1]:.3f})"
                )
                return True

            def drain_randomize_requests() -> None:
                if not scenario_randomize_requests:
                    return
                source = scenario_randomize_requests.pop()
                scenario_randomize_requests.clear()
                room_order, cone_order = randomize_scenario(source)
                env = current_scenario()
                index = active_spawn["index"]
                label, point, point_yaw = mission_spawns[index]
                status.mission("randomized", index, label, None, source, float(sim.data.time))
                pub.node.get_logger().info(f"Scenario after {source}: {env.summary()}")
                if room_order is not None:
                    pub.node.get_logger().info(
                        "Room order after " + source
                        + " (upper -> lower): " + " | ".join(room_order)
                        + "; G1 pose preserved"
                    )
                if cone_order is not None:
                    pub.node.get_logger().info(
                        "Cone colour order after " + source
                        + " (left -> right): " + " | ".join(cone_order)
                        + "; G1 pose preserved"
                    )

            def announce_respawn(index: int, label: str, target_pos, source: str) -> None:
                # counted: a respawn to the start point (fall, key B, /g1/respawn); the mission
                # runner counts these against missions.falls_allowed.
                status.mission("respawned", index, label, target_pos, source, float(sim.data.time),
                               yaw=0.0 if label == START_LABEL else mission_spawns[index][2],
                               extra={"counted": label == START_LABEL})
                if label == START_LABEL:
                    respawned_pub.publish(Empty())
                if initial_pose_pub is not None:
                    localize_requests.append(f"respawn {index}")
                pub.node.get_logger().info(
                    f"G1 respawned from {source}: point {index} "
                    f"({label}) at ({target_pos[0]:.2f}, "
                    f"{target_pos[1]:.2f}, {target_pos[2]:.2f}); "
                    "pose, velocity, policy state and command cleared"
                )
                if index == 6 and label != START_LABEL:
                    pub.node.get_logger().info(
                        "Mission 6 soccer goal and ball restored to their initial poses"
                    )

            apply_scenario(scenario, "startup")
            pub.node.get_logger().info(
                "Scenario (" + (str(args.scenario) if args.scenario else "stock layout")
                + "): " + current_scenario().summary()
            )
            status.mission(
                "startup", active_spawn["index"], mission_spawns[active_spawn["index"]][0],
                mission_spawns[active_spawn["index"]][1], "startup", float(sim.data.time),
                yaw=mission_spawns[active_spawn["index"]][2],
            )

            pub.node.get_logger().info(
                f"scene={args.scene.name} robot={spawn.prefix!r} "
                f"spawn=({x:.2f}, {y:.2f}, floor {floor_z:.2f}); {state_text}"
            )
            started = time.perf_counter()
            if args.view:
                import mujoco.viewer
                try:
                    from mujoco.glfw import glfw

                    camera_request = {"mode": "g1", "dirty": False}
                    place_mode = {"armed": False, "seen": None}

                    def selection() -> tuple:
                        pert = viewer.perturb
                        return (int(pert.select), tuple(float(v) for v in pert.localpos))

                    def selected_world_point() -> np.ndarray:
                        pert = viewer.perturb
                        body = int(pert.select)
                        rotation = sim.data.xmat[body].reshape(3, 3)
                        return sim.data.xpos[body] + rotation @ np.asarray(pert.localpos, dtype=float)

                    def poll_place_mode() -> None:
                        if not place_mode["armed"] or selection() == place_mode["seen"]:
                            return
                        point = selected_world_point()
                        place_mode["armed"] = False
                        _x, _y, yaw_now = g1_planar_pose()
                        respawn_requests.append(((float(point[0]), float(point[1]), yaw_now), "key G"))

                    frames_shown = {"value": False}
                    start_frame_xy = (float(respawn_pos[0]), float(respawn_pos[1]))

                    def door_markers() -> list[tuple[float, float, str]]:
                        order = (room_permutation.current_order if room_permutation is not None
                                 else tuple(ROOM_SEMANTICS))
                        marks = [(18.0, ROOM_SLOT_CENTERS[0], f"{order[0]} door (18.0, 17.5)"),
                                 (18.0, ROOM_SLOT_CENTERS[1], f"{order[1]} door (18.0, 10.5)"),
                                 (18.0, 3.59, f"{order[2]} door (18.0, 3.59)"),
                                 (15.5, 21.0, "cone room door (15.5, 21.0)"),
                                 (13.0, 20.125, "cliff entrance (13.0, 20.125)"),
                                 (13.0, 11.375, "cliff exit (13.0, 11.375)"),
                                 (13.0, 9.1, "soccer room door (13.0, 9.1)")]
                        if cone_permutation is not None:
                            for color, body in cone_permutation._bodies.items():
                                x, y = (float(v) for v in sim.data.xpos[body][:2])
                                marks.append((x, y, f"{color} cone ({x:.2f}, {y:.2f})"))
                        return marks

                    def draw_frames(scene) -> None:
                        if not frames_shown["value"]:
                            return
                        for x, y, label in door_markers():
                            if scene.ngeom >= scene.maxgeom:
                                return
                            geom = scene.geoms[scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.08, 0, 0]),
                                                np.array([x, y, 0.08]), np.eye(3).flatten(),
                                                np.array([1.0, 0.85, 0.1, 1.0], np.float32))
                            geom.label = label
                            scene.ngeom += 1
                        for (ox, oy), label in ((0.0, 0.0), "world origin (0, 0)"), (start_frame_xy,
                                                 f"G1 start ({start_frame_xy[0]:.0f}, {start_frame_xy[1]:.0f})"):
                            for axis, rgba in ((0, (1.0, 0.1, 0.1, 1.0)), (1, (0.1, 1.0, 0.1, 1.0)),
                                               (2, (0.1, 0.3, 1.0, 1.0))):
                                if scene.ngeom >= scene.maxgeom:
                                    return
                                tip = np.array([ox, oy, 0.02])
                                tip[axis] += 1.0
                                geom = scene.geoms[scene.ngeom]
                                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                                    np.zeros(3), np.eye(3).flatten(),
                                                    np.asarray(rgba, np.float32))
                                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.03,
                                                     np.array([ox, oy, 0.02]), tip)
                                if axis == 2:
                                    geom.label = label
                                scene.ngeom += 1
                    mission_keys = {
                        glfw.KEY_1: 1,
                        glfw.KEY_2: 2,
                        glfw.KEY_3: 3,
                        glfw.KEY_4: 4,
                        glfw.KEY_5: 5,
                        glfw.KEY_6: 6,
                        glfw.KEY_KP_1: 1,
                        glfw.KEY_KP_2: 2,
                        glfw.KEY_KP_3: 3,
                        glfw.KEY_KP_4: 4,
                        glfw.KEY_KP_5: 5,
                        glfw.KEY_KP_6: 6,
                    }
                    def on_view_key(key: int) -> None:
                        if key in mission_keys:
                            checkpoint = mission_keys[key]
                            respawn_requests.append((checkpoint, f"key {checkpoint}"))
                        elif key == glfw.KEY_B:
                            respawn_requests.append((None, "key B"))
                        elif key == glfw.KEY_G:
                            place_mode["armed"] = not place_mode["armed"]
                            place_mode["seen"] = selection()
                            pub.node.get_logger().info(
                                "key G: double-click where G1 should stand (G again cancels)"
                                if place_mode["armed"] else "key G: placing cancelled")
                        elif key == glfw.KEY_R:
                            scenario_randomize_requests.append("key R")
                        elif key == glfw.KEY_J:
                            region_toggle_requests.append("key J")
                        elif key == glfw.KEY_K:
                            random_toggle_requests.append("key K")
                        elif key == glfw.KEY_F:
                            frames_shown["value"] = not frames_shown["value"]

                    with mujoco.viewer.launch_passive(
                        sim.model, sim.data,
                        show_left_ui=False, show_right_ui=True,
                        key_callback=on_view_key,
                    ) as viewer:
                        # Start pacing after viewer creation to avoid catching up to window setup time.
                        set_g1_focus_camera(viewer.cam, sim.model, sim.data, spawn.prefix)
                        default_flags = viewer.opt.flags.copy()
                        default_frame, default_label = int(viewer.opt.frame), int(viewer.opt.label)
                        # Native number keys also toggle geom groups; restore them after mission respawns.
                        viewer.opt.geomgroup[0] = 1
                        viewer.opt.geomgroup[1] = 1
                        viewer.opt.geomgroup[2] = 1
                        viewer.opt.geomgroup[3] = 0
                        viewer.opt.geomgroup[4] = 0
                        viewer.opt.geomgroup[5] = 1
                        viewer.opt.sitegroup[5] = 1
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
                        default_flags = viewer.opt.flags.copy()
                        mission_judge.draw(viewer.user_scn, sim.data)
                        viewer.sync()
                        pub.node.get_logger().info(
                            "Viewer keys: 1-6 = G1 to a mission's start point (1 launch, "
                            "2 cone door, 3 cliff entrance, 4 launch facing the rooms, "
                            "5 Panda desk, 6 soccer door); B = respawn G1 at the start (16, 11), "
                            "counted like a fall; G then a double-click = G1 to that "
                            "point; R = new random scenario "
                            "(goal, cones, rooms, ball, pencil; G1 not moved); "
                            "J = success regions on/off, K = random spawn regions on/off, "
                            "F = coordinate frames (world origin, G1 start), door centres and cone "
                            "centres on/off (viewer only)"
                        )
                        started = time.perf_counter()
                        sim_started = float(sim.data.time)
                        pace_started = started
                        pace_sim_started = sim_started
                        previous_sim_time = sim_started
                        next_sync = started + 1.0 / 30.0
                        applied_camera = "g1"
                        auto_focused = False
                        while rclpy.ok() and not stop_requested and viewer.is_running():
                            with viewer.lock():
                                poll_place_mode()
                                drain_randomize_requests()
                                if respawn_requests:
                                    checkpoint, source = respawn_requests.pop()
                                    respawn_requests.clear()
                                    index, label, target_pos = (
                                        perform_mission_respawn(checkpoint, source)
                                    )
                                    if not isinstance(checkpoint, tuple):
                                        camera_request["mode"] = "g1"
                                        camera_request["dirty"] = True
                                    announce_respawn(index, label, target_pos, source)
                                process_mission_begin_requests()
                                process_localize_requests()
                                if panda_runtime is not None:
                                    panda_runtime.before_step(sim.data)
                                elif panda_controller is not None:
                                    panda_controller.before_step(sim.data)
                                sim.step()
                                recovered_pose = recover_unready_g1()
                                if recovered_pose is not None:
                                    pub.node.get_logger().warning(
                                        "G1 lost its standby pose before receiving cmd_vel; "
                                        f"auto-respawned (pelvis_z={recovered_pose[0]:.3f}, "
                                        f"up_z={recovered_pose[1]:.3f})"
                                    )
                                mission_judge.track(sim.data)
                                fell = check_fall()
                                if fell is not None:
                                    pub.node.get_logger().warning(
                                        f"G1 fell over (pelvis z {fell[0]:.2f} m, up_z {fell[1]:.2f}); "
                                        "respawning at launch point (16, 11)"
                                    )
                                pub.update(sim.data)
                                # Lock camera updates to prevent races with MuJoCo temporary-stack use.
                                if panda_runtime is not None:
                                    observation = panda_runtime.after_step(sim.data)
                                    if observation is not None and args.panda_camera_hz > 0:
                                        pub.publish_extra_rgb_frames(
                                            {
                                                "panda/agentview": observation["image"],
                                                "panda/wrist_camera": observation["wrist_image"],
                                            },
                                            float(observation["time"]),
                                        )
                            if panda_runtime is not None and panda_runtime.stop_requested:
                                break
                            now_wall = time.perf_counter()
                            current_sim_time = float(sim.data.time)
                            if current_sim_time + 1e-9 < previous_sim_time:
                                # Rebase wall time when episode reset rewinds data.time.
                                pace_started = now_wall
                                pace_sim_started = current_sim_time
                            previous_sim_time = current_sim_time
                            if (
                                panda_prefix and args.auto_focus_panda_distance > 0
                                and not auto_focused
                            ):
                                g1_xy = sim.data.qpos[walker.root_qadr:walker.root_qadr + 2] if walker else spawn.pos[:2]
                                panda_xy = (
                                    room_permutation.panda_position[:2]
                                    if room_permutation is not None
                                    else (args.panda_x, args.panda_y)
                                )
                                if math.dist(g1_xy, panda_xy) <= args.auto_focus_panda_distance:
                                    camera_request["mode"] = "panda"
                                    camera_request["dirty"] = True
                                    auto_focused = True
                                    pub.node.get_logger().info(
                                        "G1 entered the Panda work radius; switched to desk view "
                                        "(all free: F1=G1, F2=Panda, F3=overview)"
                                    )
                            if camera_request["mode"] in ("follow", "chase"):
                                if camera_request["mode"] == "follow":
                                    set_g1_focus_camera(
                                        viewer.cam, sim.model, sim.data, spawn.prefix
                                    )
                                else:
                                    set_g1_chase_camera(
                                        viewer.cam, sim.model, sim.data, spawn.prefix
                                    )
                                applied_camera = camera_request["mode"]
                                camera_request["dirty"] = False
                            elif (
                                camera_request["dirty"]
                                or camera_request["mode"] != applied_camera
                            ):
                                if camera_request["mode"] == "g1":
                                    set_g1_focus_camera(
                                        viewer.cam, sim.model, sim.data, spawn.prefix
                                    )
                                elif camera_request["mode"] == "panda":
                                    panda_xy = (
                                        room_permutation.panda_position[:2]
                                        if room_permutation is not None
                                        else (args.panda_x, args.panda_y)
                                    )
                                    set_panda_table_camera(viewer.cam, panda_xy)
                                else:
                                    set_scene_overview_camera(
                                        viewer.cam, sim.model, sim.data, scene_ngeom
                                    )
                                applied_camera = camera_request["mode"]
                                camera_request["dirty"] = False
                            if now_wall >= next_sync:
                                # Restore render flags and geom groups changed by native keyboard shortcuts.
                                viewer.opt.geomgroup[0] = 1
                                viewer.opt.geomgroup[1] = 1
                                viewer.opt.geomgroup[2] = 1
                                viewer.opt.geomgroup[3] = 0
                                viewer.opt.geomgroup[4] = 0
                                viewer.opt.geomgroup[5] = 1
                                viewer.opt.sitegroup[5] = 1
                                viewer.opt.flags[:] = default_flags
                                viewer.opt.frame, viewer.opt.label = default_frame, default_label
                                mission_judge.draw(viewer.user_scn, sim.data)
                                draw_frames(viewer.user_scn)
                                viewer.sync()
                                next_sync = time.perf_counter() + 1.0 / 30.0
                            ahead = ((float(sim.data.time) - pace_sim_started) / args.rtf
                                     - (time.perf_counter() - pace_started))
                            if ahead > 0:
                                time.sleep(ahead)
                            if args.duration and sim.data.time >= args.duration:
                                break
                except KeyboardInterrupt:
                    pass
            else:
                try:
                    pace_started = started
                    pace_sim_started = float(sim.data.time)
                    previous_sim_time = pace_sim_started
                    while rclpy.ok() and not stop_requested:
                        drain_randomize_requests()
                        if respawn_requests:
                            checkpoint, source = respawn_requests.pop()
                            respawn_requests.clear()
                            index, label, target_pos = (
                                perform_mission_respawn(checkpoint, source)
                            )
                            announce_respawn(index, label, target_pos, source)
                        process_mission_begin_requests()
                        process_localize_requests()
                        if panda_runtime is not None:
                            panda_runtime.before_step(sim.data)
                        elif panda_controller is not None:
                            panda_controller.before_step(sim.data)
                        sim.step()
                        recovered_pose = recover_unready_g1()
                        if recovered_pose is not None:
                            pub.node.get_logger().warning(
                                "G1 lost its standby pose before receiving cmd_vel; "
                                f"auto-respawned (pelvis_z={recovered_pose[0]:.3f}, "
                                f"up_z={recovered_pose[1]:.3f})"
                            )
                        mission_judge.track(sim.data)
                        fell = check_fall()
                        if fell is not None:
                            pub.node.get_logger().warning(
                                f"G1 fell over (pelvis z {fell[0]:.2f} m, up_z {fell[1]:.2f}); "
                                "respawning at launch point (16, 11)"
                            )
                        pub.update(sim.data)
                        if panda_runtime is not None:
                            observation = panda_runtime.after_step(sim.data)
                            if observation is not None and args.panda_camera_hz > 0:
                                pub.publish_extra_rgb_frames(
                                    {
                                        "panda/agentview": observation["image"],
                                        "panda/wrist_camera": observation["wrist_image"],
                                    },
                                    float(observation["time"]),
                                )
                            if panda_runtime.stop_requested:
                                break
                        if args.realtime:
                            now_wall = time.perf_counter()
                            current_sim_time = float(sim.data.time)
                            if current_sim_time + 1e-9 < previous_sim_time:
                                pace_started = now_wall
                                pace_sim_started = current_sim_time
                            previous_sim_time = current_sim_time
                            ahead = ((current_sim_time - pace_sim_started) / args.rtf
                                     - (now_wall - pace_started))
                            if ahead > 0:
                                time.sleep(ahead)
                        if args.duration and sim.data.time >= args.duration:
                            break
                except KeyboardInterrupt:
                    pass
            wall = time.perf_counter() - started
            pub.node.get_logger().info(
                f"stopped at sim={sim.data.time:.2f}s wall={wall:.2f}s "
                f"RTF={sim.data.time / max(wall, 1e-9):.2f}"
            )
    finally:
        if "previous_sigint" in locals():
            signal.signal(signal.SIGINT, previous_sigint)
        if walker is not None:
            walker.set_command(0.0, 0.0, 0.0)
            walker.uninstall()
        if panda_runtime is not None:
            panda_runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
