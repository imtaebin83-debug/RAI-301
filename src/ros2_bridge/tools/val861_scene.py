"""Compile and view the saved val_861 layout with Unitree G1 and LIBERO Panda."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import secrets
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from environment.test_zone.src import layout_loader as EDITOR  # noqa: E402
from ros2_bridge.tools import g1_sensor_rig as G1_SENSOR_RIG  # noqa: E402
from ros2_bridge.tools.interactive_sim import ROBOT_XML  # noqa: E402
from ros2_bridge.tools.panda_controller import (  # noqa: E402
    PANDA_HOME,
    PandaController,
    initialize_panda_state,
)

ENV_ROOT = SRC_ROOT / "environment"
DEFAULT_LAYOUT = ENV_ROOT / "test_zone" / "world" / "layout.json"
DEFAULT_CATALOG = ENV_ROOT / "test_zone" / "world" / "asset_catalog.json"
G1_PREFIX = "g1/"
PANDA_PREFIX = "panda/"
PANDA_MODEL = Path(ROBOT_XML["panda"])
# Keep collection and inference camera poses identical.
PANDA_AGENTVIEW_POS = [1.28, 0.0, 0.92]
PANDA_AGENTVIEW_QUAT = [0.66331780, 0.24496836, 0.24496836, 0.66331780]
PANDA_AGENTVIEW_FOVY = 46.0
TASK_DESK_ID = "school_room7_student_table_front_right"
TASK_CUP_ID = "school_room7_empty_desk_pencil_cup"
TASK_PENCIL_ID = "panda_task_random_pencil"
DESK_HALF_SIZE = np.array([0.423274, 0.884573])
DESK_TOP_Z = 0.8016
TASK_MIN_REACH = 0.28
TASK_MAX_REACH = 0.79
TASK_INNER_REACH_MARGIN = 0.08
TASK_OUTER_REACH_MARGIN = 0.12
TASK_PENCIL_EDGE_CLEARANCE = 0.19
TASK_CUP_EDGE_CLEARANCE = 0.13
TASK_DESK_PROFILE_BUFFER = 0.005
TASK_PENCIL_SPAWN_OFFSET_Z = 0.0151
TASK_REACH_PROFILE = {
    215: 0.43, 217: 0.44, 219: 0.45, 221: 0.46, 223: 0.48,
    225: 0.49, 227: 0.51, 229: 0.53, 231: 0.56, 233: 0.58,
    235: 0.61, 237: 0.64, 239: 0.68, 241: 0.72, 243: 0.77,
    245: 0.79, 247: 0.79, 249: 0.79, 251: 0.79, 253: 0.79,
    255: 0.79, 257: 0.79, 259: 0.79, 261: 0.79, 263: 0.79,
    265: 0.79, 267: 0.79, 269: 0.79, 271: 0.79, 273: 0.79,
    275: 0.79, 277: 0.79, 279: 0.79, 281: 0.79, 283: 0.79,
    285: 0.79, 287: 0.79, 289: 0.79, 291: 0.79, 293: 0.79,
    295: 0.79, 297: 0.77, 299: 0.72, 301: 0.68, 303: 0.64,
    305: 0.61, 307: 0.58, 309: 0.56, 311: 0.53, 313: 0.51,
    315: 0.49, 317: 0.48, 319: 0.46, 321: 0.45, 323: 0.44,
    325: 0.43,
}
TASK_PENCIL_ANGLES_DEG = (236.0, 269.0)
TASK_PENCIL_RADIUS = (0.37, 0.64)
TASK_PENCIL_YAW_DEG = (-30.0, 30.0)
# Limit yaw near the workspace edge to avoid the Panda wrist joint limit.
TASK_PENCIL_OUTER_YAW_RADIUS = 0.60
TASK_PENCIL_OUTER_YAW_DEG = (-20.0, 20.0)
TASK_CUP_ANGLES_DEG = (273.0, 305.0)
TASK_CUP_RADIUS = (0.38, 0.64)
TASK_OBJECT_MIN_SEPARATION = 0.26
MAZE_ROOM_BOUNDS = (5.5, 13.0, 10.5, 21.0)
CLIFF_PATH_ROUTE = np.asarray(
    [
        [13.0, 20.125], [6.075, 20.125], [6.075, 18.675],
        [12.425, 18.675], [12.425, 17.025], [6.075, 17.025],
        [6.075, 15.225], [12.425, 15.225], [12.425, 13.325],
        [6.075, 13.325], [6.075, 11.325], [12.425, 11.325],
        [13.0, 11.375],
    ],
    dtype=float,
)
CLIFF_SHORTCUTS = (
    (9.25, 19.25, 19.60, 0.35),
    (9.25, 17.60, 18.10, 0.50),
    (9.25, 15.80, 16.45, 0.65),
    (9.25, 13.90, 14.65, 0.75),
    (9.25, 11.90, 12.75, 0.85),
)


def copy_stand_pose(
    robot_model: mujoco.MjModel,
    combined_model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    stand_key_id = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_key_id < 0:
        raise RuntimeError("The G1 model does not contain the expected 'stand' keyframe.")

    stand_qpos = robot_model.key_qpos[stand_key_id]
    data.qpos[:] = combined_model.qpos0

    for robot_joint_id in range(robot_model.njnt):
        if robot_model.jnt_type[robot_joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue

        joint_name = mujoco.mj_id2name(
            robot_model, mujoco.mjtObj.mjOBJ_JOINT, robot_joint_id
        )
        combined_joint_id = mujoco.mj_name2id(
            combined_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{G1_PREFIX}{joint_name}",
        )
        if combined_joint_id < 0:
            raise RuntimeError(f"Attached G1 joint is missing: {joint_name}")

        source_qpos_adr = robot_model.jnt_qposadr[robot_joint_id]
        target_qpos_adr = combined_model.jnt_qposadr[combined_joint_id]
        data.qpos[target_qpos_adr] = stand_qpos[source_qpos_adr]

    for actuator_id in range(combined_model.nu):
        if combined_model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = combined_model.actuator_trnid[actuator_id, 0]
        data.ctrl[actuator_id] = data.qpos[combined_model.jnt_qposadr[joint_id]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--g1-x", type=float, default=17.2)
    parser.add_argument("--g1-y", type=float, default=17.5)
    parser.add_argument("--g1-z", type=float, default=0.0)
    parser.add_argument("--g1-yaw", type=float, default=0.0)
    parser.add_argument(
        "--g1-maze-start",
        action="store_true",
        help="Spawn G1 outside the Mission 3 upper entrance, facing west.",
    )
    parser.add_argument("--panda-x", type=float, default=21.18)
    parser.add_argument("--panda-y", type=float, default=20.05)
    parser.add_argument("--panda-z", type=float, default=0.8016)
    parser.add_argument("--panda-yaw", type=float, default=-90.0)
    parser.add_argument(
        "--original-g1-hands",
        action="store_true",
        help="Use the legacy asymmetric cached G1 instead of the symmetric-hand model.",
    )
    parser.add_argument(
        "--pin-g1",
        action="store_true",
        help="Remove G1's free joint for a stable controller/actuator demonstration.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Start a gentle actuator demo. Recommended together with --pin-g1.",
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        help="Reproduce one pencil/cup placement; omitted means a new random seed each run.",
    )
    parser.add_argument(
        "--no-task-randomization",
        action="store_true",
        help="Use deterministic seed 0 instead of generating a new task placement.",
    )
    parser.add_argument(
        "--show-task-zones",
        dest="show_task_zones",
        action="store_true",
        help="Draw the desk boundary, safe reach rings, and curved sampling zones (default).",
    )
    parser.add_argument(
        "--hide-task-zones",
        dest="show_task_zones",
        action="store_false",
        help="Hide Panda reach and random-spawn visualization.",
    )
    parser.add_argument(
        "--task-map",
        type=Path,
        help="Save a top-down PNG of the actual desk, reach limits, zones, and samples.",
    )
    parser.add_argument(
        "--maze-map",
        type=Path,
        help="Save a top-down PNG of the Mission 3 elevated cliff course.",
    )
    parser.add_argument(
        "--view",
        choices=("classroom", "overview", "nurse", "restaurant", "maze"),
        default="classroom",
        help="Initial MuJoCo camera preset (default: classroom).",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.set_defaults(show_task_zones=True)
    return parser.parse_args()


def resolve_assets_dir(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    env = os.environ.get("MLSPACES_ASSETS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    dedicated = Path.home() / ".cache" / "molmospaces-two-scenes" / "assets"
    if dedicated.is_dir():
        return dedicated.resolve()
    raise RuntimeError("Set MLSPACES_ASSETS_DIR or pass --assets-dir.")


def yaw_quaternion(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def remove_root_freejoint(spec: mujoco.MjSpec, root: mujoco.MjsBody) -> None:
    for joint in list(root.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(joint)


def _layout_item(layout: dict, instance_id: str) -> dict:
    matches = [item for item in layout["objects"] if item.get("id") == instance_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one layout object {instance_id!r}, found {len(matches)}")
    return matches[0]


def _sector(
    base_xy: np.ndarray,
    angles_deg: tuple[float, float],
    radius_limits: tuple[float, float],
    color: list[float],
    desk_bounds: tuple[float, float, float, float],
    edge_clearance: float,
) -> dict:
    safe_profile: dict[int, float] = {}
    x0, x1, y0, y1 = desk_bounds
    safe_bounds = (
        x0 + edge_clearance,
        x1 - edge_clearance,
        y0 + edge_clearance,
        y1 - edge_clearance,
    )
    profile_angles = [
        angle for angle in sorted(TASK_REACH_PROFILE)
        if angles_deg[0] <= angle <= angles_deg[1]
    ]
    for endpoint in (int(angles_deg[0]), int(angles_deg[1])):
        if endpoint not in profile_angles:
            profile_angles.append(endpoint)
    for angle_deg in sorted(profile_angles):
        angle = math.radians(angle_deg)
        direction = np.asarray([math.cos(angle), math.sin(angle)])
        limits: list[float] = []
        for coordinate, component, lower, upper in zip(
            base_xy, direction, safe_bounds[::2], safe_bounds[1::2]
        ):
            if component > 1e-9:
                limits.append((upper - coordinate) / component)
            elif component < -1e-9:
                limits.append((lower - coordinate) / component)
        desk_limit = min(value for value in limits if value >= 0.0)
        ik_limit = float(np.interp(
            angle_deg,
            np.asarray(sorted(TASK_REACH_PROFILE), dtype=float),
            np.asarray([TASK_REACH_PROFILE[key] for key in sorted(TASK_REACH_PROFILE)]),
        ))
        safe_profile[angle_deg] = min(
            radius_limits[1],
            ik_limit - TASK_OUTER_REACH_MARGIN,
            desk_limit - TASK_DESK_PROFILE_BUFFER,
        )
    safe_min_radius = max(
        radius_limits[0], TASK_MIN_REACH + TASK_INNER_REACH_MARGIN
    )
    if min(safe_profile.values()) <= safe_min_radius + 0.025:
        raise RuntimeError("Panda task sector has no useful safety-buffered area")
    return {
        "center": base_xy.copy(),
        "min_radius": safe_min_radius,
        "max_radius": TASK_MAX_REACH,
        "reach_profile": safe_profile,
        "angles_deg": angles_deg,
        "rgba": color,
        "edge_clearance": edge_clearance,
    }


def _sample_sector(rng: np.random.Generator, sector: dict) -> np.ndarray:
    angle = math.radians(rng.uniform(*sector["angles_deg"]))
    max_radius = _profile_radius(sector, math.degrees(angle))
    radius = math.sqrt(rng.uniform(sector["min_radius"] ** 2, max_radius**2))
    return np.asarray(sector["center"], dtype=float) + radius * np.array(
        [math.cos(angle), math.sin(angle)]
    )


def _profile_radius(sector: dict, angle_deg: float) -> float:
    profile = sector.get("reach_profile")
    if not profile:
        return float(sector["max_radius"])
    angles = np.asarray(sorted(float(angle) for angle in profile), dtype=float)
    radii = np.asarray([profile[angle] for angle in angles], dtype=float)
    return float(np.interp(angle_deg, angles, radii))


def _pencil_yaw_range(radius: float) -> tuple[float, float]:
    if radius > TASK_PENCIL_OUTER_YAW_RADIUS:
        return TASK_PENCIL_OUTER_YAW_DEG
    return TASK_PENCIL_YAW_DEG


def _assert_sector_fits_desk(sector: dict, desk_bounds: tuple[float, ...]) -> None:
    x0, x1, y0, y1 = desk_bounds
    clearance = float(sector.get("edge_clearance", 0.10))
    angles = np.linspace(*sector["angles_deg"], 65)
    for inner in (True, False):
        radians = np.radians(angles)
        radii = np.full_like(angles, sector["min_radius"]) if inner else np.asarray(
            [_profile_radius(sector, angle) for angle in angles]
        )
        points = np.asarray(sector["center"]) + radii[:, None] * np.column_stack(
            (np.cos(radians), np.sin(radians))
        )
        if not (
            np.all(points[:, 0] >= x0 + clearance)
            and np.all(points[:, 0] <= x1 - clearance)
            and np.all(points[:, 1] >= y0 + clearance)
            and np.all(points[:, 1] <= y1 - clearance)
        ):
            raise RuntimeError(
                "The Panda base was moved so its exact task sectors no longer fit "
                "on the target desk. Move the base with the desk or adjust the sectors."
            )


def randomize_panda_task(layout: dict, args: argparse.Namespace) -> dict:
    desk = _layout_item(layout, TASK_DESK_ID)
    center = np.asarray(desk["pos"][:2], dtype=float)
    desk_bounds = (
        float(center[0] - DESK_HALF_SIZE[0]),
        float(center[0] + DESK_HALF_SIZE[0]),
        float(center[1] - DESK_HALF_SIZE[1]),
        float(center[1] + DESK_HALF_SIZE[1]),
    )
    requested_seed = getattr(args, "task_seed", None)
    seed = (
        0
        if getattr(args, "no_task_randomization", False)
        else requested_seed if requested_seed is not None else secrets.randbits(32)
    )
    rng = np.random.default_rng(seed)
    base_xy = np.array([float(args.panda_x), float(args.panda_y)])
    pencil_sector = _sector(
        base_xy,
        TASK_PENCIL_ANGLES_DEG,
        TASK_PENCIL_RADIUS,
        [0.08, 0.24, 0.86, 0.55],
        desk_bounds,
        TASK_PENCIL_EDGE_CLEARANCE,
    )
    cup_sector = _sector(
        base_xy,
        TASK_CUP_ANGLES_DEG,
        TASK_CUP_RADIUS,
        [1.0, 0.38, 0.015, 0.55],
        desk_bounds,
        TASK_CUP_EDGE_CLEARANCE,
    )
    _assert_sector_fits_desk(pencil_sector, desk_bounds)
    _assert_sector_fits_desk(cup_sector, desk_bounds)
    pencil_xy = _sample_sector(rng, pencil_sector)
    pencil_radius = float(np.linalg.norm(pencil_xy - base_xy))
    for _ in range(1_000):
        cup_xy = _sample_sector(rng, cup_sector)
        if np.linalg.norm(cup_xy - pencil_xy) >= TASK_OBJECT_MIN_SEPARATION:
            break
    else:
        raise RuntimeError("Could not keep the random pencil and cup safely separated")

    cup = _layout_item(layout, TASK_CUP_ID)
    cup["pos"] = [float(cup_xy[0]), float(cup_xy[1]), DESK_TOP_Z]
    cup["yaw"] = float(rng.uniform(-15.0, 15.0))
    layout["objects"] = [
        item for item in layout["objects"] if item.get("id") != TASK_PENCIL_ID
    ]
    layout["objects"].append(
        {
            "id": TASK_PENCIL_ID,
            "catalog_id": "custom:wooden_pencil_model",
            "asset_id": "wooden_pencil_model",
            "active": True,
            "dynamic": True,
            "pos": [
                float(pencil_xy[0]),
                float(pencil_xy[1]),
                DESK_TOP_Z + TASK_PENCIL_SPAWN_OFFSET_Z,
            ],
            "yaw": float(rng.uniform(*_pencil_yaw_range(pencil_radius))),
        }
    )
    return {
        "seed": int(seed),
        "desk_center": center,
        "desk_bounds": desk_bounds,
        "base_xy": base_xy,
        "pencil_sector": pencil_sector,
        "cup_sector": cup_sector,
        "pencil_xy": pencil_xy,
        "cup_xy": cup_xy,
        "desk_top_z": DESK_TOP_Z,
        "pencil_spawn_z": DESK_TOP_Z + TASK_PENCIL_SPAWN_OFFSET_Z,
        "cup_spawn_z": DESK_TOP_Z,
        "object_min_separation": TASK_OBJECT_MIN_SEPARATION,
        "pencil_yaw_range_deg": TASK_PENCIL_YAW_DEG,
        "pencil_outer_yaw_radius": TASK_PENCIL_OUTER_YAW_RADIUS,
        "pencil_outer_yaw_range_deg": TASK_PENCIL_OUTER_YAW_DEG,
        "cup_yaw_range_deg": (-15.0, 15.0),
        "min_reach": TASK_MIN_REACH,
        "max_reach": TASK_MAX_REACH,
        "reach_profile": TASK_REACH_PROFILE.copy(),
    }


def respawn_panda_task(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: dict,
    pencil_body: int,
    cup_body: int,
    rng: np.random.Generator,
) -> None:
    pencil_xy = _sample_sector(rng, task["pencil_sector"])
    pencil_radius = float(np.linalg.norm(pencil_xy - np.asarray(task["base_xy"], dtype=float)))
    for _ in range(1_000):
        cup_xy = _sample_sector(rng, task["cup_sector"])
        if np.linalg.norm(cup_xy - pencil_xy) >= TASK_OBJECT_MIN_SEPARATION:
            break
    else:
        raise RuntimeError("Could not keep the random pencil and cup safely separated")

    place_panda_task(
        model, data, task, pencil_body, cup_body, pencil_xy,
        rng.uniform(*_pencil_yaw_range(pencil_radius)), cup_xy, rng.uniform(*task["cup_yaw_range_deg"]),
    )


def place_panda_task(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: dict,
    pencil_body: int,
    cup_body: int,
    pencil_xy,
    pencil_yaw_deg: float,
    cup_xy,
    cup_yaw_deg: float,
) -> None:
    pencil_xy = np.asarray(pencil_xy, dtype=float)[:2]
    cup_xy = np.asarray(cup_xy, dtype=float)[:2]
    pencil_z, cup_z = float(task["pencil_spawn_z"]), float(task["cup_spawn_z"])
    for body_id, xy, z, yaw in (
        (pencil_body, pencil_xy, pencil_z, pencil_yaw_deg),
        (cup_body, cup_xy, cup_z, cup_yaw_deg),
    ):
        quat = yaw_quaternion(float(yaw))
        joint_id = int(model.body_jntadr[body_id])
        if model.body_jntnum[body_id] and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            data.qpos[qpos_adr:qpos_adr + 7] = [*xy, z, *quat]
            data.qvel[dof_adr:dof_adr + 6] = 0.0
        else:
            model.body_pos[body_id] = [*xy, z]
            model.body_quat[body_id] = quat
    task["pencil_xy"], task["cup_xy"] = pencil_xy, cup_xy

    for site_name, xyz in (
        ("panda_task_pencil_spawn", [*pencil_xy, pencil_z + 0.017]),
        ("panda_task_cup_drop_target", [*cup_xy, cup_z + 0.230]),
    ):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            model.site_pos[site_id] = xyz
    mujoco.mj_forward(model, data)


def _add_line_loop(
    body: mujoco.MjsBody,
    name: str,
    points: list[np.ndarray],
    z: float,
    rgba: list[float],
) -> None:
    for index, (start, end) in enumerate(zip(points, points[1:] + points[:1])):
        body.add_geom(
            name=f"{name}_{index}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[float(start[0]), float(start[1]), z, float(end[0]), float(end[1]), z],
            size=[0.007, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
            group=5,
        )


def _add_annular_sector(
    scene_spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    name: str,
    sector: dict,
    z: float,
) -> None:
    angles = np.radians(np.linspace(*sector["angles_deg"], 97))
    center = np.asarray(sector["center"], dtype=float)
    thickness = 0.0002
    vertices: list[list[float]] = []
    for angle in angles:
        direction = np.array([math.cos(angle), math.sin(angle)])
        outer = center + _profile_radius(sector, math.degrees(angle)) * direction
        inner = center + sector["min_radius"] * direction
        vertices.extend(
            [
                [float(outer[0]), float(outer[1]), z],
                [float(inner[0]), float(inner[1]), z],
                [float(outer[0]), float(outer[1]), z + thickness],
                [float(inner[0]), float(inner[1]), z + thickness],
            ]
        )
    faces: list[list[int]] = []
    for index in range(len(angles) - 1):
        bo, bi, top_o, top_i = (4 * index + offset for offset in range(4))
        next_bo, next_bi, next_top_o, next_top_i = (
            4 * (index + 1) + offset for offset in range(4)
        )
        faces.extend(
            [
                [top_o, next_top_o, next_top_i], [top_o, next_top_i, top_i],
                [bo, next_bi, next_bo], [bo, bi, next_bi],
                [bo, next_bo, next_top_o], [bo, next_top_o, top_o],
                [bi, top_i, next_top_i], [bi, next_top_i, next_bi],
            ]
        )
    last = 4 * (len(angles) - 1)
    faces.extend(
        [
            [0, 2, 3], [0, 3, 1],
            [last, last + 1, last + 3], [last, last + 3, last + 2],
        ]
    )
    mesh = scene_spec.add_mesh(
        name=f"{name}_mesh",
        uservert=np.asarray(vertices, dtype=float).reshape(-1).tolist(),
        userface=np.asarray(faces, dtype=int).reshape(-1).tolist(),
    )
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        rgba=sector["rgba"],
        contype=0,
        conaffinity=0,
        group=5,
    )


def add_task_zone_visuals(scene_spec: mujoco.MjSpec, task: dict) -> None:
    body = scene_spec.worldbody.add_body(name="panda_task_zone_visuals")
    z = DESK_TOP_Z + 0.0005
    _add_annular_sector(
        scene_spec, body, "pencil_spawn_sector", task["pencil_sector"], z
    )
    _add_annular_sector(
        scene_spec, body, "cup_spawn_sector", task["cup_sector"], z
    )
    desk_x0, desk_x1, desk_y0, desk_y1 = task["desk_bounds"]
    _add_line_loop(
        body,
        "desk_boundary",
        [
            np.array([desk_x0, desk_y0]),
            np.array([desk_x1, desk_y0]),
            np.array([desk_x1, desk_y1]),
            np.array([desk_x0, desk_y1]),
        ],
        z + 0.006,
        [1.0, 1.0, 1.0, 0.9],
    )
    for name, radius, color in (
        ("minimum_reach", task["min_reach"], [1.0, 0.82, 0.12, 0.9]),
        ("maximum_reach", task["max_reach"], [0.16, 1.0, 0.35, 0.9]),
    ):
        points = [
            task["base_xy"]
            + radius * np.array([math.cos(angle), math.sin(angle)])
            for angle in np.linspace(0.0, 2.0 * math.pi, 49)[:-1]
        ]
        _add_line_loop(body, name, points, z + 0.008, color)


def add_task_sites(scene_spec: mujoco.MjSpec, task: dict, visible: bool) -> None:
    alpha = 0.9 if visible else 0.0
    scene_spec.worldbody.add_site(
        name="panda_task_pencil_spawn",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[float(task["pencil_xy"][0]), float(task["pencil_xy"][1]), DESK_TOP_Z + 0.025],
        size=[0.006, 0.0, 0.0],
        rgba=[0.86, 0.55, 0.20, alpha],
        group=5,
    )
    scene_spec.worldbody.add_site(
        name="panda_task_cup_drop_target",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[float(task["cup_xy"][0]), float(task["cup_xy"][1]), DESK_TOP_Z + 0.230],
        size=[0.014, 0.0, 0.0],
        rgba=[0.1, 1.0, 0.25, alpha],
        group=5,
    )


def save_task_map(task: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Polygon, Rectangle

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8.5, 7.0), dpi=150)
    x0, x1, y0, y1 = task["desk_bounds"]
    axes.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#b87836", edgecolor="#4a2a14", linewidth=2.5, label="desk 0.847 x 1.769 m"))
    legend_patches = []
    for sector, color, label in (
        (task["pencil_sector"], "#143ddb", "pencil safe sector (left)"),
        (task["cup_sector"], "#ff6104", "holder safe sector (right)"),
    ):
        angles = np.linspace(*sector["angles_deg"], 65)
        outer = [
            np.asarray(sector["center"])
            + _profile_radius(sector, angle)
            * np.asarray([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            for angle in angles
        ]
        inner = [
            np.asarray(sector["center"])
            + sector["min_radius"]
            * np.asarray([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            for angle in angles[::-1]
        ]
        axes.add_patch(
            Polygon(
                np.asarray(outer + inner), closed=True, facecolor=color,
                edgecolor=color, linewidth=1.5, alpha=0.55, zorder=3,
            )
        )
        legend_patches.append(Patch(facecolor=color, alpha=0.55, label=label))
    for radius, color, style, label in (
        (task["min_reach"], "#eab308", "--", f"IK-tested envelope min {task['min_reach']:.2f} m"),
        (task["max_reach"], "#16a34a", "-", f"IK-tested envelope max {task['max_reach']:.2f} m"),
    ):
        axes.add_patch(Circle(task["base_xy"], radius, fill=False, edgecolor=color, linestyle=style, linewidth=2.2, label=label))
    axes.scatter(*task["base_xy"], s=140, marker="^", color="#dc2626", edgecolor="white", linewidth=1.2, zorder=8, label="Panda base")
    axes.scatter(*task["pencil_xy"], s=120, marker="_", color="#0f172a", linewidth=4, zorder=9, label="sampled pencil")
    axes.scatter(*task["cup_xy"], s=100, marker="o", facecolor="none", edgecolor="#111827", linewidth=3, zorder=9, label="sampled cup")
    axes.set_aspect("equal")
    axes.set_xlim(x0 - 0.18, x1 + 0.18)
    axes.set_ylim(y0 - 0.16, y1 + 0.20)
    axes.set_xlabel("world X (m)")
    axes.set_ylabel("world Y (m)")
    axes.set_title(f"Panda pencil-to-cup task zones (seed={task['seed']})")
    axes.grid(alpha=0.2)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(
        legend_patches + handles,
        [patch.get_label() for patch in legend_patches] + labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_maze_map(layout: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch, Rectangle

    floors = {floor["id"]: floor for floor in layout["structure"]["floors"]}
    if "maze_red_floor" not in floors:
        raise RuntimeError("Missing Mission 3 cliff path floor in layout")

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1 = MAZE_ROOM_BOUNDS
    figure, axes = plt.subplots(figsize=(9.2, 7.0), dpi=150)
    axes.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="#e8edf3",
            edgecolor="#1f2937",
            linewidth=4.0,
            zorder=1,
        )
    )
    axes.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor="#ef431c", edgecolor="#7f1d1d", linewidth=2, zorder=2,
        )
    )
    axes.add_patch(
        PolygonPatch(
            np.asarray(floors["maze_red_floor"]["polygon"]), closed=True,
            facecolor="#3d454d", edgecolor="#111827", linewidth=2.5, zorder=3,
        )
    )
    if "maze_red_exit_platform" in floors:
        axes.add_patch(
            PolygonPatch(
                np.asarray(floors["maze_red_exit_platform"]["polygon"]), closed=True,
                facecolor="#3d454d", edgecolor="#111827", linewidth=2.5, zorder=3,
            )
        )
    axes.plot(
        CLIFF_PATH_ROUTE[:, 0], CLIFF_PATH_ROUTE[:, 1], "--",
        color="#f8fafc", linewidth=2.2, zorder=5,
    )
    for index, (jump_x, low_y, high_y, width) in enumerate(CLIFF_SHORTCUTS, 1):
        axes.annotate(
            "", xy=(jump_x, high_y), xytext=(jump_x, low_y),
            arrowprops={"arrowstyle": "<->", "color": "#fde047", "lw": 2.5},
            zorder=6,
        )
        label_on_left = jump_x > 11.0
        axes.text(
            jump_x - 0.12 if label_on_left else jump_x + 0.12,
            (low_y + high_y) / 2,
            f"jump {index}: {width:.2f} m", color="#fef08a", fontsize=10,
            va="center", ha="right" if label_on_left else "left", zorder=6,
        )
    axes.annotate(
        "entrance / G1 start",
        xy=(12.98, 20.125),
        xytext=(13.25, 20.75),
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 2.5},
        color="#991b1b",
        fontsize=12,
        ha="left",
    )
    axes.annotate(
        "exit",
        xy=(12.98, 11.375),
        xytext=(13.25, 10.85),
        arrowprops={"arrowstyle": "->", "color": "#16a34a", "lw": 2.5},
        color="#166534",
        fontsize=12,
        ha="left",
    )
    axes.text(
        9.25,
        10.05,
        "uniform road width 1.15 m  |  drop 1.50 m  |  gaps 0.35 / 0.50 / 0.65 / 0.75 / 0.85 m",
        ha="center",
        fontsize=11,
    )
    axes.set_aspect("equal")
    axes.set_xlim(5.0, 13.5)
    axes.set_ylim(9.75, 21.7)
    axes.set_xlabel("world X (m)")
    axes.set_ylabel("world Y (m)")
    axes.set_title("Mission 3: continuous mirrored path with widening cliff gaps")
    axes.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def set_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    value: float,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"Missing joint after attachment: {name}")
    data.qpos[model.jnt_qposadr[joint_id]] = value


def build_combined_model(args: argparse.Namespace):
    assets_dir = resolve_assets_dir(args.assets_dir)
    os.environ["MLSPACES_ASSETS_DIR"] = str(assets_dir)
    _, catalog = EDITOR.load_catalog(args.catalog.expanduser().resolve())
    EDITOR.relocate_catalog_paths(catalog, assets_dir)
    layout = EDITOR.load_layout(args.layout.expanduser().resolve())
    EDITOR._portableize_layout_texture_paths(layout, assets_dir)
    EDITOR._realign_custom_walls(layout, EDITOR._scene_description(assets_dir))
    task = randomize_panda_task(layout, args)
    task_map = getattr(args, "task_map", None)
    if task_map is not None:
        save_task_map(task, task_map)
    maze_map = getattr(args, "maze_map", None)
    if maze_map is not None:
        save_maze_map(layout, maze_map)
    EDITOR.hydrate_layout_assets(layout, catalog, assets_dir)
    scene_spec, _, _, removed, _ = EDITOR.build_editor_spec(
        assets_dir, catalog, layout, movable_instances=False
    )
    scene_spec.copy_during_attach = True
    show_task_zones = getattr(args, "show_task_zones", False)
    add_task_sites(scene_spec, task, show_task_zones)
    if show_task_zones:
        add_task_zone_visuals(scene_spec, task)

    legacy_g1_path = Path(ROBOT_XML["g1"])
    symmetric_g1_path = Path(ROBOT_XML["g1_sensors"])
    if not args.original_g1_hands:
        generated_text = (
            symmetric_g1_path.read_text(errors="replace")
            if symmetric_g1_path.is_file()
            else ""
        )
        if G1_SENSOR_RIG.EXTERIOR_MARKER not in generated_text:
            G1_SENSOR_RIG.build_sensor_xml(legacy_g1_path, symmetric_g1_path)
    g1_path = legacy_g1_path if args.original_g1_hands else symmetric_g1_path
    panda_path = PANDA_MODEL
    missing = [path for path in (g1_path, panda_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing robot model(s):\n" + "\n".join(map(str, missing)))

    source_g1_model = mujoco.MjModel.from_xml_path(str(g1_path))
    g1_spec = mujoco.MjSpec.from_file(str(g1_path))
    g1_root = g1_spec.body("pelvis")
    if g1_root is None:
        raise RuntimeError("G1 pelvis body is missing")
    pinned_height = 0.0
    if args.pin_g1:
        source_freejoint = next(
            joint_id
            for joint_id in range(source_g1_model.njnt)
            if source_g1_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        )
        source_qpos = source_g1_model.jnt_qposadr[source_freejoint]
        pinned_height = float(source_g1_model.qpos0[source_qpos + 2])
        remove_root_freejoint(g1_spec, g1_root)
        g1_root.pos = [0.0, 0.0, pinned_height]
    if getattr(args, "g1_maze_start", False):
        g1_spawn = (13.55, 20.125, float(args.g1_z), 180.0)
    else:
        g1_spawn = (
            float(args.g1_x),
            float(args.g1_y),
            float(args.g1_z),
            float(args.g1_yaw),
        )
    g1_frame = scene_spec.worldbody.add_frame(
        pos=g1_spawn[:3],
        quat=yaw_quaternion(g1_spawn[3]),
    )
    g1_frame.attach_body(g1_root, G1_PREFIX, "")

    panda_spec = mujoco.MjSpec.from_file(str(panda_path))
    panda_root = panda_spec.body("robot0_base")
    if panda_root is None:
        raise RuntimeError("LIBERO Panda robot0_base body is missing")
    for geom in panda_spec.geoms:
        if geom.group == 0:
            geom.group = 3
            geom.rgba = [0.0, 0.0, 0.0, 0.0]
        elif geom.group == 1:
            geom.group = 2
    cameras = getattr(args, "panda_cameras", None) or {}
    agentview = cameras.get("agentview") or {}
    panda_root.add_camera(
        name="agentview",
        pos=agentview.get("pos", PANDA_AGENTVIEW_POS),
        quat=agentview.get("quat", PANDA_AGENTVIEW_QUAT),
        fovy=agentview.get("fovy", PANDA_AGENTVIEW_FOVY),
    )
    wrist = cameras.get("wrist")
    if wrist:
        eye_in_hand = panda_spec.camera("robot0_eye_in_hand")
        if eye_in_hand is None:
            raise RuntimeError("LIBERO Panda has no robot0_eye_in_hand camera")
        eye_in_hand.pos = list(wrist["pos"])
        eye_in_hand.quat = list(wrist["quat"])
        eye_in_hand.fovy = float(wrist["fovy"])
    panda_root.pos = [args.panda_x, args.panda_y, args.panda_z]
    panda_root.quat = yaw_quaternion(args.panda_yaw)
    scene_spec.worldbody.add_frame().attach_body(panda_root, PANDA_PREFIX, "")

    model = scene_spec.compile()
    # Keep collision alpha nonzero for ray queries; hide colliders through geom groups.
    data = mujoco.MjData(model)
    copy_stand_pose(source_g1_model, model, data)
    initialize_panda_state(model, data)
    return model, data, removed, task


def actuator_id(model: mujoco.MjModel, suffix: str) -> int:
    for index in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        if name.endswith(suffix):
            return index
    return -1


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    demo: bool,
    initial_view: str,
) -> None:
    state = {"running": bool(demo), "reset": False}
    initial_qpos = data.qpos.copy()
    initial_ctrl = data.ctrl.copy()
    start_time = time.monotonic()
    g1_demo = actuator_id(model, "g1/walk_right_shoulder_pitch_joint")
    panda_controller = PandaController(model, data)

    def on_key(key: int) -> None:
        if key == 32:
            state["running"] = not state["running"]
        elif key in (ord("R"), ord("r")):
            state["reset"] = True

    print("MuJoCo controls: Space=physics pause/run, R=reset both robots, Esc=close")
    if not demo:
        print("Physics starts paused. Add --demo --pin-g1 for stable simultaneous motion.")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera_presets = {
            "classroom": ([19.7, 18.1, 1.05], 6.2, 145.0, -28.0),
            "overview": ([13.0, 13.0, 0.45], 35.0, 0.0, -78.0),
            "nurse": ([20.5, 10.5, 0.75], 8.0, 145.0, -42.0),
            "restaurant": ([20.5, 3.5, 0.75], 8.0, 145.0, -42.0),
            "maze": ([9.25, 15.75, 0.65], 13.0, 0.0, -64.0),
        }
        lookat, distance, azimuth, elevation = camera_presets[initial_view]
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = distance
        viewer.cam.azimuth = azimuth
        viewer.cam.elevation = elevation
        viewer.opt.geomgroup[3] = 0
        viewer.opt.geomgroup[4] = 0
        viewer.opt.geomgroup[5] = 1
        last = time.monotonic()
        accumulator = 0.0
        while viewer.is_running():
            now = time.monotonic()
            accumulator += min(now - last, 0.1)
            last = now
            if state["reset"]:
                data.qpos[:] = initial_qpos
                data.qvel[:] = 0.0
                data.ctrl[:] = initial_ctrl
                data.time = 0.0
                mujoco.mj_forward(model, data)
                state["reset"] = False
                start_time = now
            if state["running"]:
                if demo:
                    phase = now - start_time
                    panda_target = PANDA_HOME.copy()
                    panda_target[0] += 0.22 * math.sin(0.65 * phase)
                    panda_controller.set_joint_target(panda_target)
                    if g1_demo >= 0:
                        data.ctrl[g1_demo] = initial_ctrl[g1_demo] + 0.18 * math.sin(0.65 * phase + 0.8)
                timestep = max(float(model.opt.timestep), 1e-6)
                steps = min(50, int(accumulator / timestep))
                for _ in range(steps):
                    panda_controller.before_step(data)
                    mujoco.mj_step(model, data)
                accumulator -= steps * timestep
            else:
                accumulator = 0.0
            viewer.sync()
            time.sleep(1.0 / 120.0)


def main() -> None:
    args = parse_args()
    if args.demo and not args.pin_g1:
        print("Note: --demo without --pin-g1 requires your own balance controller; G1 may fall.")
    model, data, removed, task = build_combined_model(args)
    g1_actuators = sum(
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith(G1_PREFIX)
        for i in range(model.nu)
    )
    panda_actuators = sum(
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith(PANDA_PREFIX)
        for i in range(model.nu)
    )
    print(
        f"Compiled current val_861 layout + G1 + LIBERO Panda: "
        f"nbody={model.nbody}, ngeom={model.ngeom}, nq={model.nq}, nu={model.nu}"
    )
    print(
        f"Robot actuators: G1={g1_actuators}, Panda={panda_actuators}; "
        f"cleared original objects={removed}"
    )
    pencil_distance = float(np.linalg.norm(task["pencil_xy"] - task["base_xy"]))
    cup_distance = float(np.linalg.norm(task["cup_xy"] - task["base_xy"]))
    print(
        f"Task seed={task['seed']}: pencil=({task['pencil_xy'][0]:.3f}, "
        f"{task['pencil_xy'][1]:.3f}) r={pencil_distance:.3f} m; "
        f"cup=({task['cup_xy'][0]:.3f}, {task['cup_xy'][1]:.3f}) "
        f"r={cup_distance:.3f} m"
    )
    if args.show_task_zones:
        print(
            "Task-zone legend: blue=pencil(left), orange=cup(right), "
            "yellow/green=min/max safe reach, white=desk boundary"
        )
    if args.g1_maze_start:
        print("G1 maze start: classroom side (13.550, 20.125), facing west into first lane")
    if args.task_map is not None:
        print(f"Task map: {args.task_map.expanduser().resolve()}")
    if args.maze_map is not None:
        print(f"Maze map: {args.maze_map.expanduser().resolve()}")
    if args.check_only:
        return
    initial_view = "maze" if args.g1_maze_start and args.view == "classroom" else args.view
    run_viewer(model, data, demo=args.demo, initial_view=initial_view)


if __name__ == "__main__":
    main()
