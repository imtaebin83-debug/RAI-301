from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ros2_bridge.tools.panda_controller import (
    ARM_JOINTS,
    OSC_POSITION_SCALE,
    OSC_ROTATION_SCALE,
    PandaController,
)

SCHEMA_VERSION = "val861-libero-panda-hdf5-v4"
LEGACY_SCHEMA_VERSIONS = {
    "val861-libero-panda-hdf5-v2",
    "val861-libero-panda-hdf5-v3",
}

PENCIL_COLLISION_HALF_LENGTH = 0.1445
PENCIL_COLLISION_RADIUS = 0.0145
HOLDER_INNER_HALF_X = 0.0603571
HOLDER_INNER_HALF_Y = 0.0603571
HOLDER_BOTTOM_THICKNESS = 0.00375
HOLDER_BODY_HEIGHT = 0.150
HOLDER_MAX_STABLE_TILT_DEG = math.degrees(
    math.atan2(
        2.0 * (min(HOLDER_INNER_HALF_X, HOLDER_INNER_HALF_Y) - PENCIL_COLLISION_RADIUS),
        HOLDER_BODY_HEIGHT - HOLDER_BOTTOM_THICKNESS,
    )
)
HOLDER_SAFE_TILT_DEG = 25.0

PHONE_PITCH_DEADBAND = math.radians(4.0)
PHONE_PITCH_LIMIT = math.radians(85.0)
PHONE_PITCH_FILTER_TAU = 0.10
PHONE_PITCH_RATE_LIMIT = math.radians(75.0)
PREVIEW_WINDOW = "val_861 Panda cameras (agent | wrist | top | insertion)"
PREVIEW_TILE_SIZE = 480


def _object_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj_type, index) or ""


def _pose7(position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return np.asarray(
        [*position, quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )


def _matrix_from_pose7(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,):
        raise ValueError(f"eef_pose must have 7 values [x,y,z,qw,qx,qy,qz], got {pose.shape}")
    transform = np.eye(4)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = Rotation.from_quat(
        [pose[4], pose[5], pose[6], pose[3]]
    ).as_matrix()
    return transform


class EpisodeWriter:
    def __init__(
        self,
        root: Path,
        episode_index: int,
        instruction: str,
        fps: float,
        seed: int | None,
        camera_metadata: dict[str, Any] | None = None,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.final_path = root / f"episode_{episode_index:06d}.h5"
        self.temp_path = root / f".episode_{episode_index:06d}.recording.h5"
        self.file = h5py.File(self.temp_path, "w")
        self.file.attrs.update(
            schema=SCHEMA_VERSION,
            task_instruction=instruction,
            fps=float(fps),
            task_seed=-1 if seed is None else int(seed),
            created_unix=time.time(),
            robot="robosuite-1.4.0-Panda+PandaGripper",
            observation_space="eef_xyz[3]+eef_axis_angle[3]+finger_qpos[2]",
            action_space="normalized_eef_delta[6]+gripper(-1=open,+1=close)",
            state_frame="panda_base",
            action_frame="panda_base",
            action_position_scale_m=OSC_POSITION_SCALE,
            action_rotation_scale_rad=OSC_ROTATION_SCALE,
        )
        for key, value in (camera_metadata or {}).items():
            self.file.attrs[key] = value
        self.datasets: dict[str, h5py.Dataset] = {}
        self.count = 0

    def _append(self, name: str, value: np.ndarray) -> None:
        value = np.asarray(value)
        dataset = self.datasets.get(name)
        if dataset is None:
            if "/" in name:
                parent, leaf = name.rsplit("/", 1)
                group = self.file.require_group(parent)
            else:
                group, leaf = self.file, name
            chunks = (1, *value.shape)
            compression = "lzf" if value.size > 32 else None
            dataset = group.create_dataset(
                leaf,
                shape=(0, *value.shape),
                maxshape=(None, *value.shape),
                chunks=chunks,
                dtype=value.dtype,
                compression=compression,
            )
            self.datasets[name] = dataset
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def append(self, observation: dict[str, Any], action: np.ndarray) -> None:
        self._append("time", np.asarray(observation["time"], dtype=np.float64))
        self._append("observation/image", observation["image"])
        self._append("observation/wrist_image", observation["wrist_image"])
        self._append("observation/state", np.asarray(observation["state"], dtype=np.float32))
        self._append("actions", np.asarray(action, dtype=np.float32))
        self._append(
            "diagnostics/joint_position",
            np.asarray(observation["joint_position"], dtype=np.float32),
        )
        self._append(
            "diagnostics/joint_velocity",
            np.asarray(observation["joint_velocity"], dtype=np.float32),
        )
        for name in (
            "pencil_pose_world",
            "holder_pose_world",
            "pencil_tilt_deg",
            "pencil_holder_center_distance_xy",
        ):
            if name in observation:
                self._append(
                    f"diagnostics/{name}",
                    np.asarray(observation[name], dtype=np.float32),
                )
        self.count += 1
        if self.count % 30 == 0:
            self.file.flush()

    def finish(self, success: bool | None, reason: str) -> Path | None:
        if self.file is None:
            return None
        self.file.attrs["success"] = -1 if success is None else int(success)
        self.file.attrs["finish_reason"] = reason
        self.file.attrs["frames"] = self.count
        self.file.flush()
        self.file.close()
        self.file = None
        if self.count == 0:
            self.temp_path.unlink(missing_ok=True)
            return None
        self.temp_path.replace(self.final_path)
        return self.final_path

    def discard(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        self.temp_path.unlink(missing_ok=True)


class PandaLearningRuntime:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        task: dict[str, Any],
        *,
        dataset_dir: Path,
        instruction: str,
        rate_hz: float = 15.0,
        teledex_port: int = 8888,
        position_scale: float = 1.6,
        yaw_scale: float = 1.0,
        pitch_scale: float = 1.0,
        preview: bool = True,
        controller: PandaController | None = None,
    ) -> None:
        self.model, self.data, self.task = model, data, task
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.instruction = instruction
        self.rate_hz = float(rate_hz)
        self.period = 1.0 / self.rate_hz
        self.next_sample_time = float(data.time)
        self.position_scale = float(position_scale)
        self.yaw_scale = float(yaw_scale)
        self.pitch_scale = float(pitch_scale)
        if not np.all(np.isfinite([self.yaw_scale, self.pitch_scale])):
            raise ValueError("TeleDex yaw/pitch parameters must be finite")
        self.preview = bool(preview and os.environ.get("DISPLAY"))
        self._preview_window_initialized = False
        self.stop_requested = False
        self.last_stop_reason: str | None = None

        self.controller = controller or PandaController(model, data)
        self.joint_ids = self.controller.joint_ids
        self.qpos_adrs = self.controller.qpos_adrs
        self.dof_adrs = self.controller.dof_adrs
        self.target_gripper = self.controller.gripper_opening
        self.target_eef = self.controller.site_transform(data)
        self.initial_qpos = data.qpos.copy()
        self.initial_qvel = data.qvel.copy()
        self.initial_ctrl = data.ctrl.copy()
        self._phone_origin: np.ndarray | None = None
        self._eef_origin_base: np.ndarray | None = None
        self._initial_toggle = False
        self._last_phone_toggle_state: bool | None = None
        self._manual_gripper_closed: bool | None = None
        self._last_reported_phone_rotation_deg = np.zeros(3, dtype=float)
        self._filtered_phone_pitch = 0.0
        self._last_phone_pitch_time = float(data.time)
        self._last_button = False
        self._wait_for_button_release = False
        self._last_secondary_button = False
        self._wait_for_secondary_button_release = False
        self._connected_once = False
        self._task_rng = np.random.default_rng()
        self._pencil_body = self._find_task_body("panda_task_random_pencil")
        self._cup_body = self._find_task_body("school_room7_empty_desk_pencil_cup")

        self.render_option = mujoco.MjvOption()
        self.render_option.geomgroup[3] = 0
        self.render_option.geomgroup[4] = 0
        self.render_option.geomgroup[5] = 0
        # Hide diagnostic sites from learning observations to match LIBERO frames.
        self.render_option.sitegroup[:] = 0
        self.renderer = mujoco.Renderer(model, height=256, width=256)
        self.preview_renderer = (
            mujoco.Renderer(
                model, height=PREVIEW_TILE_SIZE, width=PREVIEW_TILE_SIZE
            )
            if self.preview
            else None
        )
        for camera_name in ("panda/agentview", "panda/robot0_eye_in_hand"):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                raise RuntimeError(f"learning camera missing after composition: {camera_name}")

        self.session = None
        self.writer: EpisodeWriter | None = None
        self.episode_index = self._next_episode_index()
        import teledex

        self.session = teledex.Session(port=int(teledex_port), show_qr=True)
        # teledex 0.0.7 SIGKILLs whatever process holds its port (Jupyter also
        # defaults to 8888) before it listens. Never kill; collect_panda.py checks the port.
        self.session._kill_process_using_port = lambda _port: None
        self._install_secondary_button_compat(self.session)
        self.session.start()
        print(
            "[Panda collect] TeleDex 연결 대기 중입니다. QR/IP를 iPhone 앱으로 연결하세요.\n"
            "  phone-only/display-up: app +X=forward, -Y=right, +Z=up; "
            "XYZ + gripper yaw/pitch 동시 적용 (position/rotation decomposed, roll locked), "
            "MuJoCo G=gripper 열기/닫기\n"
            "  TeleDex: Button A=success/save, Button B=discard+respawn\n"
            "  viewer: G=gripper, S=success, F=fail, "
            "Button B=discard+respawn, Q=quit"
        )

    @staticmethod
    def _install_secondary_button_compat(session: Any) -> None:
        session.latest_data.setdefault("button_secondary", False)
        process_data = session._process_data

        def process_data_with_secondary(raw_data: dict[str, Any]) -> None:
            process_data(raw_data)
            session.latest_data["button_secondary"] = bool(
                raw_data.get(
                    "button_secondary",
                    raw_data.get(
                        "buttonSecondary", raw_data.get("button_b", False)
                    ),
                )
            )

        session._process_data = process_data_with_secondary

    def _next_episode_index(self) -> int:
        indices = []
        if self.dataset_dir.exists():
            for path in self.dataset_dir.glob("episode_*.h5"):
                try:
                    indices.append(int(path.stem.split("_")[-1]))
                except ValueError:
                    pass
        return max(indices, default=-1) + 1

    def _site_transform(self, data: mujoco.MjData) -> np.ndarray:
        return self.controller.site_transform(data)

    def _base_transform(self, data: mujoco.MjData) -> np.ndarray:
        return self.controller.base_transform(data)

    def _clamp_task_target(self, target_world: np.ndarray) -> np.ndarray:
        target_world = np.asarray(target_world, dtype=float).copy()
        desk_x0, desk_x1, desk_y0, desk_y1 = self.task["desk_bounds"]
        target_world[0, 3] = np.clip(target_world[0, 3], desk_x0 + 0.05, desk_x1 - 0.05)
        target_world[1, 3] = np.clip(target_world[1, 3], desk_y0 + 0.05, desk_y1 - 0.05)
        target_world[2, 3] = np.clip(target_world[2, 3], 0.815, 1.35)
        offset = target_world[:2, 3] - np.asarray(self.task["base_xy"])
        radial = float(np.linalg.norm(offset))
        angle = math.degrees(math.atan2(offset[1], offset[0])) % 360.0
        profile = self.task["reach_profile"]
        verified_angles = np.asarray(sorted(profile), dtype=float)
        angle = float(np.clip(angle, verified_angles[0], verified_angles[-1]))
        max_reach = float(
            np.interp(angle, verified_angles, [profile[int(value)] for value in verified_angles])
        )
        radial = float(np.clip(radial, self.task["min_reach"], max_reach))
        target_world[:2, 3] = np.asarray(self.task["base_xy"]) + radial * np.asarray(
            [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
        )
        return target_world

    @staticmethod
    def _phone_transform(latest: dict[str, Any]) -> np.ndarray | None:
        if latest.get("position") is None or latest.get("rotation") is None:
            return None
        transform = np.eye(4)
        transform[:3, 3] = np.asarray(latest["position"], dtype=float)
        transform[:3, :3] = np.asarray(latest["rotation"], dtype=float).reshape(3, 3)
        return transform

    @staticmethod
    def _phone_screen_yaw_pitch(
        origin_rotation: np.ndarray,
        rotation: np.ndarray,
    ) -> tuple[float, float]:
        relative_world = rotation @ origin_rotation.T
        yaw = math.atan2(
            float(relative_world[1, 0]), float(relative_world[0, 0])
        )
        pitch = math.atan2(
            -float(relative_world[2, 0]),
            float(math.hypot(relative_world[0, 0], relative_world[1, 0])),
        )
        return yaw, pitch

    @staticmethod
    def _pitch_deadband(value: float) -> float:
        value = float(np.clip(value, -PHONE_PITCH_LIMIT, PHONE_PITCH_LIMIT))
        magnitude = abs(value)
        if magnitude <= PHONE_PITCH_DEADBAND:
            return 0.0
        shaped = (
            (magnitude - PHONE_PITCH_DEADBAND)
            * PHONE_PITCH_LIMIT
            / (PHONE_PITCH_LIMIT - PHONE_PITCH_DEADBAND)
        )
        return math.copysign(shaped, value)

    def _filter_phone_pitch(self, raw_pitch: float, sim_time: float) -> float:
        target = self._pitch_deadband(raw_pitch * self.pitch_scale)
        timestep = max(float(self.model.opt.timestep), 1e-4)
        dt = float(sim_time) - self._last_phone_pitch_time
        if not np.isfinite(dt) or dt <= 0.0:
            dt = timestep
        dt = float(np.clip(dt, timestep, 0.05))
        alpha = 1.0 - math.exp(-dt / PHONE_PITCH_FILTER_TAU)
        low_passed = self._filtered_phone_pitch + alpha * (
            target - self._filtered_phone_pitch
        )
        max_step = PHONE_PITCH_RATE_LIMIT * dt
        self._filtered_phone_pitch += float(
            np.clip(low_passed - self._filtered_phone_pitch, -max_step, max_step)
        )
        self._filtered_phone_pitch = float(
            np.clip(self._filtered_phone_pitch, -PHONE_PITCH_LIMIT, PHONE_PITCH_LIMIT)
        )
        self._last_phone_pitch_time = float(sim_time)
        return self._filtered_phone_pitch

    def _update_phone_target(self, data: mujoco.MjData) -> None:
        assert self.session is not None
        if not self.session.connected_clients:
            return
        latest = self.session.get_latest_data().copy()
        phone = self._phone_transform(latest)
        if phone is None:
            return
        if not self._connected_once:
            self._connected_once = True
            print("[Panda collect] iPhone connected; recording starts after pose anchor.")
        base = self._base_transform(data)
        button = bool(latest.get("button", False))
        secondary_button = bool(latest.get("button_secondary", False))
        if self._wait_for_button_release:
            self._last_button = button
            if button:
                return
            self._wait_for_button_release = False
        if self._wait_for_secondary_button_release:
            self._last_secondary_button = secondary_button
            if secondary_button:
                return
            self._wait_for_secondary_button_release = False
        if self._phone_origin is None:
            self._phone_origin = phone
            self._eef_origin_base = np.linalg.inv(base) @ self._site_transform(data)
            self._filtered_phone_pitch = 0.0
            self._last_phone_pitch_time = float(data.time)
            self._initial_toggle = bool(latest.get("toggle", False))
            self._last_phone_toggle_state = self._initial_toggle
            self.target_eef = self._site_transform(data)
            self.controller.set_eef_target(self.target_eef)
            self._start_writer()
            print("[Panda collect] first phone pose anchored; teleoperation ACTIVE")
            return

        # Keep phone translation independent of rotation; transform inversion couples them.
        delta_position = (
            phone[:3, 3] - self._phone_origin[:3, 3]
        ) * self.position_scale
        phone_yaw, raw_phone_pitch = self._phone_screen_yaw_pitch(
            self._phone_origin[:3, :3], phone[:3, :3]
        )
        phone_yaw *= self.yaw_scale
        phone_pitch = self._filter_phone_pitch(raw_phone_pitch, float(data.time))

        # Pre-multiply in the fixed Panda-base frame, not the downward-pointing tool frame.
        delta_rotation = (
            Rotation.from_euler("z", phone_yaw).as_matrix()
            @ Rotation.from_euler("y", phone_pitch).as_matrix()
        )
        target_base = self._eef_origin_base.copy()
        target_base[:3, 3] += delta_position
        target_base[:3, :3] = delta_rotation @ self._eef_origin_base[:3, :3]
        rotation_deg = np.asarray(
            [0.0, math.degrees(phone_pitch), math.degrees(phone_yaw)]
        )
        if np.max(np.abs(rotation_deg - self._last_reported_phone_rotation_deg)) >= 5.0:
            print(
                "[Panda collect] gripper base rotation xyz(deg)="
                f"{np.round(rotation_deg, 1).tolist()}"
            )
            self._last_reported_phone_rotation_deg = rotation_deg

        target_world = base @ target_base
        target_world = self._clamp_task_target(target_world)
        self.target_eef = target_world
        phone_toggle = bool(latest.get("toggle", False))
        if (
            self._last_phone_toggle_state is not None
            and phone_toggle != self._last_phone_toggle_state
        ):
            self._manual_gripper_closed = None
        self._last_phone_toggle_state = phone_toggle
        closed = (
            self._manual_gripper_closed
            if self._manual_gripper_closed is not None
            else phone_toggle != self._initial_toggle
        )
        self.target_gripper = 0.0 if closed else 0.04
        self.controller.set_eef_target(self.target_eef)
        self.controller.set_gripper_opening(self.target_gripper)

        # Button B deletion takes priority over a simultaneous success label.
        if secondary_button and not self._last_secondary_button:
            self._discard_episode()
            self._reset_episode(data, respawn_task=True)
            return
        if button and not self._last_button:
            # Use the operator success label; geometry checks can reject valid rim insertions.
            self._finish_episode(True, "teledex_button_a")
            self._reset_episode(data, respawn_task=True)
            return
        self._last_button = button
        self._last_secondary_button = secondary_button

    def before_step(self, data: mujoco.MjData) -> None:
        self._update_phone_target(data)
        self.controller.before_step(data)

    def toggle_gripper_from_viewer(self) -> None:
        self._manual_gripper_closed = self.target_gripper >= 0.02
        self.target_gripper = 0.0 if self._manual_gripper_closed else 0.04
        self.controller.set_gripper_opening(self.target_gripper)
        print(
            "[Panda collect] gripper "
            + ("CLOSE" if self._manual_gripper_closed else "OPEN")
            + " (MuJoCo G)"
        )

    def _render_observation(self, data: mujoco.MjData) -> dict[str, Any]:
        images = {}
        for short_name, camera_name in (
            ("image", "panda/agentview"),
            ("wrist_image", "panda/robot0_eye_in_hand"),
        ):
            self.renderer.update_scene(data, camera=camera_name, scene_option=self.render_option)
            images[short_name] = self.renderer.render().copy()
        if self.preview:
            assert self.preview_renderer is not None
            # Rotate only operator previews; preserve saved LIBERO tensors.
            for short_name, camera_name, rotate_k in (
                ("preview_image", "panda/agentview", 0),
                ("preview_wrist_image", "panda/robot0_eye_in_hand", 2),
            ):
                self.preview_renderer.update_scene(
                    data, camera=camera_name, scene_option=self.render_option
                )
                frame = self.preview_renderer.render()
                images[short_name] = (
                    frame.copy() if rotate_k == 0 else np.rot90(frame, rotate_k).copy()
                )
            desk_x0, desk_x1, desk_y0, desk_y1 = self.task["desk_bounds"]
            desk_center = np.asarray(
                [(desk_x0 + desk_x1) * 0.5, (desk_y0 + desk_y1) * 0.5, 0.81]
            )
            top_camera = mujoco.MjvCamera()
            top_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            top_camera.lookat[:] = desk_center
            top_camera.distance = 1.55
            top_camera.azimuth = 270.0
            top_camera.elevation = -89.0
            self.preview_renderer.update_scene(
                data, camera=top_camera, scene_option=self.render_option
            )
            images["task_top_image"] = self.preview_renderer.render().copy()

            task_center = 0.5 * (
                data.xpos[self._pencil_body] + data.xpos[self._cup_body]
            )
            task_center[2] = max(0.84, float(task_center[2]))
            separation = float(
                np.linalg.norm(
                    data.xpos[self._pencil_body, :2] - data.xpos[self._cup_body, :2]
                )
            )
            insertion_camera = mujoco.MjvCamera()
            insertion_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            grip_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "panda/gripper0_grip_site"
            )
            if grip_site_id >= 0:
                camera_target = (
                    0.72 * task_center + 0.28 * data.site_xpos[grip_site_id]
                )
                camera_target[2] = max(0.90, float(camera_target[2]))
            else:
                camera_target = task_center
            insertion_camera.lookat[:] = camera_target
            insertion_camera.distance = max(1.00, 2.1 * separation)
            insertion_camera.azimuth = 225.0
            insertion_camera.elevation = -38.0
            self.preview_renderer.update_scene(
                data, camera=insertion_camera, scene_option=self.render_option
            )
            images["task_insertion_image"] = self.preview_renderer.render().copy()
        eef_base = np.linalg.inv(self._base_transform(data)) @ self._site_transform(data)
        axis_angle = Rotation.from_matrix(eef_base[:3, :3]).as_rotvec()
        finger_qpos = self.controller.finger_positions(data)
        state = np.asarray(
            [*eef_base[:3, 3], *axis_angle, *finger_qpos], dtype=np.float32
        )
        observation = {
            "time": float(data.time),
            "task_instruction": self.instruction,
            "prompt": self.instruction,
            "image": images["image"],
            "wrist_image": images["wrist_image"],
            "state": state,
            "joint_position": data.qpos[self.qpos_adrs].astype(np.float32).copy(),
            "joint_velocity": data.qvel[self.dof_adrs].astype(np.float32).copy(),
            "pencil_pose_world": _pose7(
                data.xpos[self._pencil_body],
                data.xmat[self._pencil_body].reshape(3, 3),
            ),
            "holder_pose_world": _pose7(
                data.xpos[self._cup_body],
                data.xmat[self._cup_body].reshape(3, 3),
            ),
        }
        pencil_axis = data.xmat[self._pencil_body].reshape(3, 3)[:, 0]
        observation["pencil_tilt_deg"] = np.asarray(
            math.degrees(math.acos(float(np.clip(abs(pencil_axis[2]), 0.0, 1.0)))),
            dtype=np.float32,
        )
        observation["pencil_holder_center_distance_xy"] = np.asarray(
            np.linalg.norm(
                data.xpos[self._pencil_body, :2] - data.xpos[self._cup_body, :2]
            ),
            dtype=np.float32,
        )
        if self.preview:
            observation["preview_image"] = images["preview_image"]
            observation["preview_wrist_image"] = images["preview_wrist_image"]
            observation["task_top_image"] = images["task_top_image"]
            observation["task_insertion_image"] = images["task_insertion_image"]
        return observation

    def _current_action(self) -> np.ndarray:
        current = self._site_transform(self.data)
        base_from_world = np.linalg.inv(self._base_transform(self.data))
        current_base = base_from_world @ current
        target_base = base_from_world @ self.target_eef
        translation = (
            target_base[:3, 3] - current_base[:3, 3]
        ) / OSC_POSITION_SCALE
        rotation = Rotation.from_matrix(
            target_base[:3, :3] @ current_base[:3, :3].T
        ).as_rotvec() / OSC_ROTATION_SCALE
        # LIBERO gripper convention: -1 opens, +1 closes.
        gripper = -1.0 if self.target_gripper >= 0.02 else 1.0
        return np.clip(
            np.asarray([*translation, *rotation, gripper], dtype=np.float32),
            -1.0,
            1.0,
        )

    def after_step(self, data: mujoco.MjData) -> dict[str, Any] | None:
        if float(data.time) + 1e-9 < self.next_sample_time:
            return None
        self.next_sample_time = float(data.time) + self.period
        observation = self._render_observation(data)
        connected = self._phone_origin is not None and bool(
            self.session.connected_clients
        )
        if connected and self.writer is not None:
            self.writer.append(observation, self._current_action())
        self._show_preview(observation)
        return observation

    def _show_preview(self, observation: dict[str, Any]) -> None:
        if not self.preview:
            return
        import cv2

        views = (
            ("AGENTVIEW (dataset raw orientation)", observation["preview_image"]),
            ("WRIST (dataset raw; display aligned)", observation["preview_wrist_image"]),
            ("TASK TOP (monitor)", observation["task_top_image"]),
            ("INSERTION CLOSE-UP (monitor)", observation["task_insertion_image"]),
        )
        tiles = []
        for name, frame in views:
            tile = frame.copy()
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 32), (18, 18, 18), -1)
            cv2.putText(
                tile, name, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            tiles.append(tile)
        canvas = np.concatenate(
            (np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)),
            axis=0,
        )
        label = (
            f"COLLECT  episode {self.episode_index}  "
            f"frames {0 if self.writer is None else self.writer.count}"
        )
        status = np.full((34, canvas.shape[1], 3), 18, dtype=np.uint8)
        cv2.putText(
            status, label, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        canvas = np.concatenate((status, canvas), axis=0)
        if not self._preview_window_initialized:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                PREVIEW_WINDOW, PREVIEW_TILE_SIZE * 2, PREVIEW_TILE_SIZE * 2 + 34
            )
            self._preview_window_initialized = True
        cv2.imshow(PREVIEW_WINDOW, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key != 0xFF:
            self.handle_collection_key(chr(key))

    def handle_collection_key(self, key: str) -> bool:
        key = str(key).lower()
        if key == "q":
            self.stop_requested = True
            return True
        if key == "g":
            self.toggle_gripper_from_viewer()
            return True
        if key not in {"s", "f"}:
            return False
        self._finish_episode(key == "s", "keyboard_label")
        self._reset_episode(self.data, respawn_task=True)
        return True

    def workspace_relocated(self) -> None:
        if self.writer is not None:
            self._discard_episode()
        self.target_eef = self.controller.site_transform(self.data)
        self.target_gripper = float(
            abs(self.controller.finger_positions(self.data)[0])
        )
        self.controller.nullspace_target = self.data.qpos[self.qpos_adrs].copy()
        self.controller.set_eef_target(self.target_eef)
        self.controller.set_gripper_opening(self.target_gripper)
        self._phone_origin = None
        self._eef_origin_base = None
        self._last_phone_toggle_state = None
        self._manual_gripper_closed = None
        self.next_sample_time = float(self.data.time)

    def _start_writer(self) -> None:
        if self.writer is not None:
            return
        agentview_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "panda/agentview"
        )
        wrist_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "panda/robot0_eye_in_hand"
        )
        self.writer = EpisodeWriter(
            self.dataset_dir,
            self.episode_index,
            self.instruction,
            self.rate_hz,
            self.task.get("seed"),
            {
                "agentview_camera": "panda/agentview",
                "agentview_parent_body": self.model.body(
                    int(self.model.cam_bodyid[agentview_id])
                ).name,
                "agentview_pos_local_xyz": self.model.cam_pos[agentview_id].copy(),
                "agentview_quat_local_wxyz": self.model.cam_quat[agentview_id].copy(),
                "agentview_fovy_deg": float(self.model.cam_fovy[agentview_id]),
                "agentview_image_hwc": np.asarray([256, 256, 3], dtype=np.int32),
                "wrist_camera": "panda/robot0_eye_in_hand",
                "wrist_fovy_deg": float(self.model.cam_fovy[wrist_id]),
                "wrist_image_hwc": np.asarray([256, 256, 3], dtype=np.int32),
            },
        )
        print(f"[Panda collect] recording episode {self.episode_index:06d}")

    def _finish_episode(self, success: bool | None, reason: str) -> None:
        if self.writer is None:
            return
        metrics = self._task_insertion_metrics(self.data)
        for name, value in metrics.items():
            self.writer.file.attrs[f"final_{name}"] = value
        path = self.writer.finish(success, reason)
        if path is not None:
            print(
                f"[Panda collect] saved {path} (success={success}, "
                f"geometry_success={bool(metrics['geometry_success'])}, "
                f"tilt={metrics['pencil_tilt_deg']:.1f} deg, "
                f"tip_depth={metrics['tip_depth_below_rim_m']:.3f} m)"
            )
            self.episode_index += 1
        self.writer = None

    def _discard_episode(self) -> None:
        if self.writer is not None:
            self.writer.discard()
            self.writer = None
            print("[Panda collect] current episode discarded")

    def _find_task_body(self, name_fragment: str) -> int:
        matches = [
            body_id
            for body_id in range(self.model.nbody)
            if name_fragment
            in _object_name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one Panda task body containing {name_fragment!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def _task_insertion_metrics(self, data: mujoco.MjData) -> dict[str, float | int]:
        pencil_position = data.xpos[self._pencil_body].copy()
        holder_position = data.xpos[self._cup_body].copy()
        pencil_axis = data.xmat[self._pencil_body].reshape(3, 3)[:, 0].copy()
        axis_z = float(pencil_axis[2])
        tilt_deg = math.degrees(math.acos(float(np.clip(abs(axis_z), 0.0, 1.0))))
        rim_z = float(holder_position[2] + HOLDER_BODY_HEIGHT)
        lower_tip_z = float(
            pencil_position[2] - PENCIL_COLLISION_HALF_LENGTH * abs(axis_z)
        )
        tip_depth = rim_z - lower_tip_z
        if abs(axis_z) > 1e-6:
            rim_parameter = (rim_z - float(pencil_position[2])) / axis_z
            rim_point = pencil_position + rim_parameter * pencil_axis
            rim_delta = rim_point - holder_position
            holder_rotation = data.xmat[self._cup_body].reshape(3, 3)
            rim_local = holder_rotation.T @ rim_delta
            rim_center_distance = float(np.linalg.norm(rim_local[:2]))
            rim_local_x = float(rim_local[0])
            rim_local_y = float(rim_local[1])
            rim_inside_opening = bool(
                abs(rim_local_x)
                <= HOLDER_INNER_HALF_X - PENCIL_COLLISION_RADIUS
                and abs(rim_local_y)
                <= HOLDER_INNER_HALF_Y - PENCIL_COLLISION_RADIUS
            )
        else:
            rim_center_distance = float("inf")
            rim_local_x = float("inf")
            rim_local_y = float("inf")
            rim_inside_opening = False

        linear_speed = 0.0
        angular_speed = 0.0
        dof_adr = int(self.model.body_dofadr[self._pencil_body])
        dof_num = int(self.model.body_dofnum[self._pencil_body])
        if dof_adr >= 0 and dof_num >= 6:
            linear_speed = float(np.linalg.norm(data.qvel[dof_adr:dof_adr + 3]))
            angular_speed = float(np.linalg.norm(data.qvel[dof_adr + 3:dof_adr + 6]))
        gripper_open = bool(self.controller.gripper_opening >= 0.03)
        geometry_success = bool(
            tilt_deg <= HOLDER_SAFE_TILT_DEG
            and tip_depth >= 0.02
            and rim_inside_opening
            and gripper_open
            and linear_speed <= 0.15
            and angular_speed <= 1.0
        )
        return {
            "geometry_success": int(geometry_success),
            "pencil_tilt_deg": float(tilt_deg),
            "safe_tilt_limit_deg": float(HOLDER_SAFE_TILT_DEG),
            "physical_tilt_limit_deg": float(HOLDER_MAX_STABLE_TILT_DEG),
            "tip_depth_below_rim_m": float(tip_depth),
            "rim_center_distance_m": float(rim_center_distance),
            "rim_local_x_m": float(rim_local_x),
            "rim_local_y_m": float(rim_local_y),
            "rim_inside_square_opening": int(rim_inside_opening),
            "pencil_linear_speed_mps": float(linear_speed),
            "pencil_angular_speed_radps": float(angular_speed),
            "gripper_open": int(gripper_open),
        }

    @staticmethod
    def _sample_task_sector(rng: np.random.Generator, sector: dict[str, Any]) -> np.ndarray:
        angle_deg = float(rng.uniform(*sector["angles_deg"]))
        profile = sector.get("reach_profile") or {}
        if profile:
            angles = np.asarray(sorted(float(value) for value in profile), dtype=float)
            radii = np.asarray([profile[value] for value in angles], dtype=float)
            max_radius = float(np.interp(angle_deg, angles, radii))
        else:
            max_radius = float(sector["max_radius"])
        min_radius = float(sector["min_radius"])
        radius = math.sqrt(float(rng.uniform(min_radius**2, max_radius**2)))
        angle = math.radians(angle_deg)
        return np.asarray(sector["center"], dtype=float) + radius * np.asarray(
            [math.cos(angle), math.sin(angle)]
        )

    def _set_task_body_pose(
        self,
        data: mujoco.MjData,
        body_id: int,
        position: np.ndarray,
        yaw_deg: float,
    ) -> None:
        half_yaw = math.radians(yaw_deg) / 2.0
        world_quat = np.asarray([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])
        joint_ids = range(
            int(self.model.body_jntadr[body_id]),
            int(self.model.body_jntadr[body_id] + self.model.body_jntnum[body_id]),
        )
        free_joint = next(
            (
                joint_id
                for joint_id in joint_ids
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            ),
            None,
        )
        if free_joint is not None:
            qpos_adr = int(self.model.jnt_qposadr[free_joint])
            dof_adr = int(self.model.jnt_dofadr[free_joint])
            data.qpos[qpos_adr:qpos_adr + 3] = position
            data.qpos[qpos_adr + 3:qpos_adr + 7] = world_quat
            data.qvel[dof_adr:dof_adr + 6] = 0.0
            return

        parent_id = int(self.model.body_parentid[body_id])
        parent_position = data.xpos[parent_id]
        parent_rotation = data.xmat[parent_id].reshape(3, 3)
        self.model.body_pos[body_id] = parent_rotation.T @ (position - parent_position)
        world_rotation = Rotation.from_quat(
            [world_quat[1], world_quat[2], world_quat[3], world_quat[0]]
        ).as_matrix()
        local_rotation = parent_rotation.T @ world_rotation
        local_quat_xyzw = Rotation.from_matrix(local_rotation).as_quat()
        self.model.body_quat[body_id] = [
            local_quat_xyzw[3],
            local_quat_xyzw[0],
            local_quat_xyzw[1],
            local_quat_xyzw[2],
        ]

    def _respawn_task_objects(self, data: mujoco.MjData) -> None:
        pencil_xy = self._sample_task_sector(self._task_rng, self.task["pencil_sector"])
        minimum_separation = float(self.task.get("object_min_separation", 0.20))
        for _ in range(1_000):
            cup_xy = self._sample_task_sector(self._task_rng, self.task["cup_sector"])
            if np.linalg.norm(cup_xy - pencil_xy) >= minimum_separation:
                break
        else:
            raise RuntimeError("could not respawn separated Panda task objects")

        pencil_radius = float(
            np.linalg.norm(pencil_xy - np.asarray(self.task["pencil_sector"]["center"]))
        )
        yaw_range = self.task.get("pencil_yaw_range_deg", (-90.0, 90.0))
        outer_radius = self.task.get("pencil_outer_yaw_radius")
        if outer_radius is not None and pencil_radius > float(outer_radius):
            yaw_range = self.task.get("pencil_outer_yaw_range_deg", yaw_range)
        pencil_yaw = float(self._task_rng.uniform(*yaw_range))
        cup_yaw = float(
            self._task_rng.uniform(*self.task.get("cup_yaw_range_deg", (-15.0, 15.0)))
        )
        pencil_z = float(self.task.get("pencil_spawn_z", 0.8167))
        cup_z = float(self.task.get("cup_spawn_z", 0.8016))
        self._set_task_body_pose(
            data, self._pencil_body, np.asarray([*pencil_xy, pencil_z]), pencil_yaw
        )
        self._set_task_body_pose(
            data, self._cup_body, np.asarray([*cup_xy, cup_z]), cup_yaw
        )
        self.task["pencil_xy"] = pencil_xy
        self.task["cup_xy"] = cup_xy
        self.task["seed"] = int(self._task_rng.integers(0, 2**32, dtype=np.uint64))

        for site_name, xyz in (
            ("panda_task_pencil_spawn", [*pencil_xy, pencil_z + 0.017]),
            ("panda_task_cup_drop_target", [*cup_xy, cup_z + 0.230]),
        ):
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                self.model.site_pos[site_id] = xyz
        mujoco.mj_forward(self.model, data)
        print(
            "[Panda collect] NEW TASK: "
            f"pencil=({pencil_xy[0]:.3f}, {pencil_xy[1]:.3f}) "
            f"yaw={pencil_yaw:.1f} deg; holder=({cup_xy[0]:.3f}, {cup_xy[1]:.3f})"
        )

    def _reset_episode(self, data: mujoco.MjData, *, respawn_task: bool = False) -> None:
        data.qpos[:] = self.initial_qpos
        data.qvel[:] = self.initial_qvel
        data.ctrl[:] = self.initial_ctrl
        data.time = 0.0
        mujoco.mj_forward(self.model, data)
        if respawn_task:
            self._respawn_task_objects(data)
        mujoco.mj_forward(self.model, data)
        self.target_eef = self._site_transform(data)
        self.target_gripper = float(abs(self.controller.finger_positions(data)[0]))
        self.controller.nullspace_target = data.qpos[self.qpos_adrs].copy()
        self.controller.set_eef_target(self.target_eef)
        self.controller.set_gripper_opening(self.target_gripper)
        self._phone_origin = None
        self._eef_origin_base = None
        self._last_phone_toggle_state = None
        self._manual_gripper_closed = None
        self._last_reported_phone_rotation_deg[:] = 0.0
        self._filtered_phone_pitch = 0.0
        self._last_phone_pitch_time = 0.0
        self._last_button = True
        self._wait_for_button_release = True
        self._last_secondary_button = True
        self._wait_for_secondary_button_release = True
        self.next_sample_time = 0.0

    def close(self) -> None:
        if self.writer is not None:
            self._finish_episode(False, "shutdown_partial")
        self.renderer.close()
        if self.preview_renderer is not None:
            self.preview_renderer.close()
        if self.preview:
            try:
                import cv2

                cv2.destroyWindow(PREVIEW_WINDOW)
            except Exception:
                pass
        if self.session is not None:
            # teledex 0.0.7 Session.stop() can deadlock; close its listener and loop directly.
            try:
                import asyncio

                async def close_server() -> None:
                    for client in list(self.session.connected_clients):
                        await client.close()
                    if self.session.server is not None:
                        self.session.server.close()
                        await self.session.server.wait_closed()
                        self.session.server = None

                future = asyncio.run_coroutine_threadsafe(
                    close_server(), self.session.loop
                )
                future.result(timeout=2.0)
                self.session.loop.call_soon_threadsafe(self.session.loop.stop)
                self.session._thread.join(timeout=2.0)
                if self.session._thread.is_alive():
                    print("[Panda collect] TeleDex daemon did not exit in 2 s; process shutdown will release it")
            except Exception as exc:
                print(f"[Panda collect] TeleDex shutdown warning: {exc}")


def inspect_dataset(path: Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    files = sorted(root.glob("episode_*.h5"))
    summary = {"root": str(root), "episodes": len(files), "frames": 0, "success": 0, "failed": 0, "unknown": 0}
    required_without_action = (
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "diagnostics/joint_position",
        "diagnostics/joint_velocity",
    )
    for path in files:
        with h5py.File(path, "r") as file:
            schema = str(file.attrs.get("schema", ""))
            if schema != SCHEMA_VERSION and schema not in LEGACY_SCHEMA_VERSIONS:
                raise RuntimeError(f"{path}: unexpected schema {file.attrs.get('schema')!r}")
            action_key = "actions" if "actions" in file else "action"
            required = (*required_without_action, action_key)
            for name in required:
                if name not in file:
                    raise RuntimeError(f"{path}: missing {name}")
            if file["observation/image"].shape[1:] != (256, 256, 3):
                raise RuntimeError(f"{path}: exterior image is not 256x256 RGB")
            if file["observation/wrist_image"].shape[1:] != (256, 256, 3):
                raise RuntimeError(f"{path}: wrist image is not 256x256 RGB")
            if file["observation/state"].shape[1:] != (8,):
                raise RuntimeError(f"{path}: state is not LIBERO 8-D")
            if file[action_key].shape[1:] != (7,):
                raise RuntimeError(f"{path}: action is not LIBERO 7-D")
            lengths = {name: file[name].shape[0] for name in required}
            if len(set(lengths.values())) != 1:
                raise RuntimeError(f"{path}: inconsistent sequence lengths {lengths}")
            summary["frames"] += next(iter(lengths.values()))
            label = int(file.attrs.get("success", -1))
            summary["success" if label == 1 else "failed" if label == 0 else "unknown"] += 1
    return summary
