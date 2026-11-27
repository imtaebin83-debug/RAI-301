from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from ros2_bridge.tools.g1_walk_policy_29dof import JOINTS, KD, KP, G1WalkPolicy29DoF


class G1JointCommandSwitch29DoF:
    """Walking policy that yields the 29 joints to external joint targets while they arrive."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        walk_policy_path: str | Path,
        prefix: str = "",
        timeout: float = 0.2,
    ) -> None:
        self.model = model
        self.data = data
        self.walk = G1WalkPolicy29DoF(model, data, walk_policy_path, prefix=prefix, cmd=(0.0, 0.0, 0.0))
        self.timeout = max(0.0, float(timeout))
        self.index = {name: i for i, name in enumerate(JOINTS)}
        self.external = False
        self.target = np.zeros(len(JOINTS), dtype=float)
        self.kp = KP.copy()
        self.kd = KD.copy()
        self._stamp = 0.0
        act = self.walk.act
        limited = model.actuator_ctrllimited[act].astype(bool)
        self._low = np.where(limited, model.actuator_ctrlrange[act, 0], -np.inf)
        self._high = np.where(limited, model.actuator_ctrlrange[act, 1], np.inf)

    def __getattr__(self, name: str):
        if name == "walk" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.walk, name)

    def _indices(self, names: Sequence[str]) -> np.ndarray:
        unknown = [name for name in names if name not in self.index]
        if unknown:
            raise ValueError(f"unknown G1 joints: {unknown}")
        return np.asarray([self.index[name] for name in names], dtype=np.int32)

    def _apply_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        for actuator, p, d in zip(self.walk.act, kp, kd):
            self.model.actuator_gainprm[actuator, 0] = p
            self.model.actuator_biasprm[actuator, 1] = -p
            self.model.actuator_biasprm[actuator, 2] = -d

    def set_gains(self, names: Sequence[str], kp: Sequence[float], kd: Sequence[float]) -> None:
        index = self._indices(names)
        kp = np.asarray(kp, dtype=float)
        kd = np.asarray(kd, dtype=float)
        if kp.shape != index.shape or kd.shape != index.shape:
            raise ValueError("gains need one kp (position) and one kd (velocity) per name")
        if not (np.all(np.isfinite(kp)) and np.all(np.isfinite(kd))) or np.any(kp < 0) or np.any(kd < 0):
            raise ValueError("gains must be finite and non-negative")
        self.kp[index] = kp
        self.kd[index] = kd
        if self.external:
            self._apply_gains(self.kp, self.kd)

    def command(self, names: Sequence[str], positions: Sequence[float]) -> None:
        index = self._indices(names)
        positions = np.asarray(positions, dtype=float)
        if positions.shape != index.shape or not np.all(np.isfinite(positions)):
            raise ValueError("joint command needs one finite position per name")
        if not self.external:
            self.external = True
            self.target[:] = self.data.qpos[self.walk.qadr]
            self._apply_gains(self.kp, self.kd)
            dadr = self.walk.root_dadr
            self.data.qfrc_applied[dadr:dadr + 6] = 0.0
        self.target[index] = positions
        self._stamp = float(self.data.time)

    def release(self) -> None:
        if not self.external:
            return
        self.external = False
        self._apply_gains(KP, KD)
        self.walk.adopt_current_state()

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        if not self.external:
            self.walk.set_command(vx, vy, wz)

    def reset(self, pos: Optional[Sequence[float]] = None, yaw: float = 0.0) -> None:
        if self.external:
            self.external = False
            self._apply_gains(KP, KD)
        self.walk.reset(pos=pos, yaw=yaw)

    def control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.external and not 0.0 <= float(data.time) - self._stamp <= self.timeout:
            self.release()
        if self.external:
            data.ctrl[self.walk.act] = np.clip(self.target, self._low, self._high)
            return
        self.walk.control(model, data)

    def install(self) -> None:
        mujoco.set_mjcb_control(self.control)

    @staticmethod
    def uninstall() -> None:
        mujoco.set_mjcb_control(None)
