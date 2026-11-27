from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

SENSOR_DEFAULTS: dict[str, Any] = {
    "g1": {
        "camera": {"pitch_deg": 15.0, "forward_m": 0.0, "up_m": 0.0},
        "lidar": {"up_m": 0.0},
    },
    "panda": {
        "agentview": {"pos": [1.28, 0.0, 0.92],
                      "quat": [0.66331780, 0.24496836, 0.24496836, 0.66331780], "fovy": 46.0},
        "wrist": {"pos": [0.05, 0.0, 0.0], "quat": [0.0, 0.707108, 0.707108, 0.0], "fovy": 75.0},
    },
}


def load_sensors(path: Optional[str | Path]) -> dict[str, Any]:
    import copy

    config = copy.deepcopy(SENSOR_DEFAULTS)
    if path is None:
        return config
    import yaml

    loaded = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    if "sensors" in loaded:          # custom/robot.yaml keeps the placement under "sensors:"
        loaded = loaded["sensors"] or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: sensors: expected a mapping")
    for robot, sensors in loaded.items():
        if robot not in config or not isinstance(sensors, dict):
            raise ValueError(f"{path}: unknown section {robot!r} (g1 | panda)")
        for sensor, values in sensors.items():
            if sensor not in config[robot] or not isinstance(values, dict):
                raise ValueError(f"{path}: unknown sensor {robot}.{sensor!r} "
                                 f"({', '.join(config[robot])})")
            for key, value in values.items():
                if key not in config[robot][sensor]:
                    raise ValueError(f"{path}: unknown key {robot}.{sensor}.{key!r} "
                                     f"({', '.join(config[robot][sensor])})")
                config[robot][sensor][key] = value
    for camera in ("agentview", "wrist"):
        spec = config["panda"][camera]
        spec["pos"] = [float(v) for v in spec["pos"]]
        spec["quat"] = [float(v) for v in spec["quat"]]
        spec["fovy"] = float(spec["fovy"])
        if len(spec["pos"]) != 3 or len(spec["quat"]) != 4:
            raise ValueError(f"{path}: panda.{camera}: pos takes 3 values, quat 4 (w, x, y, z)")
    g1 = config["g1"]
    g1["camera"] = {k: float(v) for k, v in g1["camera"].items()}
    g1["lidar"]["up_m"] = float(g1["lidar"]["up_m"])
    return config


def sensors_signature(config: dict[str, Any]) -> str:
    import hashlib
    import json

    return hashlib.sha1(json.dumps(config["g1"], sort_keys=True).encode()).hexdigest()[:12]
