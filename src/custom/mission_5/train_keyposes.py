"""Fit object-relative pose/gripper/timing prototypes from actual train episodes.

Only measured robot states and actions are fit. Teacher debug target poses are
never read. Validation/test episodes are used for offline residual reporting,
never for reference selection, means, phase timing or controller tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .bc.data import read_manifest
from .keypose_policy import FORMAT, materialize, yaw_rotation


def extract_episode(path):
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("official_record", {}).get("success") is not True:
        raise ValueError("Episode lacks original referee PASS: " + path.name)
    events = metadata["events"]
    scene = None
    for event in events:
        d = event.get("debug", {})
        if all(key in d for key in ("pencil_xy", "holder_xy", "pencil_yaw")):
            scene = {key: d[key] for key in ("pencil_xy", "holder_xy", "pencil_yaw")}
            break
    if scene is None or not np.isfinite([*scene["pencil_xy"], *scene["holder_xy"], scene["pencil_yaw"]]).all():
        raise ValueError("No finite initial sensor object measurement")
    with h5py.File(path, "r") as f:
        if not bool(f.attrs.get("official_success", False)):
            raise ValueError("HDF5 does not confirm referee success")
        states = np.asarray(f["observation/state"], float)
        actions = np.asarray(f["actions"], float)
        times = np.asarray(f["time"], float)
        camera = json.loads(f.attrs["camera_metadata_json"])["agentview"]
        fps = float(f.attrs["fps"])
    if states.shape[1:] != (8,) or actions.shape != (len(states), 7):
        raise ValueError("Unexpected measured-state/action schema")
    phases = []
    for i, event in enumerate(events[:-1]):
        if event["stage"] in ("observe", "done"):
            continue
        next_event = events[i + 1]
        endstate = np.asarray(next_event["state"], float)
        if next_event["stage"] == "done":
            # Collector terminates before appending its final done observation.
            # Fit the last recorded sensor state, never an unrecorded target.
            end = len(states) - 1
            endstate = states[end].copy()
        else:
            distances = np.linalg.norm(states - endstate, axis=1)
            if distances.min() > 1e-5:
                raise ValueError("Event endpoint is not a measured HDF5 robot state")
            end = int(np.argmin(distances))
        duration = float(next_event["seconds"] - event["seconds"])
        begin = max(0, end - max(1, int(round(duration * fps))))
        grip = -1. if np.median(actions[begin:end + 1, 6]) < 0 else 1.
        phases.append({"name": event["stage"], "state": endstate,
                       "grip": grip, "duration": duration})
    if not phases or any(p["duration"] <= 0 for p in phases):
        raise ValueError("Missing or invalid demonstration phases")
    return {"scene": scene, "phases": phases, "fps": fps, "camera": camera,
            "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def canonical_pose(state, scene, reference):
    rotation = yaw_rotation(scene["pencil_yaw"])
    offset = rotation.T @ (np.asarray(state[:3]) - np.r_[scene[reference + "_xy"], 0.])
    orientation = rotation.T @ Rotation.from_rotvec(state[3:6]).as_matrix()
    return offset, orientation


def fit_model(episodes, *, manifest_sha256):
    if not episodes:
        raise ValueError("No successful training episodes")
    names = [p["name"] for p in episodes[0]["phases"]]
    if any([p["name"] for p in e["phases"]] != names for e in episodes):
        raise ValueError("Incompatible measured phase orders")
    phases = []
    for index, name in enumerate(names):
        candidates = {}
        for reference in ("pencil", "holder"):
            poses = [canonical_pose(e["phases"][index]["state"], e["scene"], reference) for e in episodes]
            offsets = np.asarray([p[0] for p in poses])
            candidates[reference] = (float(np.var(offsets, axis=0).sum()), offsets,
                                     Rotation.from_matrix(np.asarray([p[1] for p in poses])))
        reference = min(candidates, key=lambda k: candidates[k][0])
        variance, offsets, rotations = candidates[reference]
        grips = [e["phases"][index]["grip"] for e in episodes]
        phases.append({"name": name, "reference": reference,
                       "offset_xyz": np.mean(offsets, axis=0).tolist(),
                       "quaternion_xyzw": rotations.mean().as_quat().tolist(),
                       "grip": -1. if np.median(grips) < 0 else 1.,
                       "minimum_seconds": float(np.median([e["phases"][index]["duration"] for e in episodes])),
                       "reference_variance_m2": {k: v[0] for k, v in candidates.items()},
                       "endpoint_std_m": offsets.std(0).tolist(), "samples": len(episodes)})
    camera = episodes[0]["camera"]
    if any(e["camera"] != camera for e in episodes):
        raise ValueError("Training cameras differ")
    focal = 128 / math.tan(math.radians(camera["fovy"]) / 2)
    return {"format": FORMAT, "fps": float(np.median([e["fps"] for e in episodes])), "phases": phases,
            "policy_inputs": ["image", "state"], "unused_optional_input": "wrist_image",
            "camera": {"k": [[focal, 0, 127.5], [0, focal, 127.5], [0, 0, 1]],
                       "position": camera["pos"], "quaternion_wxyz": camera["quat"]},
            "controller": {"position_gain": .65, "orientation_gain": .65,
                           "max_position_step_m": .015, "max_rotation_step_rad": .10,
                           "position_tolerance_m": .006, "orientation_tolerance_rad": .08,
                           "stage_timeout_scale": 3., "stage_timeout_min": 12.},
            "training": {"method": "train-only object-relative pose/gripper/timing prototypes",
                         "manifest_sha256": manifest_sha256,
                         "episode_sha256": {e["file"]: e["sha256"] for e in episodes},
                         "target_source": "next-phase measured HDF5 robot state, never debug target_xyz",
                         "phase_reference_selection": "minimum train endpoint variance",
                         "inductive_bias": "object-relative SE(3) frames and generic finite-state pose servo"}}


def residuals(model, episodes):
    distances, angles = [], []
    for episode in episodes:
        for learned, observed in zip(model["phases"], episode["phases"]):
            if learned["name"] != observed["name"]:
                raise ValueError("Evaluation phase order differs")
            pose = materialize(learned, episode["scene"])
            distances.append(float(np.linalg.norm(pose[:3, 3] - observed["state"][:3])))
            angles.append(float((Rotation.from_matrix(pose[:3, :3]) *
                                 Rotation.from_rotvec(observed["state"][3:6]).inv()).magnitude()))
    return {"episodes": len(episodes), "position_rmse_m": float(np.sqrt(np.mean(np.square(distances)))) if distances else None,
            "orientation_mean_rad": float(np.mean(angles)) if angles else None,
            "official_policy_success": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest, root, records = read_manifest(args.manifest)
    data = {}
    for split in ("train", "val", "test"):
        data[split] = []
        for name in manifest["splits"][split]:
            episode = extract_episode(root / name)
            if episode["sha256"] != records[name]["sha256"]:
                raise ValueError("Frozen manifest demonstration changed")
            data[split].append(episode)
    model = fit_model(data["train"], manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2))
    report = {split: residuals(model, rows) for split, rows in data.items()}
    args.out.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"checkpoint": str(args.out), "phases": [(p["name"], p["reference"]) for p in model["phases"]],
                      "residuals": report}, indent=2))


if __name__ == "__main__":
    main()
