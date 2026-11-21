"""Audit class HDF5 episodes and split whole initial-condition groups."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

SCHEMAS = {f"val861-libero-panda-hdf5-v{i}" for i in (2, 3, 4)} | {"m5-two-camera-episodes-v1"}
INPUT_KEYS = ("observation/image", "observation/wrist_image", "observation/state")


def audit_episode(path: Path):
    with h5py.File(path, "r") as f:
        if str(f.attrs.get("schema", "")) not in SCHEMAS:
            raise ValueError("unrecognized class dataset schema")
        action_key = "actions" if "actions" in f else "action"
        keys = (*INPUT_KEYS, action_key, "time")
        for key in keys:
            if key not in f:
                raise ValueError(f"missing {key}")
        n = len(f[action_key])
        if n < 2 or any(len(f[k]) != n for k in keys):
            raise ValueError("empty/short episode or inconsistent sequence lengths")
        for k in INPUT_KEYS[:2]:
            if f[k].shape[1:] != (256, 256, 3) or f[k].dtype != np.uint8:
                raise ValueError(f"{k} must be uint8 (T,256,256,3)")
        state, action, times = f[INPUT_KEYS[2]][:], f[action_key][:], f["time"][:]
        if state.shape != (n, 8) or action.shape != (n, 7):
            raise ValueError("expected state (T,8), action (T,7)")
        if not all(np.isfinite(x).all() for x in (state, action, times)):
            raise ValueError("NaN/Inf in state/action/time")
        if times.shape != (n,) or np.any(np.diff(times) <= 0):
            raise ValueError("time must be strictly increasing")
        if np.max(np.abs(action)) > 1.0001:
            raise ValueError("actions must be normalized to [-1,1]")
        for key in ("state_frame", "action_frame"):
            if str(f.attrs.get(key, "panda_base")) != "panda_base":
                raise ValueError(f"unsupported {key}; expected panda_base")
        for attr, expected in (("action_position_scale_m", .05), ("action_rotation_scale_rad", .5)):
            if attr in f.attrs and not np.isclose(float(f.attrs[attr]), expected):
                raise ValueError(f"unexpected {attr}")
        fps = float(f.attrs.get("fps", 1 / np.median(np.diff(times))))
        actual_fps = float(1 / np.median(np.diff(times)))
        if not np.isfinite(fps) or fps <= 0 or abs(actual_fps / fps - 1) > .2:
            raise ValueError("recorded timestamps and fps disagree by more than 20%")
        # Only grouping uses initial object poses, when available; they never enter a policy tensor.
        pose_keys = ("diagnostics/pencil_pose_world", "diagnostics/holder_pose_world")
        if "placement_id" in f.attrs:
            group = "placement:" + str(f.attrs["placement_id"])
        elif all(k in f for k in pose_keys):
            initial = np.concatenate([np.asarray(f[k][0], np.float64) for k in pose_keys])
            group = "pose:" + hashlib.sha256(np.round(initial, 3).tobytes()).hexdigest()
        else:
            # No pose metadata: conservatively keep one seeded collection session together.
            seed = int(f.attrs.get("task_seed", -1))
            if seed >= 0:
                group = f"seed:{seed}"
            else:
                initial = state[0].tobytes() + b"".join(f[k][0].tobytes() for k in INPUT_KEYS[:2])
                group = "observation:" + hashlib.sha256(initial).hexdigest()
        camera_attrs = {k: np.asarray(v).tolist() for k, v in f.attrs.items()
                        if k.startswith(("agentview_", "wrist_"))
                        or any(word in k.lower() for word in ("camera", "sensors"))}
        camera_signature = hashlib.sha256(json.dumps(camera_attrs, sort_keys=True).encode()).hexdigest()
        success=int(f.attrs.get("success",-1))
        if success not in (-1,0,1):
            raise ValueError("success must be -1 unknown, 0 failed, or 1 successful")
        verified=f.attrs.get("official_success",None)
        if verified is not None and (not isinstance(verified,(bool,int,np.bool_,np.integer)) or int(verified) not in (0,1)):
            raise ValueError('official_success must be a boolean verdict, not text')
        if verified is not None and success==1 and not bool(verified):
            raise ValueError("success label disagrees with the official failure verdict")
        return {"file": path.name, "frames": n, "success": success,
                "fps": fps, "action_key": action_key, "group": group,
                "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "official_success":None if verified is None else bool(verified),
                "camera_signature": camera_signature, "camera_metadata": camera_attrs}


def prepare(root: Path, out: Path, seed=42):
    root = root.resolve()
    records, errors = [], []
    for p in sorted(root.rglob("episode_*.h5")):
        if p.name.endswith('.recording.h5'):
            continue  # Collector commits episodes by atomic rename after close.
        try:
            record=audit_episode(p)
            record['file']=str(p.relative_to(root))
            records.append(record)
        except Exception as e:
            errors.append({"file": p.name, "error": str(e)})
    successful = [r for r in records if r["success"] == 1]
    groups = sorted({r["group"] for r in successful})
    report = {"version": 1, "root": str(root), "split_seed": seed, "episodes": records,
              "errors": errors, "counts": {"total": len(records), "successful": len(successful),
              "failed": sum(r["success"] == 0 for r in records), "groups": len(groups)},
              "policy_inputs": list(INPUT_KEYS), "splits": {"train": [], "val": [], "test": []},
              "notes": ["Success is the recorder label; not an independently verified official verdict.",
                        "Initial diagnostic poses are used only for grouping; excluded from model inputs."]}
    if errors:
        report["ready"] = False
    elif len(groups) < 3:
        report["ready"] = False
        report["notes"].append("Need at least three distinct successful initial-condition groups.")
    elif len({r["camera_signature"] for r in successful}) > 1:
        report["ready"] = False
        report["notes"].append("Camera metadata changed: collect a consistent set or explicitly design domain randomization.")
    elif max(r["fps"] for r in successful) - min(r["fps"] for r in successful) > .5:
        report["ready"] = False
        report["notes"].append("Collection rates differ; do not mix action time scales.")
    else:
        np.random.default_rng(seed).shuffle(groups)
        n_val = max(1, int(len(groups) * .1))
        n_test = max(1, int(len(groups) * .1))
        group_split = {g: "test" if i < n_test else "val" if i < n_test + n_val else "train"
                       for i, g in enumerate(groups)}
        for r in successful:
            report["splits"][group_split[r["group"]]].append(r["file"])
        report["ready"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    report = prepare(args.data_dir, args.out, args.seed)
    print(json.dumps({k: report[k] for k in ("ready", "counts", "errors", "notes")}, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
