"""Replay the 28 saved initial RGBs through frozen H20/H18 perception sources.

Run in the haic_rai301 environment. No simulator or GPU is used. This script
writes only beside itself, never to a custom source directory. On its first run
it snapshots the two source inputs and records image/source SHA256 hashes. Later
runs reuse that exact input manifest and reject changed snapshots or images.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True

import cv2
import numpy as np
import scipy


SOURCE_INPUTS = {
    "h20": "/home/aril/RAI-301-m5/src/custom/mission_5/perception.py",
    "h18_branch_aligned": "/home/aril/RAI-301-m5-stable/src/custom/mission_5/perception.py",
}
RUNS = ("baseline", "candidate-v1", "final-unseen")
ORIGINAL_REPORTED = {"count": 28, "max_abs_directed_yaw_delta_deg": 0.8131923333377293,
                     "over90deg": 0, "max_center_delta_mm": 32.11720380725058}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_snapshot(name, path):
    spec = importlib.util.spec_from_file_location("perception_replay_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def freeze_inputs(script_dir, run_root):
    snapshots = script_dir / "perception_replay_sources"
    snapshots.mkdir(exist_ok=True)
    sources = {}
    for name, original in SOURCE_INPUTS.items():
        original = Path(original)
        snapshot = snapshots / (name + ".py")
        # The only source writes are these copies inside the analysis directory.
        snapshot.write_bytes(original.read_bytes())
        sources[name] = {"original_source_path": str(original),
                         "original_source_sha256": digest(original),
                         "snapshot_relative_path": str(snapshot.relative_to(script_dir)),
                         "snapshot_sha256": digest(snapshot),
                         "size_bytes": snapshot.stat().st_size}
    images = []
    for run in RUNS:
        for path in sorted((run_root / run).glob("case_*/trace/00000_agent.png")):
            images.append({"relative_path": str(path.relative_to(run_root)),
                           "original_image_path": str(path.resolve()),
                           "sha256": digest(path), "size_bytes": path.stat().st_size})
    if len(images) != ORIGINAL_REPORTED["count"]:
        raise ValueError(f"Expected the original28 inputs, found{len(images)}")
    return {"original_run_root": str(run_root.resolve()), "sources": sources, "images": images}


def verify_inputs(script_dir, run_root, inputs):
    for name, source in inputs["sources"].items():
        path = script_dir / source["snapshot_relative_path"]
        if digest(path) != source["snapshot_sha256"]:
            raise ValueError(f"Source snapshot changed: {name}")
        if source["snapshot_sha256"] != source["original_source_sha256"]:
            raise ValueError(f"Snapshot was not identical to original source: {name}")
    for image in inputs["images"]:
        path = run_root / image["relative_path"]
        if digest(path) != image["sha256"]:
            raise ValueError(f"Input RGB changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--run-root", type=Path, default=script_dir.parent,
                        help="Root holding baseline/, candidate-v1/, final-unseen/")
    args = parser.parse_args()
    output = script_dir / "perception_replay.json"
    if output.exists():
        inputs = json.loads(output.read_text())["inputs"]
    else:
        inputs = freeze_inputs(script_dir, args.run_root)
    verify_inputs(script_dir, args.run_root, inputs)
    modules = {name: import_snapshot(name, script_dir / source["snapshot_relative_path"])
               for name, source in inputs["sources"].items()}
    old, new = modules["h20"], modules["h18_branch_aligned"]
    results = []
    for image in inputs["images"]:
        path = args.run_root / image["relative_path"]
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        a = old.measure_scene(rgb, old.CameraModel.panda_agentview_default())
        b = new.measure_scene(rgb, new.CameraModel.panda_agentview_default())
        difference = b.pencil_yaw - a.pencil_yaw
        yaw_delta = float(np.degrees(np.arctan2(np.sin(difference), np.cos(difference))))
        displacement = b.pencil_xy - a.pencil_xy
        results.append({"relative_image_path": image["relative_path"], "image_sha256": image["sha256"],
                        "run": path.parents[2].name, "case": path.parents[1].name,
                        "old_h20": a.as_dict(), "new_h18": b.as_dict(),
                        "directed_yaw_delta_deg": yaw_delta,
                        "center_delta_xy_m": displacement.tolist(),
                        "center_delta_mm": float(np.linalg.norm(displacement) * 1000)})
    summary = {"count": len(results),
               "max_abs_directed_yaw_delta_deg": max(abs(row["directed_yaw_delta_deg"]) for row in results),
               "over90deg": sum(abs(row["directed_yaw_delta_deg"]) > 90 for row in results),
               "max_center_delta_mm": max(row["center_delta_mm"] for row in results)}
    agrees = all(bool(np.isclose(summary[key], value, rtol=0, atol=1e-6))
                 for key, value in ORIGINAL_REPORTED.items())
    camera = new.CameraModel.panda_agentview_default()
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Reproducible sensor-only offline perception comparison, not task success evaluation",
              "script_sha256": digest(__file__), "inputs": inputs,
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "opencv": cv2.__version__, "scipy": scipy.__version__},
              "camera": {"intrinsics_K": camera.k.tolist(),
                         "base_from_optical": camera.base_from_optical.tolist(),
                         "projection_plane_z_m": .0145 * 1.7},
              "method": {
                  "old": "Unmodified saved H20 perception source snapshot",
                  "new": "H18 body mask and center; new PCA axis is pi-aligned to PCA of >=30 H20 pixels in the same H18 component; existing pink-eraser resolution retained",
                  "directed_yaw_delta": "atan2(sin(new-old), cos(new-old)), in degrees",
                  "center_delta": "Euclidean displacement between old/new sensor-estimated base-plane centers; not error against ground truth"},
              "limitations": [
                  "The28 images are saved initial views from prior runs; they are not independent successful task trials.",
                  "No hidden object positions, randomized scenario fields, referee measurements, simulator, GPU, or learned policy outputs are used.",
                  "A center shift demonstrates color-mask sensitivity, not improved absolute localization accuracy or a causal explanation of task failure.",
                  "No heading reversal occurred on these28 inputs; this is not a guarantee for unseen colors, connected-component merges, or occlusions.",
                  "H20 points inside one expanded H18 component provide the branch reference. Newly joined old components can therefore change the reference PCA.",
                  "Pink eraser evidence can legitimately override the raw PCA branch; <30 H20 points retain the H18 raw PCA branch.",
                  "Reproduction checks source snapshots and every RGB SHA256. Original source paths are provenance only and may later change."],
              "summary": summary, "original_reported_summary": ORIGINAL_REPORTED,
              "matches_original_report_within_1e_minus_6": agrees, "results": results}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "summary": summary,
                      "matches_original_report": agrees,
                      "source_sha256": {name: source["snapshot_sha256"]
                                        for name, source in inputs["sources"].items()}}, indent=2))
    if not agrees:
        raise SystemExit("Replay differed from the original report")


if __name__ == "__main__":
    main()
