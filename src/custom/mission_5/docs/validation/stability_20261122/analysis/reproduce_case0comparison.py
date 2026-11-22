"""Reproduce the saved case-0 RGB/proprioception comparison; no simulator access.

Run with the haic_rai301 environment. Inputs are read-only. The JSON output is
always written beside this script, outside the competition custom source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.spatial.transform import Rotation


KA = np.array([[301.5491028, 0, 127.5], [0, 301.5491028, 127.5], [0, 0, 1.]])
KW = np.array([[166.8128477, 0, 127.5], [0, 166.8128477, 127.5], [0, 0, 1.]])
OPTICAL_FLIP = np.diag([1., -1., -1.])
RA = Rotation.from_quat([.24496836, .24496836, .6633178, .6633178]).as_matrix() @ OPTICAL_FLIP
PA = np.array([1.28, 0., .92])
T_EEF_HAND = np.eye(4)
T_EEF_HAND[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
T_EEF_HAND[:3, 3] = [0., 0., -.097]
T_HAND_WRIST = np.eye(4)
T_HAND_WRIST[:3, :3] = Rotation.from_quat([.707108, .707108, 0., 0.]).as_matrix() @ OPTICAL_FLIP
T_HAND_WRIST[:3, 3] = [.05, 0., 0.]

FRAMES = {
    "baseline": {"descend_end": 70, "close_end": 145, "lift_end": 210,
                 "upright_end": 245, "carry_end": 315, "insert_end": 335,
                 "release_start": 340, "release_late": 350},
    "candidate-v1": {"descend_end": 70, "close_end": 80, "lift_end": 145,
                     "upright_end": 180, "carry_end": 255, "insert_end": 275,
                     "release_start": 280, "release_late": 285,
                     "retreat_start": 290, "retreat_end": 310},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yellow_line(path: Path):
    """Largest elongated original-yellow-pixel component after mask dilation."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42)
            & (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 155))
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(mask.astype(np.uint8), np.ones((3, 7), np.uint8)))
    candidates = []
    for index in range(1, count):
        yy, xx = np.nonzero((labels == index) & mask)
        if len(xx) < 15:
            continue
        points = np.stack([xx, yy], axis=1)
        center = points.mean(axis=0)
        _, singular, vt = np.linalg.svd(points - center, full_matrices=False)
        if singular[0] < 2 * singular[1]:
            continue
        dx, dy = vt[0]
        cx, cy = center
        candidates.append({"pixels": len(xx),
                           "line": np.array([-dy, dx, dy * cx - dx * cy]),
                           "center_px": center,
                           "elongation": float(singular[0] / max(singular[1], 1e-12))})
    return max(candidates, key=lambda item: item["pixels"]) if candidates else None


def degrees_abs_dot(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0., 1.))))


def analyze_frame(trace_dir: Path, row, command: int):
    state = np.asarray(row["state"], dtype=float)
    t_base_eef = np.eye(4)
    t_base_eef[:3, :3] = Rotation.from_rotvec(state[3:6]).as_matrix()
    t_base_eef[:3, 3] = state[:3]
    t_base_wrist = t_base_eef @ T_EEF_HAND @ T_HAND_WRIST
    files = {name: f"{command:05d}_{name}.png" for name in ("agent", "wrist")}
    lines = {name: yellow_line(trace_dir / file) for name, file in files.items()}
    result = {"command_index": command, "trace_row_index": command // 5,
              "time_s": row["time"], "logged_policy_stage": row["policy_stage"],
              "debug_phase": row["policy_debug"].get("stage"), "images": files,
              "eef_xyz_m": state[:3].tolist(),
              "finger_gap_sum_m": float(np.abs(state[6:8]).sum()),
              "line_detection": {name: None if line is None else {
                  "pixels": line["pixels"], "elongation": line["elongation"],
                  "center_px": line["center_px"].tolist(),
                  "homogeneous_line": line["line"].tolist()}
                  for name, line in lines.items()}}
    if any(line is None for line in lines.values()):
        result.update(valid_axis_estimate=False, reason="No elongated yellow component in both views")
        return result
    normal_a = RA @ KA.T @ lines["agent"]["line"]
    normal_w = t_base_wrist[:3, :3] @ KW.T @ lines["wrist"]["line"]
    normal_a /= np.linalg.norm(normal_a)
    normal_w /= np.linalg.norm(normal_w)
    axis = np.cross(normal_a, normal_w)
    intersection_sine = float(np.linalg.norm(axis))
    if intersection_sine < 1e-6:
        result.update(valid_axis_estimate=False, reason="Nearly parallel interpretation planes")
        return result
    axis /= intersection_sine
    point = np.linalg.solve(np.stack([normal_a, normal_w, axis]),
                            [normal_a @ PA, normal_w @ t_base_wrist[:3, 3], axis @ state[:3]])
    distance = float(np.linalg.norm(point - state[:3]))
    result.update(valid_axis_estimate=True,
                  unsigned_axis_base=axis.tolist(),
                  axis_to_vertical_deg=degrees_abs_dot(axis, np.array([0., 0., 1.])),
                  axis_to_eef_local_y_deg=degrees_abs_dot(axis, t_base_eef[:3, 1]),
                  reconstructed_line_to_eef_distance_m=distance,
                  interpretation_plane_intersection_sine=intersection_sine,
                  grasp_proximity_heuristic_pass=distance <= .04)
    return result


def analyze_case(root: Path, kind: str):
    case = root / kind / "case_000"
    trace = case / "trace" / "trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    transitions = []
    previous = None
    for index, row in enumerate(rows):
        if row["policy_stage"] != previous:
            transitions.append({"stage": row["policy_stage"], "time_s": row["time"],
                                "command_index": index * 5})
            previous = row["policy_stage"]
    first_time = {item["stage"]: item["time_s"] for item in transitions}
    targets = {}
    for row in rows:
        debug = row["policy_debug"]
        if "learned_target_xyz" in debug:
            targets[debug["stage"]] = {
                "learned_target_xyz_m": debug["learned_target_xyz"],
                "command_target_xyz_m": debug.get("command_target_xyz", debug["learned_target_xyz"])}
    official = json.loads((case / "official_result.json").read_text())["missions"][0]
    return {"input_case_directory": str(case), "scenario_sha256": sha256(case / "scenario.yaml"),
            "trace_sha256": sha256(trace), "trace_rows": len(rows),
            "nominal_saved_command_stride": 5,
            "median_saved_time_stride_s": float(np.median(np.diff([r["time"] for r in rows]))),
            "stage_transitions": transitions,
            "sampled_close_duration_s": first_time["lift"] - first_time["close"],
            "last_trace_time_s": rows[-1]["time"], "targets": targets,
            "official_evaluation_only": {"verdict": official["verdict"], "seconds": official["seconds"],
                                         "referee": official["referee"], "data": official["data"]},
            "frames": {phase: analyze_frame(case / "trace", rows[command // 5], command)
                       for phase, command in FRAMES[kind].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path("/home/aril/m5-runs/stability-20261122"))
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    cases = {kind: analyze_case(args.run_root, kind) for kind in FRAMES}
    base, candidate = cases["baseline"], cases["candidate-v1"]
    shared = sorted(set(base["targets"]) & set(candidate["targets"]))
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Posthoc approximate RGB/proprioception evidence, not ground-truth object tracking or a controller input",
        "script_sha256": sha256(Path(__file__)),
        "original_used_script": "original_used_analysis.py",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "opencv": cv2.__version__},
        "method": {
            "line_detector": "OpenCV HSV H18..42, S>130, V>155; dilate3x7; PCA of original pixels in largest elongated component; min15 pixels; singular ratio>=2",
            "geometry": "Each image line l gives plane normal R @ K.T @ l; intersect two planes to obtain an unsigned 3D pencil axis",
            "eef_axis_comparison": "acos(abs(dot(pencil_axis, EEF_rotation[:,1]))), in degrees",
            "vertical_comparison": "acos(abs(pencil_axis.z)), in degrees",
            "line_to_eef_distance": "Shortest distance between the reconstructed 3D line and recorded EEF origin; diagnostic only, not an accuracy bound",
            "camera_calibration_source": "Previously read public sensor configuration/camera intrinsics and static robot geometry; constants preserved from the original analysis",
            "agent_K": KA.tolist(), "wrist_K": KW.tolist(),
            "agent_camera_position_base_m": PA.tolist(),
            "agent_optical_rotation_base": RA.tolist(),
            "T_eef_hand": T_EEF_HAND.tolist(), "T_hand_wrist_optical": T_HAND_WRIST.tolist(),
            "state_layout": "EEF xyz, EEF rotation vector xyz, two finger joint positions",
            "uses_hidden_object_poses": False,
            "official_referee": "Final official metrics copied only as separate evaluation results; never used in visual axis estimation"},
        "limitations": [
            "Approximate visual evidence: no motion capture, object-pose oracle, force measurement, or calibrated statistical error bound.",
            "Uses unsigned pencil axes: cannot distinguish pointed tip from eraser, or prove tip polarity.",
            "PCA of visible yellow surface pixels can be biased by occlusion, highlights, perspective, or an incorrect connected component.",
            "Requires the saved camera calibration, static transforms, and state conventions to match the capture; rounded constants are preserved from the initial analysis.",
            "No elongated wrist component after release means no axis estimate; a line far from the EEF is not evidence of a held object.",
            "Frames are saved every five commands (about0.5 seconds); phase-end samples and transition durations are approximate. Logged policy_stage may advance while debug/action still describe the preceding stage.",
            "Relative rotation is consistent with grasp slip/contact rotation but does not isolate friction, finger force, or every physical cause.",
            "This is one matched scenario, not an estimate of overall success probability. Learned target equality excludes a target change from yaw averaging in this case.",
            "Trace time is simulation time; official seconds are wall-clock evaluation time and must not be conflated."],
        "matched_comparison": {
            "same_scenario_sha256": base["scenario_sha256"] == candidate["scenario_sha256"],
            "same_learned_xyz_targets_all_shared_stages": all(np.allclose(
                base["targets"][stage]["learned_target_xyz_m"],
                candidate["targets"][stage]["learned_target_xyz_m"], atol=1e-12, rtol=0) for stage in shared),
            "compared_stages": shared,
            "phase_axis_to_eef_y_deg": {phase: {kind: cases[kind]["frames"][phase]["axis_to_eef_local_y_deg"]
                                                for kind in cases}
                                       for phase in ["close_end", "lift_end", "upright_end", "carry_end", "insert_end"]}},
        "cases": cases}
    output = script_dir / "case0comparison.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "matched_comparison": report["matched_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
