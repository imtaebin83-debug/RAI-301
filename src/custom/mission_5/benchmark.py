"""Paired, seeded Mission5 evaluation with the unmodified official main.

Evaluation-only: scenario coordinates are passed only to the simulator. The
Mission5 controller never imports this module or reads the generated scenarios.
This program owns /tmp/rai-gpu-simulator.lock; do not wrap it in flock again.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def source_root(value: Path) -> Path:
    root = value.expanduser().resolve()
    if (root / "src" / "main.py").is_file():
        root = root / "src"
    if not (root / "main.py").is_file() or not (root / "custom/mission_5").is_dir():
        raise ValueError("--root must name a competition repository or its src directory")
    return root


def git_info(root: Path) -> dict:
    def output(*args):
        result = subprocess.run(["git", "-C", str(root), *args], text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"head": output("rev-parse", "HEAD"),
            "branch": output("branch", "--show-current"),
            "status": output("status", "--porcelain", "--untracked-files=no")}


def fingerprints(root: Path, checkpoint: Path) -> dict:
    files = sorted((root / "custom").rglob("*.py"))
    files += [root / name for name in ("main.py", "mission_api.py", "referee.py",
                                      "custom/sensors.yaml", "custom/walking.yaml",
                                      "custom/requirements.txt",
                                      "ros2_bridge/tools/val861_scene.py")]
    return {"files": {str(p.relative_to(root)): sha256(p) for p in files
                      if p.is_file() and "__pycache__" not in p.parts},
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint),
                           "bytes": checkpoint.stat().st_size}}


def official_sampler(root: Path):
    # Imports only the official layout/sampling helpers, never a simulator.
    sys.path.insert(0, str(root))
    from ros2_bridge.tools import val861_scene
    from ros2_bridge.run_g1_ros import _arguments
    config=yaml.safe_load((root/'main_config.yaml').read_text())
    defaults = _arguments(list(config.get('simulator',{}).get('extra_args') or []))
    if not defaults.panda or defaults.panda_mode!='ros':
        raise ValueError('Benchmark requires the ordinary Panda ROS mode')
    return val861_scene, defaults


def make_scenario(root: Path, seed: int, index: int, sampler, defaults) -> tuple[dict, dict]:
    # Direct addressing makes --case-index reproduce a case without prior draws.
    case_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
    layout = sampler.EDITOR.load_layout(defaults.scene.resolve())
    from types import SimpleNamespace
    task = sampler.randomize_panda_task(layout, SimpleNamespace(
        task_seed=case_seed, no_task_randomization=False,
        panda_x=defaults.panda_x, panda_y=defaults.panda_y))
    objects = {row["id"]: row for row in layout["objects"]}
    pencil, cup = objects[sampler.TASK_PENCIL_ID], objects[sampler.TASK_CUP_ID]
    scenario = yaml.safe_load((root / "scenarios/default.yaml").read_text())
    # Isolate grasp stability in the standard room arrangement. Relocation is a
    # separate experimental factor, not silently mixed with object placement.
    from ros2_bridge.tools.scenario import DEFAULT_ROOM_ORDER
    scenario["room_order"] = list(DEFAULT_ROOM_ORDER)
    scenario["panda_task"] = {
        "pencil_xy": [float(v) for v in task["pencil_xy"]],
        "pencil_yaw_deg": float(pencil["yaw"]),
        "cup_xy": [float(v) for v in task["cup_xy"]],
        "cup_yaw_deg": float(cup["yaw"]),
    }
    metadata = {"index": index, "suite_seed": seed, "case_seed": case_seed,
                "distribution": "official val861_scene.randomize_panda_task",
                "numpy_version": np.__version__,
                "room_order": scenario["room_order"],
                "panda_base_xy": [float(defaults.panda_x), float(defaults.panda_y)],
                "object_min_separation_m": float(task["object_min_separation"])}
    return scenario, metadata


@contextlib.contextmanager
def gpu_reservation():
    import fcntl
    with open("/tmp/rai-gpu-simulator.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            print("GPU/simulator reservation busy (benchmark owns its own flock)",file=sys.stderr)
            raise SystemExit(75) from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def stop_owned_process(process: subprocess.Popen) -> None:
    for sig, seconds in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("Owned simulator launcher did not exit after cleanup")


def trace_summary(directory: Path) -> dict:
    rows, invalid = [], 0
    trace = directory / "trace.jsonl"
    if trace.is_file():
        for line in trace.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                invalid += 1
    stages = []
    for row in rows:
        name = row.get("policy_stage")
        if not stages or stages[-1]["stage"] != name:
            stages.append({"stage": name, "first_sim_time": row.get("time"), "samples": 0})
        stages[-1]["samples"] += 1
        stages[-1]["last_sim_time"] = row.get("time")
    return {"trace_rows": len(rows), "invalid_rows": invalid, "stages": stages,
            "last_stage": rows[-1].get("policy_stage") if rows else None,
            "last_debug": rows[-1].get("policy_debug") if rows else None,
            "agent_images": len(list(directory.glob("*_agent.png"))),
            "wrist_images": len(list(directory.glob("*_wrist.png")))}


def official_result(logfile: Path, root: Path, before: set[Path]) -> tuple[Path | None, dict | None]:
    text = logfile.read_text(errors="replace")
    paths = re.findall(r"\[referee\] results file: ([^\r\n]+)", text)
    candidates = [Path(p.strip()).resolve() for p in paths]
    if not candidates:
        candidates = list(set((root / "results").glob("*.json")) - before)
    candidates = [p for p in candidates if p.is_file()
                  and p.is_relative_to((root / "results").resolve())]
    if len(candidates) != 1:
        return None, None
    path = candidates[0]
    try:
        return path, json.loads(path.read_text())
    except (OSError, ValueError):
        return path, None


def run_case(args, root: Path, checkpoint: Path, case: int, scenario: dict, metadata: dict) -> dict:
    directory = args.out / f"case_{case:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    trace = directory / "trace"
    trace.mkdir()
    scenario_path = directory / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=True))
    config = yaml.safe_load((root / "main_config.yaml").read_text())
    config["simulator"]["view"] = False
    config["instruction"]["open_browser"] = False
    extra = list(config["simulator"].get("extra_args") or [])
    config["simulator"]["extra_args"] = [*extra, "--task-seed", str(metadata["case_seed"])]
    config_path = directory / "headless_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    artifact_before = fingerprints(root, checkpoint)
    before = set((root / "results").glob("*.json"))
    logfile = directory / "main.log"
    command = [sys.executable, "-u", str(root / "main.py"), "--missions", "5",
               "--no-wait", "--config", str(config_path), "--scenario", str(scenario_path)]
    environment = {**os.environ, "HAIC_ROOT": str(root.parent),
                   "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "MUJOCO_GL": "egl", "PYTHONDONTWRITEBYTECODE": "1",
                   "M5_CHECKPOINT": str(checkpoint), "M5_TRACE_DIR": str(trace)}
    timed_out, interrupted, process, error = False, False, None, None
    started = time.monotonic()
    try:
        with logfile.open("w") as stream:
            process = subprocess.Popen(command, cwd=root, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                error = "benchmark wall timeout"
            except KeyboardInterrupt:
                interrupted = True
                error = "benchmark interrupted"
            finally:
                stop_owned_process(process)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    path, official = official_result(logfile, root, before)
    if official is not None:
        write_json(directory / "official_result.json", official)
    records = (official or {}).get("missions", [])
    record = next((row for row in records if row.get("mission") == 5), None)
    detail = record.get("data", {}) if record else {}
    trace_data = trace_summary(trace)
    artifact_after = fingerprints(root, checkpoint)
    report = {**metadata, "root": str(root), "checkpoint": str(checkpoint),
              "scenario_sha256": sha256(scenario_path), "command": command,
              "returncode": None if process is None else process.returncode,
              "wall_seconds": elapsed, "timed_out": timed_out, "interrupted": interrupted,
              "error": error, "official_source": str(path) if path else None,
              "success": record.get("success") if record else None,
              "official_record": record, "trace": trace_data,
              "failure_stage": None if record and record.get("success") is True else trace_data["last_stage"],
              "controller_error": detail.get("error"),
              "artifacts_before": artifact_before,
              "artifacts_changed_during_case": artifact_before != artifact_after}
    if report["artifacts_changed_during_case"]:
        report["artifacts_after"] = artifact_after
    write_json(directory / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--case-index", type=int, help="Run one zero-based index from the same seeded suite")
    parser.add_argument("--start-case", "--startcase", dest="start_case", type=int, default=0,
                        help="Run [start-case,cases) into a new output directory")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after first failed/unjudged case")
    parser.add_argument("--checkpoint", type=Path, help="Default: variant root's mission_5/keyposes.json")
    parser.add_argument("--timeout", type=float, default=240.)
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32 or not 1 <= args.cases <= 100 or not 10 <= args.timeout <= 900:
        parser.error("seed must be uint32; cases 1..100; timeout 10..900 seconds")
    if args.case_index is not None and not 0 <= args.case_index < args.cases:
        parser.error("--case-index must be in [0,cases)")
    if not 0 <= args.start_case < args.cases:
        parser.error("--start-case must be in [0,cases)")
    if args.case_index is not None and args.start_case:
        parser.error("Use either --case-index or --start-case")
    root = source_root(args.root)
    checkpoint = (args.checkpoint or root / "custom/mission_5/keyposes.json").expanduser().resolve()
    if not checkpoint.is_file():
        parser.error("Missing checkpoint: " + str(checkpoint))
    args.out = args.out.expanduser().resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be new or empty; use a new directory for repeated trials")
    args.out.mkdir(parents=True, exist_ok=True)
    sampler, defaults = official_sampler(root)
    indices = [args.case_index] if args.case_index is not None else list(range(args.start_case,args.cases))
    metadata = {"seed": args.seed, "cases": args.cases, "indices": indices,
                "root": str(root), "git": git_info(root), "checkpoint": str(checkpoint),
                "sampler_sha256": sha256(Path(sampler.__file__)),
                "note": "Evaluation only; scenario/official truth are never policy inputs"}
    results = []
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    with gpu_reservation():
        write_json(args.out / "run_manifest.json", metadata)
        for index in indices:
            scenario, case_metadata = make_scenario(root, args.seed, index, sampler, defaults)
            result = run_case(args, root, checkpoint, index, scenario, case_metadata)
            results.append(result)
            summary = {"seed": args.seed, "requested_cases": len(indices), "attempted": len(results),
                       "passed": sum(row["success"] is True for row in results),
                       "failed": sum(row["success"] is False for row in results),
                       "not_judged": sum(row["success"] is None for row in results),
                       "cases": [{key: row.get(key) for key in ("index", "case_seed", "success",
                                  "wall_seconds", "failure_stage", "controller_error", "error",
                                  "scenario_sha256", "artifacts_changed_during_case")} for row in results]}
            write_json(args.out / "summary.json", summary)
            print(json.dumps(summary["cases"][-1]), flush=True)
            if result["interrupted"]:
                raise KeyboardInterrupt
            if args.stop_on_fail and result["success"] is not True:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
