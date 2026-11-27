#!/usr/bin/env python3
"""Start the val_861 mission simulator and run your missions."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ["ROS_LOCALHOST_ONLY"] = os.environ.get("HAIC_LOCALHOST_ONLY", "1")

from mission_api import (  # noqa: E402
    ERROR,
    FAIL,
    MISSION_LABELS,
    FallWatch,
    Mission,
    MissionAborted,
    MissionResult,
    MissionRunner,
    check_result,
)
from referee import Referee, default_results_path  # noqa: E402


CONFIG_FILE = SRC_ROOT / "main_config.yaml"
CONFIG_DEFAULTS = {
    "missions": {"run": [1, 2, 3, 4, 5, 6], "wait_for_space": True, "step": False,
                 "falls_allowed": 2},
    "scenario": {"file": "scenarios/default.yaml"},
    "referee": {"goal_radius_m": 0.5, "cone_box_m": 0.5, "cone_box_offset_m": 0.75,
                "m4_room": "classroom"},
    "randomize": {"m1_goal_region": [14.0, 17.0, 1.0, 20.0], "m1_goal_yaw_deg": 90.0,
                  "ball_region": [7.25, 11.25, 5.5, 8.5]},
    "instruction": {"expect": None, "web_port": 8501, "open_browser": True},
    "simulator": {"view": True, "start_pose": [16.0, 11.0, 0.0], "shadow_size": 0,
                  "ball_friction": [0.72, 0.02, 0.01], "extra_args": []},
}
ROBOT_FILE = SRC_ROOT / "custom" / "robot.yaml"


def robot_args() -> list[str]:
    """Simulator arguments from custom/robot.yaml (sensors, walking); checks the policy files."""
    if not ROBOT_FILE.is_file():
        return []
    import yaml

    data = yaml.safe_load(ROBOT_FILE.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"[main] {ROBOT_FILE}: expected a mapping")
    unknown = sorted(set(data) - {"sensors", "walking", "policies"})
    if unknown:
        raise SystemExit(f"[main] {ROBOT_FILE}: unknown section {unknown[0]!r} "
                         "(sensors | walking | policies)")
    extra: list[str] = []
    if data.get("sensors"):
        extra += ["--sensors", str(ROBOT_FILE)]
    walking = data.get("walking") or {}
    if not isinstance(walking, dict):
        raise SystemExit(f"[main] {ROBOT_FILE}: walking: expected a mapping")
    policy = walking.get("policy")
    if policy:
        path = (ROBOT_FILE.parent / str(policy)).resolve()
        if not path.is_file():
            raise SystemExit(f"[main] {ROBOT_FILE}: walking.policy {policy!r} not found at {path}")
        extra += ["--policy", str(path)]
    caps = walking.get("max_vel") or {}
    if caps:
        try:
            extra += ["--max-vel", *(str(float(caps[k])) for k in ("vx", "vy", "wz"))]
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"[main] {ROBOT_FILE}: walking.max_vel needs vx, vy, wz (numbers)") from None
    policies = data.get("policies") or {}
    if not isinstance(policies, dict):
        raise SystemExit(f"[main] {ROBOT_FILE}: policies: expected  name: path  entries")
    for name, entry in policies.items():
        path = (ROBOT_FILE.parent / str(entry)).resolve()
        if not path.exists():
            raise SystemExit(f"[main] {ROBOT_FILE}: policies.{name} {entry!r} not found at {path}")
    return extra


def load_config(path: Path) -> dict:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    return {section: {**defaults, **(data.get(section) or {})}
            for section, defaults in CONFIG_DEFAULTS.items()}


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=CONFIG_FILE)
    config_file = pre.parse_known_args(argv)[0].config
    try:
        config = load_config(config_file)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[main] cannot read {config_file}: {exc}")
    mis, ins, sim = config["missions"], config["instruction"], config["simulator"]

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog=f"defaults from {config_file}")
    ap.add_argument("--config", type=Path, default=config_file,
                    help="run configuration (default: src/main_config.yaml)")
    ap.add_argument("--missions", type=int, nargs="+", default=list(mis["run"]),
                    metavar="N", help="test: only these missions (config: missions.run)")
    ap.add_argument("--llm-web", action="store_true",
                    help="instruction mode: type the instruction on a web page; your planner "
                         "(custom/planner/planner.py) turns it into the missions to run")
    ap.add_argument("--sim-only", action="store_true",
                    help="start the simulator (and custom/bringup.launch.py) and keep it running "
                         "(no missions)")
    ap.add_argument("--wait", action=argparse.BooleanOptionalAction, default=bool(mis["wait_for_space"]),
                    help="wait for SPACE in this terminal before the first mission "
                         "(config: missions.wait_for_space; never waits without a TTY)")
    ap.add_argument("--step", action=argparse.BooleanOptionalAction, default=bool(mis["step"]),
                    help="also wait for SPACE before each mission (n skips it, q quits)")
    ap.add_argument("--scenario", type=Path, default=None, metavar="FILE",
                    help="test: another scenario file (config: scenario.file)")
    ap.add_argument("--ground-truth", action="store_true",
                    help="DEBUG ONLY: also publish the simulator's ground-truth /odom")
    args = ap.parse_args(argv)
    bad = [n for n in args.missions + list(ins["expect"] or []) if n not in MISSION_LABELS]
    if bad:
        ap.error(f"unknown mission(s) {bad}; valid: 1-6")
    scenario = args.scenario or config["scenario"]["file"]
    args.scenario = None if not scenario else (SRC_ROOT / scenario).resolve()
    if args.scenario is not None and not args.scenario.is_file():
        ap.error(f"scenario file not found: {args.scenario} (config: scenario.file)")
    args.judge = config["referee"]
    args.m4_room = str(config["referee"]["m4_room"])
    args.randomize = config["randomize"]
    args.expect = ins["expect"]
    args.web_port = int(ins["web_port"])
    args.open_browser = bool(ins["open_browser"])
    args.view = bool(sim["view"])
    args.falls_allowed = int(mis["falls_allowed"])
    args.start_pose = [float(v) for v in sim["start_pose"]]
    if len(args.start_pose) not in (2, 3):
        ap.error("simulator.start_pose takes [x, y, yaw_deg]")
    args.shadow_size = int(sim["shadow_size"])
    args.ball_friction = [float(v) for v in sim["ball_friction"]]
    if len(args.ball_friction) != 3:
        ap.error("simulator.ball_friction takes [sliding, torsional, rolling]")
    args.sim_arg = [str(a) for a in sim["extra_args"] or []]
    return args


def start_simulator(args: argparse.Namespace) -> subprocess.Popen:
    command = [sys.executable, "-m", "ros2_bridge.run_g1_ros", "--view" if args.view else "--no-view",
               "--spawn", " ".join(f"{v:g}" for v in args.start_pose),
               "--m4-room", args.m4_room,
               "--judge-goal-radius", str(args.judge["goal_radius_m"]),
               "--judge-cone-box", str(args.judge["cone_box_m"]),
               "--judge-cone-offset", str(args.judge["cone_box_offset_m"]),
               "--ball-friction", *(str(v) for v in args.ball_friction),
               "--m1-goal-region", *(str(v) for v in args.randomize["m1_goal_region"]),
               "--m1-goal-yaw", str(args.randomize["m1_goal_yaw_deg"]),
               "--ball-region", *(str(v) for v in args.randomize["ball_region"])]
    if args.scenario is not None:
        command += ["--scenario", str(args.scenario)]
    if args.llm_web:
        command.append("--hide-targets")
    command += robot_args()
    if args.shadow_size > 0:
        command += ["--shadow-size", str(args.shadow_size)]
    if args.ground_truth:
        command.append("--ground-truth")
    command += args.sim_arg
    print("[main] starting the simulator: " + " ".join(command[1:]), flush=True)
    return subprocess.Popen(command, cwd=SRC_ROOT, start_new_session=True,
                            preexec_fn=_die_with_parent)


URDF = SRC_ROOT / "environment" / "robots" / "description" / "g1_description" / "g1_sensors.urdf"


def start_robot_state_publisher() -> subprocess.Popen:
    import yaml

    text = URDF.read_text().replace("package://g1_navigation/meshes/",
                                    f"file://{URDF.parent / 'meshes'}/")
    params = tempfile.NamedTemporaryFile("w", prefix="haic_rsp_", suffix=".yaml", delete=False)
    yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {
        "use_sim_time": True, "robot_description": text}}}, params)
    params.close()
    from ament_index_python.packages import get_package_prefix

    # Run the executable directly so PR_SET_PDEATHSIG reaches the child.
    executable = Path(get_package_prefix("robot_state_publisher"), "lib",
                      "robot_state_publisher", "robot_state_publisher")
    command = [str(executable), "--ros-args", "--params-file", params.name]
    print("[main] starting robot_state_publisher (G1 URDF -> /tf)", flush=True)
    return subprocess.Popen(command, cwd=SRC_ROOT, start_new_session=True,
                            preexec_fn=_die_with_parent,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# Children have separate sessions; forward signals to their process groups.
_HURRY = threading.Event()
try:
    _PRCTL = ctypes.CDLL(None, use_errno=True).prctl
except (OSError, AttributeError):
    _PRCTL = None


def _die_with_parent() -> None:
    if _PRCTL is not None:
        _PRCTL(1, int(signal.SIGINT))


def _detach_output() -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)


def _interrupt(signum, _frame) -> None:
    if signum == signal.SIGHUP:
        _detach_output()
    raise KeyboardInterrupt


def _hurry(signum, _frame) -> None:
    if signum == signal.SIGHUP:
        _detach_output()
    elif not _HURRY.is_set():
        print("[main] stopping now (SIGKILL)", flush=True)
    _HURRY.set()


def stop_all(procs: list[tuple[subprocess.Popen | None, str]]) -> None:
    alive = [(proc, name) for proc, name in procs if proc is not None and proc.poll() is None]
    for _, name in alive:
        print(f"[main] stopping the {name}", flush=True)
    for sig, grace in ((signal.SIGINT, 20.0), (signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        if _HURRY.is_set():
            sig = signal.SIGKILL
        for proc, _ in alive:
            try:
                if os.getpgid(proc.pid) == proc.pid:
                    os.killpg(proc.pid, sig)
                else:
                    proc.send_signal(sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while alive and time.monotonic() < deadline:
            alive = [(proc, name) for proc, name in alive if proc.poll() is None]
            if _HURRY.is_set() and sig != signal.SIGKILL:
                break
            time.sleep(0.1)
        if not alive:
            return


class TerminalKeys:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._saved = None

    def __enter__(self) -> "TerminalKeys":
        if self.enabled:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._saved is not None:
            import termios

            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except termios.error:
                pass
            self._saved = None

    def read(self, timeout: float) -> Optional[str]:
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None

    def wait_for(self, accepted: str, alive=lambda: True) -> Optional[str]:
        if not self.enabled:
            return None
        while alive():
            key = self.read(0.25)
            if key is not None and key in accepted:
                return key
        return None


def prompt_to_start(keys: TerminalKeys, what: str, *, skippable: bool, alive=lambda: True) -> str:
    if not keys.enabled:
        return "start"
    options = "SPACE to start, " + ("n to skip, " if skippable else "") + "q to quit"
    print(f"\n>>> {what}: press {options}.", flush=True)
    key = keys.wait_for(" qQ" + ("nN" if skippable else ""), alive)
    if key is None or key == " ":
        return "start"
    return "quit" if key in "qQ" else "skip"


BRINGUP_LAUNCH = SRC_ROOT / "custom" / "bringup.launch.py"


def start_bringup() -> subprocess.Popen | None:
    if not BRINGUP_LAUNCH.is_file():
        return None
    print("[main] launching custom/bringup.launch.py (your nodes)", flush=True)
    return subprocess.Popen(["ros2", "launch", str(BRINGUP_LAUNCH)], cwd=SRC_ROOT,
                            start_new_session=True, preexec_fn=_die_with_parent,
                            stdin=subprocess.DEVNULL)


def make_plan(instruction: str) -> list[tuple[int, dict]]:
    from custom.planner.planner import plan

    steps = []
    for item in plan(instruction) or []:
        params: dict = {}
        if isinstance(item, (tuple, list)):
            if len(item) != 2 or not isinstance(item[1], dict):
                raise ValueError(f"plan() items must be N or (N, {{...}}), got {item!r}")
            item, params = item
        try:
            index = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"plan() must return mission numbers, got {item!r}") from None
        if index not in MISSION_LABELS:
            raise ValueError(f"the plan names mission {index}; valid: 1-6")
        steps.append((index, dict(params)))
    return steps


def mission_class(index: int):
    module = importlib.import_module(f"custom.mission_{index}.mission_{index}")
    cls = getattr(module, f"Mission{index}", None)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, Mission)):
        raise TypeError(f"{module.__file__} must define class Mission{index}(mission_api.Mission)")
    return cls


def run_step(runner: MissionRunner, referee: Referee, index: int, *, params: dict | None = None,
             plan: int | None = None, number: int = 0, of: int = 0,
             falls_allowed: int = 2) -> dict:
    print(f"\n[main] ===== mission {index}: {MISSION_LABELS[index]}"
          + (f" (step {number}/{of}{', ' + str(params) if params else ''})" if of else "")
          + " =====", flush=True)
    referee.begin(index, plan=plan, step=number, of=of, rooms=runner.api.m4_rooms)
    mission = None
    aborted = None
    try:
        if runner.api.falls > falls_allowed:
            raise MissionAborted(f"respawned at the start {runner.api.falls} times in this run "
                                 f"({falls_allowed} allowed): mission {index} is not run")
        mission = runner.start(mission_class(index))
        mission.params = dict(params or {})
        with FallWatch(runner.api, falls_allowed):
            result = check_result(index, mission.run())
    except MissionAborted as exc:
        print(f"[main] {exc}", flush=True)
        aborted = str(exc)
        result = MissionResult(index, FAIL, aborted)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:                              # noqa: BLE001 - incl. SystemExit
        import traceback

        traceback.print_exc()
        result = MissionResult(index, ERROR, f"{type(exc).__name__}: {exc}")
    if mission is not None:
        mission.stop()
    runner.api.stop()
    record = referee.finish(runner.api, result, failed=aborted)
    if mission is not None:
        runner.finish(mission)
    return record


def run_instruction(runner: MissionRunner, referee: Referee, args: argparse.Namespace,
                    instruction: str, progress=lambda **update: None) -> None:
    progress(state="planning")
    clock = time.monotonic()
    try:
        plan_steps = make_plan(instruction)
        missions = [index for index, _ in plan_steps]
    except Exception as exc:                                  # noqa: BLE001 - reported, nothing runs
        import traceback

        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
        referee.add_plan(instruction, [], [], args.expect, error=error)
        print(f"[main] the planner failed: {error}", flush=True)
        progress(state="error", error=error)
        return
    steps = [f"mission {index}" + (f" {params}" if params else "") for index, params in plan_steps]
    print(f"[main] plan for {instruction!r}:", flush=True)
    for number, (index, params) in enumerate(plan_steps, 1):
        print(f"[main]   {number}. mission {index}  ({MISSION_LABELS[index]})"
              + (f"  {params}" if params else ""), flush=True)
    plan = referee.add_plan(instruction, steps, missions, args.expect)
    records: list[dict] = []
    progress(state="running", plan=steps, records=records)
    for number, (index, params) in enumerate(plan_steps, 1):
        records.append(run_step(runner, referee, index, params=params, plan=plan, number=number,
                                of=len(plan_steps), falls_allowed=args.falls_allowed))
        progress(records=records)
    referee.plan_done(plan, time.monotonic() - clock)
    passed = sum(r["success"] is True for r in records)
    progress(state="done", total={"passed": passed, "missions": len(records),
                                  "seconds": round(time.monotonic() - clock, 1)})


WEB_PAGE = SRC_ROOT / "web" / "instruction_page.py"


def start_web_page(port: int, open_browser: bool = True) -> subprocess.Popen:
    url = f"http://localhost:{port}"
    print(f"[main] starting the instruction page on {url}", flush=True)
    page = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(WEB_PAGE), "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=SRC_ROOT, start_new_session=True, preexec_fn=_die_with_parent,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if open_browser and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        import socket
        import webbrowser

        deadline = time.monotonic() + 20.0
        while page.poll() is None and time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.25)
        if not webbrowser.open_new_tab(url):
            print(f"[main] no browser found; open {url} yourself", flush=True)
    return page


def serve_web_page(runner: MissionRunner, referee: Referee, args: argparse.Namespace, alive) -> None:
    from web.instruction_page import PROGRESS, REQUEST, read_json, write_json

    for stale in (REQUEST, PROGRESS):
        stale.unlink(missing_ok=True)
    print(f"[main] type the instruction on http://localhost:{args.web_port}; Ctrl-C here stops "
          "everything", flush=True)
    done = None
    while alive():
        request = read_json(REQUEST)
        if not request or request.get("id") == done:
            time.sleep(0.5)
            continue
        done = request["id"]
        state = {"id": done, "instruction": request.get("instruction", ""), "plan": [],
                 "records": [], "results_file": str(referee.path)}

        def progress(**update) -> None:
            records = update.pop("records", None)
            if records is not None:
                update["records"] = [{k: r.get(k) for k in ("step", "mission", "label", "verdict",
                                                            "seconds", "checks", "rooms", "reported")}
                                     for r in records]
            state.update(update)
            write_json(PROGRESS, state)

        print(f"\n[main] instruction from the page: {state['instruction']!r}", flush=True)
        run_instruction(runner, referee, args, state["instruction"], progress)


def run_missions(runner: MissionRunner, referee: Referee, args: argparse.Namespace,
                 keys: TerminalKeys, alive) -> bool:
    if args.wait and not args.step:
        if prompt_to_start(keys, "ready", skippable=False, alive=alive) == "quit":
            return False
    for index in args.missions:
        if args.step:
            choice = prompt_to_start(keys, f"mission {index} ({MISSION_LABELS[index]})",
                                     skippable=True, alive=alive)
            if choice == "quit":
                return False
            if choice == "skip":
                referee.skip(index, MISSION_LABELS[index], "skipped (n)")
                continue
        if not alive():
            return False
        run_step(runner, referee, index, falls_allowed=args.falls_allowed)
    return True


def main(argv=None) -> int:
    args = parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)
    sim = rsp = web = None
    bringup = None
    referee = None
    complete = False
    try:
        sim = start_simulator(args)
        rsp = start_robot_state_publisher()
        with MissionRunner("mission_runner") as runner, TerminalKeys() as keys:
            runner.api.wait_for_sim(timeout=900.0)
            bringup = start_bringup()
            alive = lambda: sim.poll() is None                       # noqa: E731
            if args.sim_only:
                print("[main] simulator is up; run custom/mission_N/mission_N.py from another "
                      "terminal, Ctrl-C here to stop", flush=True)
                while alive():
                    time.sleep(1.0)
                return 0
            mode = "instruction (web page)" if args.llm_web else "auto"
            referee = Referee(default_results_path(SRC_ROOT), run={
                "package": SRC_ROOT.parent.name, "mode": mode,
                "config": str(args.config),
                "scenario": None if args.scenario is None else str(args.scenario),
                "referee": {**args.judge, "m1_goal": runner.api.m1_goal,
                            "m2_color": runner.api.m2_color}})
            if args.llm_web:
                web = start_web_page(args.web_port, args.open_browser)
                serve_web_page(runner, referee, args, lambda: alive() and web.poll() is None)
                complete = True
            else:
                complete = run_missions(runner, referee, args, keys, alive)
                if not complete:
                    print("[main] quit (q)", flush=True)
    except KeyboardInterrupt:
        print("\n[main] interrupted", flush=True)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _hurry)
        stop_all([(sim, "simulator"), (rsp, "robot_state_publisher"), (web, "instruction page"),
                  (bringup, "custom/bringup.launch.py")])
    if referee is None:
        return 0
    total = referee.close(complete=complete)
    return 0 if total["missions"] and total["passed"] == total["missions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
