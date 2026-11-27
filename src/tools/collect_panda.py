"""Record Panda demonstrations for mission 5 with the TeleDex iPhone app.

    python tools/collect_panda.py                      # episodes -> custom/mission_5/dataset
    python tools/collect_panda.py --dataset-dir DIR --port 8889

The simulator starts with G1 at the Panda desk and the arm following the phone.
Every demonstration is one file, episode_000000.h5, 000001, ... (numbering continues):

    observation/image        (T, 256, 256, 3) uint8   desk camera  = /panda/agentview/image_raw
    observation/wrist_image  (T, 256, 256, 3) uint8   wrist camera = /panda/wrist_camera/image_raw
    observation/state        (T, 8)  gripper position, rotation vector (arm base frame), 2 fingers
    actions                  (T, 7)  normalized gripper step in the arm base frame, grip (-1 open)
    attrs: task_instruction, fps (10), success (1 saved as success, 0 failed), frames, ...

Phone: Button A = success, save, next task; Button B = discard, next task.
MuJoCo window: G = gripper, S = save as success, F = save as failed, Q = quit.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import main as runner_main  # noqa: E402  (src/main.py: config and the robot's simulator arguments)

DESK_POSE = "22.25 19.55 180"       # G1 beside the Panda desk, as in mission 5
INSTRUCTION = "Pick up the wooden pencil and place it inside the gray square pencil holder."
# import name -> pip name (setup/requirements.txt)
MODULES = {"mujoco": "mujoco", "h5py": "h5py", "teledex": "teledex", "qrcode": "qrcode",
           "cv2": "opencv-python"}


def log(message: str) -> None:
    print(f"[collect] {message}", flush=True)


def lan_addresses() -> list[str]:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [a for a in out.split() if "." in a and not a.startswith(("127.", "172.17."))]


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def episode_count(folder: Path) -> int:
    return len(list(folder.glob("episode_*.h5")))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", type=Path, default=SRC_ROOT / "custom" / "mission_5" / "dataset",
                    help="where the episode files go (default: custom/mission_5/dataset)")
    ap.add_argument("--task-instruction", default=INSTRUCTION,
                    help="the sentence stored with every episode")
    ap.add_argument("--port", type=int, default=8888, help="TCP port the phone connects to")
    ap.add_argument("--position-scale", type=float, default=1.6,
                    help="hand motion -> gripper motion")
    ap.add_argument("--yaw-scale", type=float, default=1.0, help="phone yaw gain; -1 inverts")
    ap.add_argument("--pitch-scale", type=float, default=1.0,
                    help="phone tilt gain; -1 inverts, 0 disables")
    ap.add_argument("--no-preview", action="store_true",
                    help="no window with the two recorded cameras")
    ap.add_argument("--no-view", action="store_true", help="no MuJoCo window")
    ap.add_argument("--config", type=Path, default=runner_main.CONFIG_FILE,
                    help="main_config.yaml (simulator.shadow_size is used)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    missing = [pip for module, pip in MODULES.items() if importlib.util.find_spec(module) is None]
    if missing:
        log(f"missing Python packages: {' '.join(missing)}")
        log("install them: bash setup/install_conda_env.sh   (README: 설치 ③)")
        return 1
    if importlib.util.find_spec("rclpy") is None:
        log("rclpy not found -- conda activate your haic_<team> environment first")
        return 1
    view = not options.no_view
    if view and not os.environ.get("DISPLAY"):
        log("no DISPLAY: teleoperation needs the MuJoCo window (or pass --no-view to run blind)")
        return 1
    if not port_free(options.port):
        log(f"TCP port {options.port} is already in use (an earlier collect_panda.py, Jupyter, ...).")
        log("stop that program, or pick another port: --port 8889")
        return 1

    config = runner_main.load_config(options.config)
    shadow = int((config.get("simulator") or {}).get("shadow_size") or 0)
    dataset = options.dataset_dir.expanduser().resolve()
    command = [
        sys.executable, "-m", "ros2_bridge.run_g1_ros", "--view" if view else "--no-view",
        "--spawn", DESK_POSE,
        "--panda-mode", "collect",
        "--panda-preview" if view and not options.no_preview else "--no-panda-preview",
        "--lidar-hz", "0",
        "--dataset-dir", str(dataset), "--task-instruction", options.task_instruction,
        "--teledex-port", str(options.port),
        "--panda-position-scale", str(options.position_scale),
        "--panda-yaw-scale", str(options.yaw_scale),
        "--panda-pitch-scale", str(options.pitch_scale),
        *runner_main.robot_args(),
    ]
    if shadow > 0:
        command += ["--shadow-size", str(shadow)]

    before = episode_count(dataset)
    log(f"episodes are saved to {dataset} ({before} there now)")
    log(f"task: {options.task_instruction!r}")
    addresses = lan_addresses()
    if addresses:
        log("TeleDex app (same Wi-Fi/LAN as this computer): scan the QR below, or type "
            + "  or  ".join(f"{a}:{options.port}" for a in addresses))
    log("phone: A = success+save, B = discard;  window: G gripper, S success, F failed, Q quit")

    # Same terminal as the simulator: it prints the QR code and the per-episode messages.
    simulator = subprocess.Popen(command, cwd=SRC_ROOT)

    def forward(_signum, _frame) -> None:      # terminal closed / kill: stop the simulator too
        if simulator.poll() is None:
            simulator.send_signal(signal.SIGINT)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGHUP, forward)
    signal.signal(signal.SIGINT, lambda *_: None)   # Ctrl-C already reaches the simulator
    try:
        code = simulator.wait()
    finally:
        if simulator.poll() is None:
            simulator.kill()
    added = episode_count(dataset) - before
    log(f"simulator stopped (exit {code}); {added} new episode(s), {before + added} in {dataset}")
    return 0 if code in (0, -signal.SIGINT, 130) else code


if __name__ == "__main__":
    raise SystemExit(main())
