from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Set ROS_LOCALHOST_ONLY before rclpy.init(); all nodes must match.
os.environ["ROS_LOCALHOST_ONLY"] = os.environ.get("HAIC_LOCALHOST_ONLY", "1")

from rclpy.node import Node  # noqa: E402

TOPICS = {
    "mission_begin": "/g1/mission_begin",
    "randomize": "/g1/randomize",
    "cmd_vel": "/cmd_vel",
    "mission_status": "/g1/mission_status",
    "mission_end": "/g1/mission_end",
}

MISSION_LABELS = {
    1: "Point-goal navigation",
    2: "Cone colour navigation",
    3: "Cliff crossing",
    4: "Semantic delivery",
    5: "Pencil pick-and-place (Panda)",
    6: "Soccer goal scoring",
}

DONE = "done"
SUCCESS = "success"
FAIL = "fail"
ERROR = "error"
NOT_IMPLEMENTED = "not_implemented"

ACK_TIMEOUT = 60.0
M4_ROOMS = ("cafeteria", "classroom", "medical")
START_POSE_WORLD = (16.0, 11.0, 0.0)
FALLS_ALLOWED = 2
ROBOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom", "robot.yaml")


def load_policies() -> dict[str, Any]:
    """custom/robot.yaml ``policies`` as {name: absolute pathlib.Path}."""
    from pathlib import Path

    import yaml

    if not os.path.isfile(ROBOT_FILE):
        return {}
    data = yaml.safe_load(Path(ROBOT_FILE).read_text()) or {}
    entries = (data.get("policies") if isinstance(data, dict) else None) or {}
    return {str(name): (Path(ROBOT_FILE).parent / str(entry)).resolve()
            for name, entry in entries.items()}


@dataclass
class MissionResult:
    mission: int
    status: str = DONE
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    judge: Optional[dict[str, Any]] = None

    @property
    def label(self) -> str:
        return MISSION_LABELS.get(self.mission, f"mission {self.mission}")

    def as_dict(self) -> dict[str, Any]:
        return {"mission": self.mission, "label": self.label, "status": self.status,
                "detail": self.detail, "seconds": round(self.seconds, 2), "data": self.data,
                "judge": self.judge}


class MissionTimeout(TimeoutError):
    pass


class MissionAborted(BaseException):
    """BaseException prevents mission-level except Exception from swallowing an abort."""


def _send_twist(publisher, twist_type, vx: float, vy: float, wz: float) -> None:
    import rclpy

    if not rclpy.ok():
        return
    msg = twist_type()
    msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
    try:
        publisher.publish(msg)
    except Exception as exc:              # noqa: BLE001
        # Ctrl-C can shut down the ROS context between the check and the call.
        if "context is invalid" not in str(exc):
            raise


class SimLink(Node):
    """The runner's own connection to the simulator (main.py, referee.py, tools/).

    Mission status, the fall count and the referee request live here and not on
    the mission nodes: a mission reaches the simulator only through the methods
    of :class:`Mission`.
    """

    # DDS discovery can lose initial commands; wait for a match and retry until acknowledged.
    MATCH_TIMEOUT_S = 5.0
    REPUBLISH_S = 3.0

    def __init__(self, node_name: str = "mission_runner") -> None:
        from geometry_msgs.msg import Twist
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Empty, Int32, String

        super().__init__(node_name)
        self._api_group = ReentrantCallbackGroup()
        self._Empty, self._Int32, self._Twist = Empty, Int32, Twist
        self._pub = {
            "mission_begin": self.create_publisher(Int32, TOPICS["mission_begin"], 10),
            "randomize": self.create_publisher(Empty, TOPICS["randomize"], 10),
            "cmd_vel": self.create_publisher(Twist, TOPICS["cmd_vel"], 10),
            "mission_end": self.create_publisher(Int32, TOPICS["mission_end"], 10),
        }
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._cv = threading.Condition()
        self._status: Optional[dict[str, Any]] = None
        self._status_seq = -1
        self.falls = 0
        self._status_sub = self.create_subscription(
            String, TOPICS["mission_status"], self._on_status, latched,
            callback_group=self._api_group)

    def log(self, text: str) -> None:
        print(text, flush=True)

    def _on_status(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self._cv:
            first = self._status is None
            self._status = payload
            self._status_seq = int(payload.get("seq", self._status_seq + 1))
            if not first and payload.get("event") == "respawned" and payload.get("counted"):
                self.falls += 1
            self._cv.notify_all()

    def last_status(self) -> Optional[dict[str, Any]]:
        with self._cv:
            return self._status

    def _wait_status(self, predicate, timeout: Optional[float], after_seq: int, what: str) -> dict:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while True:
                if self._status is not None and self._status_seq > after_seq and predicate(self._status):
                    return self._status
                if deadline is not None and time.monotonic() >= deadline:
                    raise MissionTimeout(f"no answer from the simulator: {what}")
                self._cv.wait(timeout=0.5)

    def _command(self, key: str, message, predicate, what: str,
                 timeout: Optional[float] = ACK_TIMEOUT) -> dict[str, Any]:
        publisher = self._pub[key]
        with self._cv:
            before = self._status_seq
        match_deadline = time.monotonic() + self.MATCH_TIMEOUT_S
        while publisher.get_subscription_count() == 0 and time.monotonic() < match_deadline:
            time.sleep(0.05)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            publisher.publish(message)
            slice_timeout = self.REPUBLISH_S if deadline is None else min(
                self.REPUBLISH_S, max(0.0, deadline - time.monotonic()))
            try:
                return self._wait_status(predicate, slice_timeout, before, what)
            except MissionTimeout:
                if deadline is not None and time.monotonic() >= deadline:
                    raise

    def wait_for_sim(self, timeout: Optional[float] = 600.0) -> dict[str, Any]:
        self.log(f"[mission] waiting for the simulator on {TOPICS['mission_status']} ...")
        status = self._wait_status(lambda m: True, timeout, -1, "simulator startup")
        self.log(f"[mission] simulator up (mission point {status.get('mission')}: {status.get('label')})")
        return status

    def begin_mission(self, index: int, timeout: Optional[float] = ACK_TIMEOUT) -> dict[str, Any]:
        return self._command(
            "mission_begin", self._Int32(data=int(index)),
            lambda m: m.get("event") == "selected" and m.get("mission") == index,
            f"begin mission {index}", timeout)

    def judge(self, index: int, timeout: float = 60.0) -> dict[str, Any]:
        """The simulator's verdict for mission ``index`` (referee.py)."""

        status = self._command(
            "mission_end", self._Int32(data=int(index)),
            lambda m: m.get("event") == "judged" and m.get("mission") == index,
            f"referee for mission {index}", timeout)
        return status.get("judge") or {}

    def public(self, key: str) -> Any:
        status = self.last_status()
        return None if status is None else status.get(key)

    @property
    def m1_goal(self) -> Optional[list[float]]:
        return self.public("m1_goal")

    @property
    def m2_color(self) -> Optional[str]:
        return self.public("m2_color")

    @property
    def m4_rooms(self) -> list[str]:
        return list(self.public("m4_rooms") or M4_ROOMS)

    def randomize(self, timeout: Optional[float] = ACK_TIMEOUT) -> dict[str, Any]:
        return self._command("randomize", self._Empty(), lambda m: m.get("event") == "randomized",
                             "randomize", timeout)

    def cmd_vel(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        _send_twist(self._pub["cmd_vel"], self._Twist, vx, vy, wz)

    def stop(self) -> None:
        self.cmd_vel(0.0, 0.0, 0.0)


_LINK: Optional[SimLink] = None            # the running MissionRunner's link


def _link() -> SimLink:
    if _LINK is None:
        raise RuntimeError("no mission runner: start missions with main.py or run_standalone()")
    return _LINK


class Mission(Node):
    """run() executes beside background ROS callbacks; the runner owns the node lifecycle."""

    MISSION: int = 0

    def __init__(self, node_name: Optional[str] = None) -> None:
        from geometry_msgs.msg import Twist
        from rclpy.callback_groups import ReentrantCallbackGroup

        super().__init__(node_name or (f"mission_{self.MISSION}" if self.MISSION else "mission"))
        self._api_group = ReentrantCallbackGroup()
        self._Twist = Twist
        self._cmd_vel_pub = self.create_publisher(Twist, TOPICS["cmd_vel"], 10)
        self._tf_buffer = None
        self.params: dict[str, Any] = {}
        self.policies = load_policies()

    def run(self) -> "MissionResult":
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented yet")

    def result(self, **data: Any) -> MissionResult:
        return MissionResult(self.MISSION, data=dict(data))

    def log(self, text: str) -> None:
        print(text, flush=True)

    def _tf(self):
        if self._tf_buffer is None:
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from tf2_msgs.msg import TFMessage
            from tf2_ros import Buffer

            self._tf_buffer = Buffer()

            def add(message, static: bool) -> None:
                for transform in message.transforms:
                    if static:
                        self._tf_buffer.set_transform_static(transform, "mission_api")
                    else:
                        self._tf_buffer.set_transform(transform, "mission_api")

            static_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                                    durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self._tf_subs = [
                self.create_subscription(TFMessage, "/tf", lambda m: add(m, False), 100,
                                         callback_group=self._api_group),
                self.create_subscription(TFMessage, "/tf_static", lambda m: add(m, True), static_qos,
                                         callback_group=self._api_group),
            ]
        return self._tf_buffer

    def pose(self, frame: str = "map", child: str = "pelvis",
             timeout: float = 2.0) -> Optional[tuple[float, float, float]]:
        """Return (x, y, yaw_rad) from TF in the requested frame."""

        import math

        from rclpy.duration import Duration
        from rclpy.time import Time

        try:
            t = self._tf().lookup_transform(frame, child, Time(),
                                            timeout=Duration(seconds=timeout)).transform
        except Exception:                                     # noqa: BLE001 - tf2 lookup/connectivity errors
            return None
        q = t.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return float(t.translation.x), float(t.translation.y), yaw

    def begin_mission(self, index: int, timeout: Optional[float] = ACK_TIMEOUT) -> None:
        """Reset mission objects and activate the mission without moving G1."""

        _link().begin_mission(index, timeout)

    @property
    def m1_goal(self) -> Optional[list[float]]:
        """World [x, y, yaw_deg] in testing; None in instruction mode."""

        return _link().m1_goal

    @property
    def m2_color(self) -> Optional[str]:
        """Target cone colour in testing; None in instruction mode."""

        return _link().m2_color

    @property
    def m4_rooms(self) -> list[str]:
        """The three room names to tell apart (sorted names, no positions)."""

        return _link().m4_rooms

    def randomize(self, timeout: Optional[float] = ACK_TIMEOUT) -> None:
        """Practice only: a new random scenario."""

        _link().randomize(timeout)

    def cmd_vel(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        """Body-frame forward/left velocity in m/s and counter-clockwise yaw rate in rad/s."""

        _send_twist(self._cmd_vel_pub, self._Twist, vx, vy, wz)

    def stop(self) -> None:
        self.cmd_vel(0.0, 0.0, 0.0)


class MissionRunner:
    def __init__(self, node_name: str = "mission_runner", *, log: Callable[[str], None] = print) -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.executor = MultiThreadedExecutor()
        global _LINK

        self.api = SimLink(node_name)
        self.api.log = log
        _LINK = self.api
        self.executor.add_node(self.api)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, name="mission-spin", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set() and self._rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                return
            except Exception as exc:                          # noqa: BLE001
                # A mission node destroyed while its subscription was still in this
                # spin's wait list: skip it, or every later mission gets no callbacks.
                if type(exc).__name__ != "InvalidHandle":
                    raise

    def start(self, mission_class: Callable[[], Mission]) -> Mission:
        mission = mission_class()
        if not isinstance(mission, Mission):
            raise TypeError(f"{mission_class!r} must create a mission_api.Mission, "
                            f"got {type(mission).__name__}")
        self.executor.add_node(mission)
        return mission

    def finish(self, mission: Mission) -> None:
        try:
            self.executor.remove_node(mission)
        except Exception:                                      # noqa: BLE001 - never added
            pass
        time.sleep(0.3)                    # let the spin that still lists this node finish
        mission.destroy_node()

    def close(self) -> None:
        global _LINK

        _LINK = None
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.executor.remove_node(self.api)
            self.executor.shutdown(timeout_sec=1.0)
            self.api.destroy_node()
        finally:
            if self._owns_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()

    def __enter__(self) -> "MissionRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.api.stop()
        self.close()


class FallWatch:
    def __init__(self, link: SimLink, allowed: int = FALLS_ALLOWED) -> None:
        self.mission, self.allowed = link, int(allowed)     # falls are counted over the whole run
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="fall-watch", daemon=True)
        self._previous = None

    def _watch(self) -> None:
        import signal

        while not self._stop.wait(0.2):
            if self.mission.falls > self.allowed:
                self.tripped = True
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def __enter__(self) -> "FallWatch":
        import signal

        def abort(_signum, _frame) -> None:
            raise MissionAborted(f"respawned at the start {self.mission.falls} times in this run "
                                 f"({self.allowed} allowed): the mission is over")

        self._previous = signal.signal(signal.SIGUSR1, abort)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        import signal

        self._stop.set()
        self._thread.join(timeout=1.0)
        signal.signal(signal.SIGUSR1, self._previous or signal.SIG_DFL)


def check_result(mission: int, result: Any) -> MissionResult:
    if isinstance(result, MissionResult):
        result.mission = mission
        return result
    if result is None:
        return MissionResult(mission)
    return MissionResult(mission, ERROR,
                         f"run() returned {type(result).__name__}; return self.result()")


def run_standalone(mission_class: Callable[[], Mission]) -> int:
    from referee import Referee

    referee = Referee(None)
    with MissionRunner("mission_standalone") as runner:
        runner.api.wait_for_sim()
        mission = runner.start(mission_class)
        index = mission.MISSION
        referee.begin(index, rooms=runner.api.m4_rooms)
        try:
            with FallWatch(runner.api):
                result = check_result(index, mission.run())
        except MissionAborted as exc:
            result = MissionResult(index, FAIL, str(exc))
        except KeyboardInterrupt:
            mission.stop()
            print("interrupted")
            return 130
        mission.stop()
        record = referee.finish(runner.api, result)
        runner.finish(mission)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str))
    return 0 if record["success"] else 1
