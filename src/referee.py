from __future__ import annotations

import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

M4_REQUIRE_DELIVERY = True
ROOM_ALIASES = {
    "class room": "classroom", "medical room": "medical", "infirmary": "medical",
    "clinic": "medical", "nurse's office": "medical", "nurse office": "medical",
    "health room": "medical", "canteen": "cafeteria", "dining room": "cafeteria",
    "lunchroom": "cafeteria", "lunch room": "cafeteria",
}
SLOT_NAMES = ("north", "middle", "south")


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def room_key(name: Any) -> str:
    text = re.sub(r"\(.*?\)", "", str(name).lower())
    text = re.sub(r"[\s_-]+", " ", text).strip()
    return ROOM_ALIASES.get(text, text)


def _point(value: Any) -> Optional[tuple[float, float]]:
    try:
        if isinstance(value, dict):
            value = value.get("xy") or value.get("pos") or (value["x"], value["y"])
        return float(value[0]), float(value[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def parse_room_answer(answer: Any) -> dict[str, Optional[tuple[float, float]]]:
    if isinstance(answer, dict):
        items = list(answer.items())
    elif isinstance(answer, (list, tuple)):
        items = [(a.get("room") or a.get("name"), a) for a in answer if isinstance(a, dict)]
    else:
        return {}
    return {room_key(name): _point(where) for name, where in items if name}


def check_rooms(boxes: dict[str, Sequence[float]], answer: Any) -> dict[str, Any]:
    order = sorted(boxes, key=lambda r: -(boxes[r][2] + boxes[r][3]))
    slot = {room: SLOT_NAMES[i] if i < len(SLOT_NAMES) else f"slot {i}" for i, room in enumerate(order)}

    def room_at(point) -> Optional[str]:
        if point is None:
            return None
        return next((r for r, b in boxes.items() if b[0] <= point[0] <= b[1] and b[2] <= point[1] <= b[3]),
                    None)

    if isinstance(answer, (list, tuple)) and answer and all(isinstance(a, str) for a in answer):
        given = {}
        for name, room in zip(answer, order):
            b = boxes[room]
            given[room_key(name)] = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2)
    else:
        given = parse_room_answer(answer)
    rows = []
    for room in order:
        point = given.get(room)
        found = room_at(point)
        rows.append({"room": room, "slot": slot[room], "answer_xy": point, "answer_in": found,
                     "ok": found == room})
    correct = sum(row["ok"] for row in rows)
    named_in = {room: [name for name, point in given.items() if room_at(point) == room] for room in order}
    stray = sorted(name for name, point in given.items() if name not in boxes or room_at(point) is None)
    truth_line = " | ".join(f"{slot[r]} {r}" for r in order)
    answer_line = " | ".join(f"{slot[r]} {'/'.join(named_in[r]) or '-'}" for r in order)
    if stray:
        answer_line += "   (not in any room / unknown: " + ", ".join(stray) + ")"
    return {"ok": bool(given) and correct == len(order), "correct": correct, "total": len(order),
            "truth": truth_line, "answer": answer_line if given else "no answer (self.result(rooms=[...]))",
            "rows": rows, "given": {k: v for k, v in given.items()}}


RESPAWN_PENALTY_S = 30.0      # added to the total time for every counted respawn of the run


class Referee:
    def __init__(self, path: Optional[Path], *, run: Optional[dict[str, Any]] = None,
                 log: Callable[[str], None] = print) -> None:
        self.path = None if path is None else Path(path)
        self.log = log
        self.run: dict[str, Any] = {"started_at": _now(), **(run or {}), "plans": []}
        self.records: list[dict[str, Any]] = []
        self._run_t0 = time.monotonic()
        self._t0: Optional[float] = None
        self._started_at: Optional[str] = None
        self._step: Optional[tuple[int, int, int]] = None
        self._closed = False
        self.respawns = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write(complete=False)
            self.log(f"[referee] results file: {self.path}")

    def add_plan(self, instruction: str, steps: Sequence[str], missions: Sequence[int],
                 expected: Optional[Sequence[int]] = None, **extra: Any) -> int:
        plan = {"instruction": instruction, "steps": list(steps), "missions": list(missions),
                "planned_at": _now(), **extra}
        if expected:
            plan["expected"] = list(expected)
            plan["matches_expected"] = list(missions) == list(expected)
            self.log(f"[referee] plan {list(missions)} vs expected {list(expected)}: "
                     + ("same" if plan["matches_expected"] else "DIFFERENT"))
        self.run["plans"].append(plan)
        self._write(complete=False)
        return len(self.run["plans"]) - 1

    def plan_done(self, plan: int, seconds: float) -> None:
        self.run["plans"][plan]["seconds"] = round(seconds, 2)
        self.log(f"[referee] instruction -> last verdict: {seconds:.1f} s (planning + missions + judging)")
        self._write(complete=False)

    def begin(self, mission: int, *, plan: Optional[int] = None, step: int = 0, of: int = 0,
              rooms: Optional[Sequence[str]] = None) -> None:
        self._t0 = time.monotonic()
        self._started_at = _now()
        self._step = (plan, step, of)
        if mission == 4 and rooms:
            self.log(f"[referee] mission 4 rooms to identify: {', '.join(rooms)} "
                     "(answer: result.data['rooms'] = {room: [x, y] world})")

    @staticmethod
    def _ask_simulator(link, index: int, settle: float = 1.0, timeout: float = 60.0) -> dict[str, Any]:
        """The simulator's verdict for mission ``index`` (asked through the runner's SimLink)."""

        link.stop()
        time.sleep(settle)
        return link.judge(index, timeout)

    def finish(self, node, result, *, judge: bool = True,
               failed: Optional[str] = None) -> dict[str, Any]:
        """``failed``: the mission was ended by the rules (respawn limit) -- FAIL, the simulator is not asked."""

        seconds = 0.0 if self._t0 is None else time.monotonic() - self._t0
        ended_at = _now()
        self.respawns = int(getattr(node, "falls", self.respawns))
        verdict: dict[str, Any] = {}
        note = None
        if failed:
            verdict = {"mission": int(result.mission), "success": False, "summary": failed}
        elif judge:
            try:
                verdict = self._ask_simulator(node, result.mission)
            except Exception as exc:                          # noqa: BLE001 - no verdict, recorded
                note = f"no verdict from the simulator: {exc}"
        result.judge = verdict
        result.seconds = seconds
        record = self._record(result, verdict, seconds, ended_at, note)
        self.records.append(record)
        self._report(record)
        self._write(complete=False)
        self._t0 = None
        return record

    def skip(self, mission: int, label: str, reason: str) -> dict[str, Any]:
        record = {"mission": mission, "label": label, "verdict": "N/A", "success": None,
                  "seconds": 0.0, "started_at": _now(), "ended_at": _now(),
                  "checks": [], "reported": {"status": "skipped", "detail": reason}}
        self.records.append(record)
        self.log(f"[referee] mission {mission} ({label}) not run: {reason}")
        self._write(complete=False)
        return record

    def _record(self, result, verdict: dict, seconds: float, ended_at: str,
                note: Optional[str]) -> dict[str, Any]:
        plan, step, of = self._step or (None, 0, 0)
        success = verdict.get("success")
        checks = []
        if verdict:
            checks.append({"name": "delivery" if result.mission == 4 else "check",
                           "ok": success, "summary": verdict.get("summary", "")})
        rooms = None
        if result.mission == 4 and verdict.get("rooms"):
            rooms = check_rooms(verdict["rooms"], (result.data or {}).get("rooms"))
            checks.insert(0, {"name": "rooms", "ok": rooms["ok"],
                              "summary": f"{rooms['correct']}/{rooms['total']} rooms right"})
            if success is not None:
                success = rooms["ok"] and (bool(success) or not M4_REQUIRE_DELIVERY)
        record: dict[str, Any] = {
            "mission": result.mission, "label": result.label,
            "verdict": {True: "PASS", False: "FAIL"}.get(success, "N/A"), "success": success,
            "seconds": round(seconds, 2), "started_at": self._started_at, "ended_at": ended_at,
            "checks": checks,
            "reported": {"status": result.status, "detail": result.detail},
            "referee": verdict, "data": result.data,
        }
        if plan is not None:
            record.update({"plan": plan, "step": step, "of": of})
        if rooms is not None:
            record["rooms"] = rooms
        if note:
            record["note"] = note
        return record

    def _report(self, r: dict[str, Any]) -> None:
        where = f" (step {r['step']}/{r['of']})" if r.get("of") else ""
        self.log(f"[referee] ===== mission {r['mission']}{where} {r['label']}: "
                 f"{r['verdict']} in {r['seconds']:.1f} s =====")
        rooms = r.get("rooms")
        if rooms is not None:
            mark = "PASS" if rooms["ok"] else "FAIL"
            self.log(f"[referee]   rooms     truth  : {rooms['truth']}")
            self.log(f"[referee]             answer : {rooms['answer']}")
            self.log(f"[referee]             {rooms['correct']}/{rooms['total']} rooms right  [{mark}]")
        for check in r["checks"]:
            if check["name"] == "rooms":
                continue
            ok = {True: "PASS", False: "FAIL"}.get(check["ok"], "N/A")
            self.log(f"[referee]   {check['name']:<9} {check['summary']}  [{ok}]")
        if r.get("note"):
            self.log(f"[referee]   note      {r['note']}")
        self.log(f"[referee]   reported  {r['reported']['status']}: {r['reported']['detail']}")

    def totals(self) -> dict[str, Any]:
        judged = [r for r in self.records if r["success"] is not None]
        return {
            "missions": len(self.records),
            "passed": sum(r["success"] is True for r in self.records),
            "failed": sum(r["success"] is False for r in self.records),
            "not_judged": len(self.records) - len(judged),
            "mission_seconds": round(sum(r["seconds"] for r in self.records), 2),
            "respawns": self.respawns,
            "penalty_seconds": round(self.respawns * RESPAWN_PENALTY_S, 2),
            "total_seconds": round(sum(r["seconds"] for r in self.records)
                                   + self.respawns * RESPAWN_PENALTY_S, 2),
            "run_seconds": round(time.monotonic() - self._run_t0, 2),
            "by_mission": [f"{r['mission']} {r['verdict']} {r['seconds']:.1f}s" for r in self.records],
        }

    def _write(self, *, complete: bool) -> None:
        if self.path is None:
            return
        total = {**self.totals(), "complete": complete, "updated_at": _now()}
        text = json.dumps({"run": self.run, "missions": self.records, "total": total},
                          indent=2, ensure_ascii=False, default=str)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, self.path)

    def close(self, *, complete: bool = True) -> dict[str, Any]:
        if self._closed:
            return self.totals()
        self._closed = True
        total = self.totals()
        self.run["ended_at"] = _now()
        self._write(complete=complete)
        if not self.records:
            return total
        bar = "=" * 30
        self.log(f"\n{bar} Mission results (referee, ground truth) {bar}")
        self.log(f"  {'step':<5} #  {'mission':<30} {'result':<6} {'time':>8}  check")
        for r in self.records:
            step = f"{r['step']}/{r['of']}" if r.get("of") else "-"
            check = "; ".join(f"{c['name']}: {c['summary']}" for c in r["checks"]) or r["reported"]["detail"]
            self.log(f"  {step:<5} {r['mission']}  {r['label']:<30} {r['verdict']:<6} "
                     f"{r['seconds']:7.1f}s  {check}")
        for plan in self.run["plans"]:
            if "expected" in plan:
                self.log(f"  plan for {plan['instruction']!r}: {plan['missions']} "
                         f"(expected {plan['expected']}: "
                         + ("same)" if plan["matches_expected"] else "DIFFERENT)"))
            if "seconds" in plan:
                self.log(f"  instruction -> last verdict: {plan['seconds']:.1f} s")
        self.log(f"  {total['passed']}/{total['missions']} passed, missions {total['mission_seconds']:.1f} s, "
                 f"run {total['run_seconds']:.1f} s"
                 + ("" if complete else " (run interrupted)"))
        self.log(f"  respawns {total['respawns']} x {RESPAWN_PENALTY_S:.0f} s = +{total['penalty_seconds']:.0f} s penalty"
                 f" -> total time {total['total_seconds']:.1f} s (missions + penalty)")
        if self.path is not None:
            self.log(f"  results file: {self.path}")
        self.log("=" * (len(bar) * 2 + 42))
        return total


def default_results_path(root: Path) -> Path:
    return root / "results" / datetime.datetime.now().strftime("run_%Y%m%d-%H%M%S.json")
