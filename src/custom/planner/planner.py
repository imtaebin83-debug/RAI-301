"""LLM instruction planner: compatible API or local Qwen CPU inference.

Rules are available only when HAIC_ALLOW_RULE_PLANNER=1 is explicitly set for
development. HAIC_REQUIRE_LLM=1 always disables that development path.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

SYSTEM = '''Extract an ordered robot mission plan from the user's instruction.
Return only valid JSON with a steps array. Every step has integer mission and object params.
Missions: 1 navigate to requested world coordinate; 2 find requested green/blue/orange cone;
3 cross the cliff/gap course; 4 identify rooms and deliver pencil to classroom;
5 Panda insert pencil into pencil holder; 6 score soccer goal.
Mission 1 params must include goal:[x,y]. Mission 2 params must include color.
Other missions have empty params. Preserve requested order. Do not invent coordinates,
colors or missions. Missing required arguments: return {"steps":[]}.
Example instruction: Go to the blue cone, then score a goal.
Example answer: {"steps":[{"mission":2,"params":{"color":"blue"}},{"mission":6,"params":{}}]}
Example instruction: Navigate to (14.2,7.7), then cross the cliff.
Example answer: {"steps":[{"mission":1,"params":{"goal":[14.2,7.7]}},{"mission":3,"params":{}}]}
Example instruction: Deliver the pencil to the classroom, then put a pencil into the holder.
Example answer: {"steps":[{"mission":4,"params":{}},{"mission":5,"params":{}}]}'''


def validate(payload: dict) -> list:
    if (not isinstance(payload, dict) or set(payload) != {"steps"} or
            not isinstance(payload.get("steps"), list)):
        raise ValueError("planner response must contain steps list")
    result = []
    for step in payload["steps"]:
        if not isinstance(step, dict):
            raise ValueError("each step must be an object")
        if "mission" not in step or not set(step).issubset({"mission", "params"}):
            raise ValueError("step contains missing or unknown fields")
        n, params = step.get("mission"), step.get("params", {})
        if type(n) is not int or n not in range(1, 7) or not isinstance(params, dict):
            raise ValueError("invalid mission or params")
        if n == 1:
            if set(params) != {"goal"}:
                raise ValueError("mission 1 accepts only goal")
            goal = params.get("goal")
            if (not isinstance(goal, list) or len(goal) != 2 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in goal)):
                raise ValueError("mission 1 needs two finite goal coordinates")
            params = {"goal": [float(x) for x in goal]}
        elif n == 2:
            if set(params) != {"color"}:
                raise ValueError("mission 2 accepts only color")
            if params.get("color") not in ("green", "blue", "orange"):
                raise ValueError("mission 2 needs green, blue, or orange")
            params = {"color": params["color"]}
        elif params:
            raise ValueError("unexpected mission parameters")
        result.append((n, params) if params else n)
    if not result:
        raise ValueError("no executable missions found")
    return result


def parse_plan(text: str) -> list:
    """Reject duplicate JSON keys instead of silently dropping earlier missions."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate planner JSON key: " + key)
            result[key] = value
        return result

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    payload, end = decoder.raw_decode(text)
    remainder = text[end:].strip()
    # Qwen sometimes emits a dangling quote after an otherwise complete object.
    # Remove only quoting punctuation, never another task, JSON value or prose.
    if remainder:
        if any(character not in "\"'`" for character in remainder):
            raise ValueError("extra content after planner JSON; refusing ambiguous plan")
        print("[planner] removed trailing quoting after complete LLM JSON", file=sys.stderr)
    return validate(payload)


def _fallback(instruction: str) -> list:
    text, found = instruction.lower(), []
    explicit = []
    for group in re.finditer(r"(?:\bmissions?\s*|미션\s*)([1-6](?:\s*[,→>]\s*[1-6])*)", text):
        explicit.extend((group.start(1) + digit.start(), int(digit.group()))
                        for digit in re.finditer(r"[1-6]", group[1]))
    numbers = [n for _, n in explicit]
    if len(numbers) != len(set(numbers)):
        raise ValueError("repeated mission numbers require configured LLM planning")
    coordinate = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", text)
    if coordinate and re.search(r"(?:\bmissions?\s*|미션\s*)$", text[:coordinate.start()]):
        raise ValueError("a bracketed mission list is not a navigation coordinate")
    if coordinate:
        explicit_one = next((position for position, n in explicit if n == 1), coordinate.start())
        found.append((explicit_one, {"mission": 1, "params": {
            "goal": [float(coordinate[1]), float(coordinate[2])]}}))
    elif 1 in numbers:
        raise ValueError("mission 1 needs coordinates explicitly stated in instruction")
    patterns = {
        2: r"\bcone\b|콘", 3: r"\bcliff\b|\bgap(?:s)?\b|절벽|틈|낭떠러지",
        4: r"\bdeliver\b|\brooms?\b|배달|교실|방을|방 배치",
        5: r"\b(?:insert|holder|panda|pick.and.place)\b|연필꽂이|꽂|판다",
        6: r"\b(?:soccer|football|goal|score)\b|축구|골대|득점"}
    if coordinate and not re.search(r"soccer|football|score|축구|골대|득점", text):
        patterns[6] = r"(?!)"
    colors = {"green": "green", "초록": "green", "녹색": "green",
              "blue": "blue", "파란": "blue", "파랑": "blue", "청색": "blue",
              "orange": "orange", "주황": "orange"}
    for n, pattern in patterns.items():
        match = re.search(pattern, text)
        explicit_position = next((position for position, number in explicit if number == n), None)
        if not match and explicit_position is None:
            continue
        position = explicit_position if explicit_position is not None else match.start()
        params = {}
        if n == 2:
            matches = [(m.start(), colors[m.group()]) for m in re.finditer(
                "|".join(re.escape(k) for k in colors), text)]
            if not matches:
                raise ValueError("cone color missing from instruction")
            params["color"] = min(matches, key=lambda c: abs(c[0] - position))[1]
        found.append((position, {"mission": n, "params": params}))
    return validate({"steps": [step for _, step in sorted(found, key=lambda p: p[0])]})


def plan(instruction: str) -> list:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction is empty")
    key = os.getenv("HAIC_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("HAIC_LLM_MODEL")
    if key and model:
        endpoint = os.getenv("HAIC_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = {"model": model, "temperature": 0, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]}
        request = urllib.request.Request(endpoint + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key,
                                                     "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            answer = json.load(response)["choices"][0]["message"]["content"]
        return parse_plan(answer)
    if (os.getenv("HAIC_ALLOW_RULE_PLANNER") == "1" and
            os.getenv("HAIC_REQUIRE_LLM") != "1"):
        print("[planner] DEVELOPMENT FALLBACK: explicitly enabled rules", file=sys.stderr)
        return _fallback(instruction)
    helper = Path(__file__).resolve().parent / "local_model.py"
    print("[planner] local Qwen CPU language-model inference", file=sys.stderr)
    completed = subprocess.run([sys.executable, str(helper)],
        input=json.dumps({"system": SYSTEM, "instruction": instruction}),
        text=True, capture_output=True, timeout=45, check=False)
    if completed.returncode:
        raise RuntimeError("local planner LLM failed: " + completed.stderr[-1000:])
    return parse_plan(completed.stdout)


if __name__ == "__main__":
    for number, step in enumerate(plan(" ".join(sys.argv[1:])), 1):
        print(f"{number}. {step}")
