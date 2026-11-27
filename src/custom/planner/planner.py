"""LLM planner: team-hosted local-model API or local Qwen CPU inference.

Rules are available only when HAIC_ALLOW_RULE_PLANNER=1 is explicitly set for
development. HAIC_REQUIRE_LLM=1 always disables that development path.
"""
from __future__ import annotations

import json
import ipaddress
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
from urllib.parse import urlsplit
from custom.planner.worker import generate_local

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

SYSTEM += '''
Understand both English and Korean. Output every requested action in order,
including the final action in a long sentence. Color values must be English:
초록/초록색/녹색 = green; 파랑/파란색/청색 = blue; 주황/주황색 = orange.
절벽/틈 건너기 = mission 3; 방 알아내기/교실에 연필 배달 = mission 4;
연필꽂이에 연필 꽂기 = mission 5; 축구/득점 = mission 6.
Example instruction: 초록색 콘으로 이동하고 절벽을 건넌 후 교실에 연필을 배달해.
Example answer: {"steps":[{"mission":2,"params":{"color":"green"}},{"mission":3,"params":{}},{"mission":4,"params":{}}]}
Example instruction: (16, 4.5)로 가고 주황색 콘을 찾아가.
Example answer: {"steps":[{"mission":1,"params":{"goal":[16,4.5]}},{"mission":2,"params":{"color":"orange"}}]}
Example instruction: 콘을 찾아가.
Example answer: {"steps":[]}
Example instruction: Make coffee.
Example answer: {"steps":[]}
'''


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


def _team_endpoint(endpoint: str) -> str:
    """v2 permits local models on a team server, not external model APIs.

    Require an explicit localhost/private IP; no DNS guessing or default
    commercial endpoint. Operators must use their own authorized model server.
    """
    parsed = urlsplit(endpoint)
    allowed = parsed.hostname == 'localhost'
    if not allowed:
        try:
            address = ipaddress.ip_address(parsed.hostname or '')
            networks = ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16',
                        '100.64.0.0/10', '127.0.0.0/8', '::1/128', 'fc00::/7')
            allowed = any(address in ipaddress.ip_network(net) for net in networks)
        except ValueError:
            pass
    if (not allowed or parsed.scheme not in ('http', 'https') or
            parsed.username is not None or parsed.password is not None or
            parsed.query or parsed.fragment):
        raise ValueError('v2 requires HAIC_LLM_BASE_URL on an explicit localhost/private team server')
    return endpoint.rstrip('/')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('team model endpoint redirects are disabled')


def plan(instruction: str) -> list:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction is empty")
    if len(instruction) > 4000:
        raise ValueError("instruction exceeds 4000 characters")
    key = os.getenv("HAIC_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("HAIC_LLM_MODEL")
    base = os.getenv('HAIC_LLM_BASE_URL')
    if model and (key or base):
        endpoint = _team_endpoint(base or '')
        body = {"model": model, "temperature": 0, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]}
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = urllib.request.Request(endpoint + "/chat/completions",
            data=json.dumps(body).encode(), headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=30) as response:
            answer = json.load(response)["choices"][0]["message"]["content"]
        return parse_plan(answer)
    if (os.getenv("HAIC_ALLOW_RULE_PLANNER") == "1" and
            os.getenv("HAIC_REQUIRE_LLM") != "1"):
        print("[planner] DEVELOPMENT FALLBACK: explicitly enabled rules", file=sys.stderr)
        return _fallback(instruction)
    helper = Path(__file__).resolve().parent / "local_model.py"
    print("[planner] local Qwen CPU language-model inference", file=sys.stderr)
    if os.getenv('HAIC_PLANNER_ONESHOT') != '1':
        return parse_plan(generate_local({'system': SYSTEM, 'instruction': instruction}, timeout=45))
    completed = subprocess.run([sys.executable, str(helper)],
        input=json.dumps({"system": SYSTEM, "instruction": instruction}),
        text=True, capture_output=True, timeout=45, check=False)
    if completed.returncode:
        raise RuntimeError("local planner LLM failed: " + completed.stderr[-1000:])
    return parse_plan(completed.stdout)


if __name__ == "__main__":
    for number, step in enumerate(plan(" ".join(sys.argv[1:])), 1):
        print(f"{number}. {step}")
