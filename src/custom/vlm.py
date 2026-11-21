"""Real VLM inference. No room-answer fallback is used on failure."""
from __future__ import annotations

import base64
import copy
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOM_LABELS = ("classroom", "medical", "cafeteria")

PROMPT = ("Identify the room visible through the doorway. A classroom has student desks, "
          "chairs and a teaching board; a medical room has a hospital bed or medical equipment; "
          "a cafeteria has dining tables, kitchen or food equipment. "
          "Answer with exactly one word: classroom, medical, or cafeteria. "
          "If the room cannot be seen clearly answer unknown.")


def _label(text: str) -> str:
    words = set(re.findall(r"\b(?:classroom|medical|cafeteria)\b", text.lower()))
    if len(words) != 1 or "unknown" in text.lower():
        raise ValueError("VLM did not give one unambiguous room label: " + text[:160])
    return words.pop()


def assign_rooms(observations: list, min_margin: float = 1e-6) -> tuple[list[str], dict]:
    """Use the public one-room-per-class prior with actual image/VLM scores.

    No spatial room ordering is assumed. Missing scores (e.g. API backend) retain
    the original requirement that three independent labels must already be unique.
    """
    if not math.isfinite(min_margin) or min_margin < 0:
        raise ValueError('minimum assignment margin must be finite and nonnegative')
    if len(observations) != 3:
        raise ValueError("three actual room observations are required")
    if all("scores" in observation for observation in observations):
        for observation in observations:
            scores = observation["scores"]
            if (not isinstance(scores, dict) or set(scores) != set(ROOM_LABELS) or
                    any(type(value) not in (float, int) or not math.isfinite(value)
                        for value in scores.values())):
                raise ValueError("invalid VLM label likelihood scores")
        ranked = sorted(((sum(observations[i]["scores"][label] for i, label in enumerate(order)), order)
                         for order in itertools.permutations(ROOM_LABELS)), reverse=True)
        margin = ranked[0][0] - ranked[1][0]
        if margin <= max(1e-6, min_margin):
            raise ValueError("VLM scores do not distinguish a room assignment")
        return list(ranked[0][1]), {"method": "vlm_likelihood_unique_room_assignment",
            "unconstrained_labels": [o["label"] for o in observations],
            "score": ranked[0][0], "runner_up_margin": margin}
    labels = [observation["label"] for observation in observations]
    if set(labels) != set(ROOM_LABELS):
        raise ValueError("VLM labels not unique and likelihood scores unavailable")
    return labels, {"method": "independent_vlm_labels", "unconstrained_labels": labels}


def combine_views(observations: list) -> dict:
    """Average actual same-room view scores, without manufacturing a label."""
    if not observations:
        raise ValueError('at least one room view is required')
    if all('scores' in item for item in observations):
        for item in observations:
            scores = item['scores']
            if (not isinstance(scores, dict) or set(scores) != set(ROOM_LABELS)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in scores.values())):
                raise ValueError('invalid room view likelihood scores')
        scores = {label: sum(item['scores'][label] for item in observations) / len(observations)
                  for label in ROOM_LABELS}
        return {'label': max(scores, key=scores.get), 'scores': scores,
                'score_kind': 'mean_view_answer_sequence_log_likelihood',
                'views': observations}
    labels = [item.get('label') for item in observations]
    if len(set(labels)) != 1 or labels[0] not in ROOM_LABELS:
        raise ValueError('room views disagree and likelihood scores are unavailable')
    return {'label': labels[0], 'views': observations}


def classify_views(images, timeout: float = 120.) -> list[dict]:
    """One bounded CPU worker loads the local VLM once for all survey views."""
    if not images or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('images and a positive finite timeout are required')
    started = time.monotonic()
    if os.getenv('HAIC_VLM_BASE_URL'):
        results = []
        for image in images:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('VLM survey timeout')
            results.append(classify_room(image, timeout=remaining))
        return results
    with tempfile.TemporaryDirectory(prefix='haic-room-survey-') as temp:
        paths = []
        for index, image in enumerate(images):
            path = Path(temp) / f'view-{index}.png'
            image.save(path)
            paths.append(str(path))
        completed = subprocess.run([sys.executable, '-m', 'custom.vlm', '--batch', *paths],
            text=True, capture_output=True, timeout=timeout, check=False,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    if completed.returncode:
        raise RuntimeError('local VLM survey failed: ' + completed.stderr[-500:])
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if not isinstance(result, list) or len(result) != len(images):
        raise ValueError('VLM worker returned an incomplete survey')
    return result


def classify_room(image, timeout: float = 65.0) -> dict:
    """PIL RGB image -> actual VLM label plus inference provenance."""
    api = os.getenv("HAIC_VLM_BASE_URL")
    if api:
        key = os.getenv("HAIC_VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        model = os.getenv("HAIC_VLM_MODEL")
        if not key or not model:
            raise RuntimeError("vision API needs HAIC_VLM_MODEL and HAIC_VLM_API_KEY")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        body = {"model": model, "max_tokens": 30, "temperature": 0, "messages": [
            {"role": "user", "content": [{"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": url}}]}]}
        request = urllib.request.Request(api.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key,
                                                    "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = json.load(response)["choices"][0]["message"]["content"]
        return {"label": _label(text), "text": text, "model": model, "backend": "api"}
    with tempfile.TemporaryDirectory(prefix="haic-room-") as temp:
        path = Path(temp) / "room.png"
        image.save(path)
        completed = subprocess.run([sys.executable, "-m", "custom.vlm", str(path)],
            text=True, capture_output=True, timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError("local VLM failed: " + completed.stderr[-500:])
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _local_batch(image_paths: list[str]) -> list[dict]:
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    default = Path(__file__).resolve().parent / "models" / "SmolVLM-500M-Instruct"
    model_name = os.getenv("HAIC_VLM_MODEL", str(default))
    if model_name == str(default) and not (default / "config.json").is_file():
        raise RuntimeError("SmolVLM weights missing at custom/models/SmolVLM-500M-Instruct")
    torch.set_num_threads(max(1, int(os.getenv("HAIC_VLM_THREADS", "4"))))
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float32,
                                                  _attn_implementation="eager").to('cpu').eval()
    load_seconds = time.monotonic() - started
    results = []
    for image_path in image_paths:
        started = time.monotonic()
        result = _score_image(image_path, model, processor, model_name)
        result['load_seconds'] = load_seconds if not results else 0.
        result['inference_seconds'] = time.monotonic() - started
        results.append(result)
    return results


def _score_image(image_path, model, processor, model_name):
    import torch
    from PIL import Image

    messages = [{"role": "user", "content": [{"type": "image"},
                  {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[Image.open(image_path).convert("RGB")],
                       return_tensors="pt")
    scores = {}
    # Encode the image/prompt once. Candidate answers share this exact prefix.
    # This is teacher-forced conditional log likelihood; no ground truth is used.
    with torch.inference_mode():
        prefix = model(**inputs, use_cache=True)
        first_log_probs = torch.log_softmax(prefix.logits[:, -1, :].float(), dim=-1)
    for label in ROOM_LABELS:
        suffix = processor.tokenizer(label, add_special_tokens=False, return_tensors="pt")["input_ids"]
        score = first_log_probs[0, suffix[0, 0]].item()
        if suffix.shape[1] > 1:
            attention = torch.cat((inputs["attention_mask"], torch.ones_like(suffix[:, :-1])), dim=1)
            with torch.inference_mode():
                continuation = model(input_ids=suffix[:, :-1], attention_mask=attention,
                    past_key_values=copy.deepcopy(prefix.past_key_values), use_cache=False)
                log_probs = torch.log_softmax(continuation.logits.float(), dim=-1)
                score += log_probs.gather(-1, suffix[:, 1:].unsqueeze(-1)).sum().item()
        if not math.isfinite(score):
            raise RuntimeError("nonfinite VLM likelihood")
        scores[label] = score
    label = max(scores, key=scores.get)
    return {"label": label, "text": label, "text_kind": "highest_likelihood_candidate", "scores": scores,
            "score_kind": "answer_sequence_log_likelihood",
            "model": model_name, "backend": "local-cpu"}


def _local(image_path: str) -> dict:
    return _local_batch([image_path])[0]


if __name__ == "__main__":
    if sys.argv[1] == '--batch':
        print(json.dumps(_local_batch(sys.argv[2:])))
    else:
        print(json.dumps(_local(sys.argv[1])))
