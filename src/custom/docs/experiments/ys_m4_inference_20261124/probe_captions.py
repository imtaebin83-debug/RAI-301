"""Evaluation-only fixed candidate; expected answers are read after inference."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('cases', nargs='+')
a = p.parse_args()
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from custom.vlm import _score_image, assign_rooms
from custom.mission_4 import room_scoring as scoring

# Frozen before examining the remaining surveys: visible objects in development
# cafeteria frames include food on tables and a coffee machine.
scoring.DESCRIPTIONS = {**scoring.DESCRIPTIONS,
    'cafeteria': 'The room contains dining tables, food and a coffee machine.'}
torch.set_num_threads(4)
model_path = '/home/aril/RAI-301/src/custom/models/SmolVLM-500M-Instruct'
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(model_path, local_files_only=True,
    torch_dtype=torch.float32, _attn_implementation='sdpa').to('cpu').eval()
original = model.forward
def forward(*args, **kwargs):
    if 'pixel_values' in kwargs:
        kwargs['logits_to_keep'] = 1
    return original(*args, **kwargs)
model.forward = forward
cache, results = {}, []
for case in a.cases:
    root = Path(case)
    observations = []
    started = time.monotonic()
    for i in range(3):
        path = root/'camera'/f'view-{i}.png'
        item = scoring.score_room_image(path, lambda path: _score_image(path, model, processor,
            model_path, prompt_text=scoring.DESCRIPTION_PROMPT, answers=scoring.DESCRIPTIONS), cache)
        observations.append({**item, 'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    record = json.loads((root/'record.json').read_text())
    expected = record['case']['room_order']
    mission = next(m for m in record['official_result']['missions'] if m['mission']==4)
    baseline = mission['data']['vlm_observations']
    try:
        assignment, details = assign_rooms(observations, min_margin=.25)
    except ValueError as error:
        assignment, details = [], {'error':str(error)}
    results.append({'case_dir':case, 'expected':expected, 'observations':observations,
        'assignment':assignment, 'assignment_details':details,
        'candidate_correct':sum(o['label']==e for o,e in zip(observations,expected)),
        'baseline_correct':sum(o['label']==e for o,e in zip(baseline,expected)),
        'seconds':time.monotonic()-started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    Path(a.output).write_text(json.dumps(results, indent=2))
    print(root.name, results[-1]['candidate_correct'], assignment, flush=True)
