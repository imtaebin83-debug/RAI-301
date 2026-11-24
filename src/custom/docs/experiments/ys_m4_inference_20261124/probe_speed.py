"""CPU-only development benchmark; never imported by mission control."""
import argparse
import json
from pathlib import Path
import resource
import time

p = argparse.ArgumentParser()
p.add_argument('--attention', choices=['eager', 'sdpa'], required=True)
p.add_argument('--last-logit', action='store_true')
p.add_argument('--quantize', action='store_true')
p.add_argument('--text-only', action='store_true')
p.add_argument('--output', required=True)
p.add_argument('images', nargs='+')
a = p.parse_args()
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from custom.vlm import _score_image
from custom.mission_4.room_scoring import DESCRIPTION_PROMPT, DESCRIPTIONS, score_room_image
torch.set_num_threads(4)
started = time.monotonic()
model_path = '/home/aril/RAI-301/src/custom/models/SmolVLM-500M-Instruct'
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(model_path, local_files_only=True,
    torch_dtype=torch.float32, _attn_implementation=a.attention).to('cpu').eval()
if a.quantize:
    target = model.model.text_model if a.text_only else model
    torch.ao.quantization.quantize_dynamic(target, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
original = model.forward
if a.last_logit:
    def forward(*args, **kwargs):
        if 'pixel_values' in kwargs:
            kwargs['logits_to_keep'] = 1
        return original(*args, **kwargs)
    model.forward = forward
loaded = time.monotonic()-started
cache = {}
results = []
for path in a.images:
    before = time.monotonic()
    item = score_room_image(path, lambda path: _score_image(path, model, processor,
        model_path, prompt_text=DESCRIPTION_PROMPT, answers=DESCRIPTIONS), cache)
    results.append({'image': path, 'seconds': time.monotonic()-before, **item})
    print(path, results[-1]['seconds'], item['label'], flush=True)
Path(a.output).write_text(json.dumps({'attention':a.attention, 'last_logit':a.last_logit,
    'quantize': a.quantize,
    'text_only': a.text_only,
    'load_seconds':loaded, 'seconds':time.monotonic()-started,
    'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'results':results}, indent=2))
