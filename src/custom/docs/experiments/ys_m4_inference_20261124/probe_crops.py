"""Evaluation only: evaluate the existing fixed retry crop as the primary view."""
import argparse
import json
from pathlib import Path
import tempfile
import time
from PIL import Image
from custom.vlm import _local_batch, assign_rooms, combine_views

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('cases', nargs='+')
a = p.parse_args()
results = []
for case in a.cases:
    root = Path(case)
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for i in range(3):
            with Image.open(root/'camera'/f'view-{i}.png') as image:
                width, height = image.size
                box = (width//10, 0, 9*width//10, 4*height//5)
                path = Path(directory)/f'{i}.png'
                image.crop(box).save(path)
                paths.append(str(path))
        started = time.monotonic()
        observations = _local_batch(paths)
        elapsed = time.monotonic()-started
    record = json.loads((root/'record.json').read_text())
    expected = record['case']['room_order']
    baseline = next(m for m in record['official_result']['missions'] if m['mission']==4)['data']['vlm_observations']
    trials = {}
    for name, items in [('crop-only', observations), ('mean-full-crop',
            [combine_views([a,b]) for a,b in zip(baseline,observations)])]:
        try:
            assignment, details = assign_rooms(items, min_margin=.25)
        except ValueError as error:
            assignment, details = [], {'error':str(error)}
        trials[name] = {'observations':items, 'assignment':assignment, 'details':details,
            'correct':sum(o['label']==e for o,e in zip(items,expected))}
    results.append({'case_dir':case, 'expected':expected, 'crop_box':list(box),
        'baseline_correct':sum(o['label']==e for o,e in zip(baseline,expected)),
        'seconds':elapsed, 'trials':trials})
    Path(a.output).write_text(json.dumps(results, indent=2))
    print(root.name, {k:(v['correct'],v['assignment']) for k,v in trials.items()}, flush=True)
