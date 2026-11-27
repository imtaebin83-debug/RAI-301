"""Evaluation only: benchmark production inference against saved camera surveys."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    from custom import vlm
    from custom.mission_4 import room_scoring
    import torch
    import transformers
    records = []
    for directory in args.case_dir:
        images = [directory/'camera'/f'view-{i}.png' for i in range(3)]
        started = time.monotonic()
        # Production entry point: each survey has its own model/calibration
        # cache, matching the runtime worker rather than sharing across cases.
        observations = vlm._local_batch([str(path) for path in images])
        seconds = time.monotonic()-started
        saved = json.loads((directory/'record.json').read_text())
        expected = saved['case']['room_order']
        baseline = next(m for m in saved['official_result']['missions']
                        if m['mission']==4)['data']['vlm_observations']
        labels, details = vlm.assign_rooms(observations, min_margin=.25)
        records.append({'case_dir': str(directory), 'expected': expected,
            'image_sha256': [hashlib.sha256(path.read_bytes()).hexdigest() for path in images],
            'observations': observations, 'assignment': labels, 'assignment_details': details,
            'independent_correct': sum(o['label']==e for o,e in zip(observations,expected)),
            'same_independent_labels': all(a['label']==b['label'] for a,b in zip(observations,baseline)),
            'max_score_difference': max(abs(a['scores'][k]-b['scores'][k])
                for a,b in zip(observations,baseline) for k in vlm.ROOM_LABELS),
            'seconds': seconds, 'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        report = {'torch': torch.__version__, 'transformers': transformers.__version__,
            'source_sha256': {str(Path(module.__file__).name):
                hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (vlm, room_scoring)}, 'records': records}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(directory.name, records[-1]['independent_correct'], labels,
              records[-1]['max_score_difference'], round(seconds, 2), flush=True)


if __name__ == '__main__':
    main()
