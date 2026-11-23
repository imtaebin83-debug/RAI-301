"""Evaluation only: compare VLM variants on saved surveys, never robot control."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, action='append', required=True)
    parser.add_argument('--variant', choices=['formatted', 'neutral'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from custom.vlm import ROOM_LABELS, _score_image, assign_rooms

    torch.set_num_threads(4)
    model_path = Path(__file__).resolve().parents[1] / 'models/SmolVLM-500M-Instruct'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.float32, _attn_implementation='eager').to('cpu').eval()
    bias = None
    if args.variant == 'neutral':
        with tempfile.TemporaryDirectory() as directory:
            neutral = Path(directory) / 'neutral.png'
            Image.new('RGB', (640, 480), (127, 127, 127)).save(neutral)
            bias = _score_image(neutral, model, processor, str(model_path))['scores']
    else:
        tokenizer = processor.tokenizer
        class FormattedTokenizer:
            def __getattr__(self, name):
                return getattr(tokenizer, name)
            def __call__(self, text, *a, **kw):
                if isinstance(text, str) and text in ROOM_LABELS and kw.get('add_special_tokens') is False:
                    text = ' ' + text.capitalize() + '.'
                return tokenizer(text, *a, **kw)
        processor.tokenizer = FormattedTokenizer()
    results = []
    for root in args.case_dir:
        record = json.loads((root / 'record.json').read_text())
        mission = next(m for m in record['official_result']['missions'] if m['mission'] == 4)
        baseline = mission['data']['vlm_observations']
        expected = record['case']['room_order']
        if len(baseline) != 3:
            raise ValueError('a complete three-view baseline is required')
        observations = []
        for index in range(3):
            image = root / 'camera' / f'view-{index}.png'
            started = time.monotonic()
            if bias is None:
                item = _score_image(image, model, processor, str(model_path))
            else:
                scores = {k: baseline[index]['scores'][k]-bias[k] for k in ROOM_LABELS}
                item = {'scores': scores, 'label': max(scores, key=scores.get)}
            observations.append({**item, 'expected': expected[index],
                'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                'seconds': time.monotonic()-started})
        try:
            assignment, details = assign_rooms(observations, min_margin=.25)
        except ValueError as error:
            assignment, details = [], {'error': str(error)}
        results.append({'case_dir': str(root), 'variant': args.variant, 'bias': bias,
            'observations': observations, 'assignment': assignment, 'details': details,
            'baseline_correct': sum(o['label']==e for o,e in zip(baseline,expected)),
            'candidate_correct': sum(o['label']==e for o,e in zip(observations,expected)),
            'assignment_correct': assignment == expected})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(root.name, results[-1]['candidate_correct'], assignment, flush=True)


if __name__ == '__main__':
    main()
