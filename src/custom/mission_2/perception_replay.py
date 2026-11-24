"""Evaluation only: stress saved RGB-D frames without navigation or ground truth."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from custom.mission_2.perception import ConeSensor


def replay(path):
    color = path.name.split('-')[0]
    if color not in ('blue', 'green', 'orange'):
        raise ValueError('Capture filename must start with its survey color')
    with np.load(path, allow_pickle=False) as archive:
        original = tuple(archive[key].copy() for key in ('rgb', 'depth', 'world', 'height', 'valid'))
    detector = ConeSensor.__new__(ConeSensor)
    reference = detector.cone(color, original)
    results = []
    for variant in ('original', 'dim-60-percent', 'bright-130-percent',
                    'missing-depth-20-percent', 'vertical-offset-2m', 'all-invalid'):
        rgb, depth, world, height, valid = (item.copy() for item in original)
        if variant.startswith('dim'):
            rgb = (rgb.astype(float)*.6).astype(np.uint8)
        elif variant.startswith('bright'):
            rgb = np.clip(rgb.astype(float)*1.3, 0, 255).astype(np.uint8)
        elif variant.startswith('missing'):
            missing = np.random.default_rng(1124).random(depth.shape) < .2
            depth[missing] = np.nan
            world[missing] = np.nan
            valid[missing] = False
        elif variant.startswith('vertical'):
            world[:, :, 2] += 2.
            height += 2.
        elif variant == 'all-invalid':
            valid[:] = False
        item = detector.cone(color, (rgb, depth, world, height, valid))
        error = (float(np.hypot(item.x-reference.x, item.y-reference.y))
                 if item is not None and reference is not None else None)
        # No visibility is invented for frames where the unmodified sensor
        # cannot detect the target. Those remain explicitly unscored.
        passed = (item is None if variant == 'all-invalid' else
                  error is not None and error <= .05 if reference is not None else None)
        results.append({'variant': variant, 'detected': item is not None,
                        'xy': [item.x, item.y] if item else None,
                        'shift_from_original_m': error, 'passed': passed})
    return {'capture': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'color': color, 'reference_detected': reference is not None, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('captures', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    results = [replay(path) for path in args.captures]
    scored = [item for frame in results for item in frame['results'] if item['passed'] is not None]
    report = {'frames': results, 'passed': sum(item['passed'] for item in scored),
              'scored': len(scored), 'unscored': 6*len(results)-len(scored),
              'detector_sha256': hashlib.sha256(
                  Path(__file__).with_name('perception.py').read_bytes()).hexdigest(),
              'scope': 'Saved-frame perturbations, not simulator success or new camera observations'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ('passed', 'scored', 'unscored')}))
    if any(item['passed'] is False for item in scored):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
