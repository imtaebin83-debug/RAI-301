"""Recreate the evaluation-only teammate overlay in a new detached worktree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='new, nonexistent worktree path')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
destination = args.output.resolve()
if destination.exists():
    parser.error('output already exists; never overwrite an existing checkout')
files = [
    ('228446808d3f3c2c090428afdcc92049d2d52fe3', 'src/custom/mission_2/mission_2.py'),
    ('228446808d3f3c2c090428afdcc92049d2d52fe3', 'src/custom/mission_2/perception.py'),
    ('36d6ee5fb18adbbb3900d6345e4ad20672925087', 'src/custom/mission_4/mission_4.py'),
    ('36d6ee5fb18adbbb3900d6345e4ad20672925087', 'src/custom/mission_4/room_scoring.py'),
    ('36d6ee5fb18adbbb3900d6345e4ad20672925087', 'src/custom/vlm.py'),
]
# Resolve all required read-only objects before creating the destination.
payloads = [(revision, name, subprocess.check_output(
    ['git', 'show', revision + ':' + name], cwd=repo)) for revision, name in files]
subprocess.run(['git', 'worktree', 'add', '--detach', str(destination), '2cae201'], cwd=repo, check=True)
manifest = []
for revision, name, original in payloads:
    data = original
    if name.endswith(('mission_2.py', 'mission_4.py')):
        data = data.replace(b'from mission_api import Mission, MissionResult, run_standalone',
            b'from custom.sensor_mission import Mission\nfrom mission_api import MissionResult, run_standalone')
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    manifest.append({'revision': revision, 'path': name,
        'original_sha256': hashlib.sha256(original).hexdigest(),
        'evaluation_sha256': hashlib.sha256(data).hexdigest()})
(destination / 'CSH_EVALUATION_FIXTURE.json').write_text(json.dumps({
    'base': '2cae201', 'kind': 'evaluation-only overlay', 'files': manifest,
    'vision.py': 'unchanged origin/main; optional owner data argument used only in separate CPU suite',
    'models': 'set HAIC_VLM_MODEL to read-only candidate custom/models/SmolVLM-500M-Instruct'
}, indent=2) + '\n')
print(destination / 'src')
