"""Owner unit suites with their existing optional vision API, CPU fixture only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
source = repo.parent / (repo.name + '-v2-integration') / 'src'
vision = subprocess.check_output(['git', 'show', '2284468:src/custom/vision.py'], cwd=repo)
with tempfile.TemporaryDirectory(prefix='csh-v2-owner-cpu-') as directory:
    temporary = Path(directory)
    for path in (source / 'custom').rglob('*.py'):
        if 'models' in path.relative_to(source).parts:
            continue
        target = temporary / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ('mission_api.py', 'referee.py'):
        shutil.copy2(source / name, temporary / name)
    (temporary / 'custom/vision.py').write_bytes(vision)
    result = subprocess.run([sys.executable, '-m', 'unittest',
        'custom.mission_2.test_mission_2', 'custom.mission_4.test_mission_4', '-v'],
        cwd=temporary, env={**os.environ,
            'PYTHONPATH': str(temporary) + os.pathsep + os.environ.get('PYTHONPATH', '')},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30.)
    (root / 'v2_owner_tests_complete_overlay.log').write_text(result.stdout)
    (root / 'v2_owner_tests_complete_overlay.json').write_text(json.dumps({
        'returncode': result.returncode, 'kind': 'CPU unit suites only; no simulator',
        'base_source': str(source), 'vision_from': '2284468:src/custom/vision.py',
        'vision_sha256': hashlib.sha256(vision).hexdigest(),
        'difference_from_live_simulation_fixture':
            'owner optional RGBD.cone(color, data=None) argument; simulation used unchanged main vision.py',
        'submission_vision_changes': 'none'}, indent=2) + '\n')
    print(result.stdout[-700:])
    raise SystemExit(result.returncode)
