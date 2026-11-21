#!/usr/bin/env bash
# Activate the intended competition Conda environment before running this tool.
set -euo pipefail
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SRC_ROOT"
if [[ "${1:-}" == "--headless" ]]; then
  shift
  CONFIG="$(mktemp /tmp/rai-headless.XXXXXX.yaml)"
  trap 'rm -f "$CONFIG"' EXIT
  python - "$SRC_ROOT/main_config.yaml" "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
settings=yaml.safe_load(Path(sys.argv[1]).read_text())
settings['simulator']['view']=False
settings['instruction']['open_browser']=False
Path(sys.argv[2]).write_text(yaml.safe_dump(settings))
PY
  export MUJOCO_GL=egl
  flock -n -E 75 /tmp/rai-gpu-simulator.lock python "$SRC_ROOT/main.py" --config "$CONFIG" "$@"
else
  flock -n -E 75 /tmp/rai-gpu-simulator.lock python "$SRC_ROOT/main.py" "$@"
fi
