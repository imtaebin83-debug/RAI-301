#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u
# Evaluation-only teammate overlay; models remain read-only inside custom.
export HAIC_VLM_MODEL="$HAIC_ROOT/src/custom/models/SmolVLM-500M-Instruct"
integration_src=/home/aril/rai-worktrees/CSH-m1-20261127-02-v2-integration/src
for attempt in {1..30}; do
  if flock -w 55 -E 75 /tmp/rai-gpu-simulator.lock \
    python -u custom/testing/m1_regression.py --source "$integration_src" \
      --case cone-delivery-return --output ../experiments/CSH/m1/v2-integration; then
    exit 0
  else
    status=$?
    if [[ "$status" != 75 ]]; then exit "$status"; fi
  fi
done
exit 75
