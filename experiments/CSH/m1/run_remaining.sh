#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u

# Each flock wait is bounded; only this task's evaluator is launched. Separate
# batches release the shared device between unrelated validation groups.
with_gpu() {
  local attempt status
  for attempt in {1..20}; do
    if flock -w 55 -E 75 /tmp/rai-gpu-simulator.lock "$@"; then
      return 0
    else
      status=$?
      if [[ "$status" != 75 ]]; then return "$status"; fi
    fi
  done
  return 75
}

with_gpu python -u custom/testing/m1_regression.py \
  --source /home/aril/rai-worktrees/CSH-m1-20261127-02-integration/src \
  --case rooms-2 --case rooms-3 --output ../experiments/CSH/m1/final-integration-rooms
with_gpu python -u custom/testing/m1_regression.py \
  --source /home/aril/rai-worktrees/CSH-m1-20261127-02-integration/src \
  --case after-cone --output ../experiments/CSH/m1/final-integration-continuous
with_gpu python -u custom/testing/m1_regression.py \
  --case mission3 --case mission6 --output ../experiments/CSH/m1/final-other-missions
with_gpu python -u custom/testing/web_smoke.py --output ../experiments/CSH/m1/final-web
with_gpu python -u custom/testing/m1_regression.py --seed 9407 --cases 2 \
  --output ../experiments/CSH/m1/final-evaluation
