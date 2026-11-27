#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u
baseline_src=/home/aril/rai-worktrees/CSH-m1-20261127-02-v2-baseline/src
with_gpu() {
  local attempt status
  for attempt in {1..20}; do
    if flock -w 55 -E 75 /tmp/rai-gpu-simulator.lock "$@"; then return 0; else
      status=$?
      if [[ "$status" != 75 ]]; then return "$status"; fi
    fi
  done
  return 75
}
with_gpu python -u custom/testing/randomized_suite.py --source "$baseline_src" \
  --seed 301 --cases 2 --missions 1 2 --output ../experiments/CSH/m1/v2-baseline-seed301
with_gpu python -u custom/testing/m1_regression.py --source "$baseline_src" \
  --case roundtrip --output ../experiments/CSH/m1/v2-baseline-roundtrip
