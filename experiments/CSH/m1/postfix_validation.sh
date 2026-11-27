#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u
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
with_gpu python -u custom/testing/m1_regression.py --case mission3 --case mission6 \
  --output ../experiments/CSH/m1/postfix-other
with_gpu python -u custom/testing/randomized_suite.py --seed 301 --cases 2 --missions 1 2 \
  --output ../experiments/CSH/m1/postfix-seed301
with_gpu python -u custom/testing/m1_regression.py --seed 9509 --cases 2 \
  --output ../experiments/CSH/m1/postfix-evaluation
