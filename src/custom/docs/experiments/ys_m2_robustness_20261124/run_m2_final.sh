#!/usr/bin/env bash
# Dedicated ys evaluation process; the standard simulator wrapper owns the lock.
source /home/aril/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301_verify || exit 2
cd /home/aril/rai-worktrees/ys-m2-stop-20261124/src || exit 2
export PYTHONPATH="$PWD:$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1
export HAIC_ROOT=/home/aril/rai-worktrees/ys-m2-stop-20261124
unset HAIC_M2_CAPTURE_DIR
python custom/mission_2/evaluate.py --run --output /tmp/ys-m2-stop-matrix-1124 \
  --case order-3-blue --case order-3-green --case order-3-orange \
  --case order-4-blue --case order-4-green --case order-4-orange \
  --case order-5-blue --case order-5-green --case order-5-orange
ys_status=$?
printf '%s\n' "$ys_status" > /tmp/ys-opt-1124/m2-stop-matrix.status
exit "$ys_status"
