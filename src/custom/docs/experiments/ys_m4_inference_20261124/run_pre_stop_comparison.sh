#!/usr/bin/env bash
source /home/aril/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301_verify || exit 2
cd /home/aril/rai-worktrees/ys-m2-m4-prestop-20261125/src || exit 2
export PYTHONPATH="$PWD:$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1
export HAIC_ROOT=/home/aril/rai-worktrees/ys-m2-m4-prestop-20261125
python custom/mission_4/evaluate.py --run --output /tmp/ys-m2-m4-prestop-compare-1125 \
  --case after-cone-blue
ys_status=$?
printf '%s\n' "$ys_status" > /tmp/ys-opt-1124/prestop-comparison.status
exit "$ys_status"
