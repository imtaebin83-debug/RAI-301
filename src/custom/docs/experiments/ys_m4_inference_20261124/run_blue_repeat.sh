#!/usr/bin/env bash
source /home/aril/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301_verify || exit 2
cd /home/aril/rai-worktrees/ys-m2-m4-opt-20261124/src || exit 2
export PYTHONPATH="$PWD:$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1
export HAIC_ROOT=/home/aril/rai-worktrees/ys-m2-m4-opt-20261124
python custom/mission_4/evaluate.py --run --output /tmp/ys-m2-m4-opt-blue-repeat-1125 \
  --case after-cone-blue
ys_status=$?
printf '%s\n' "$ys_status" > /tmp/ys-opt-1124/blue-repeat.status
exit "$ys_status"
