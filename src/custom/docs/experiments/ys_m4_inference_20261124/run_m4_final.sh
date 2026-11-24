#!/usr/bin/env bash
# Both batches use the standard wrapper; never acquire its lock a second time.
source /home/aril/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301_verify || exit 2
export PYTHONDONTWRITEBYTECODE=1
cd /home/aril/rai-worktrees/ys-m4-opt-20261124/src || exit 2
export PYTHONPATH="$PWD:$PYTHONPATH" HAIC_ROOT=/home/aril/rai-worktrees/ys-m4-opt-20261124
python custom/mission_4/evaluate.py --run --output /tmp/ys-m4-opt-live-1124 \
  --case rooms-0 --case rooms-2 --case rooms-3
ys_standalone_status=$?
printf '%s\n' "$ys_standalone_status" > /tmp/ys-opt-1124/m4-opt-live.status
if [ "$ys_standalone_status" -eq 75 ]; then exit 75; fi
cd /home/aril/rai-worktrees/ys-m2-m4-opt-20261124/src || exit 2
export PYTHONPATH="$PWD:$PYTHONPATH" HAIC_ROOT=/home/aril/rai-worktrees/ys-m2-m4-opt-20261124
python custom/mission_4/evaluate.py --run --output /tmp/ys-m2-m4-opt-live-1124 \
  --case after-cone --case after-cone-blue
ys_combined_status=$?
printf '%s\n' "$ys_combined_status" > /tmp/ys-opt-1124/m2-m4-opt-live.status
exit "$ys_combined_status"
