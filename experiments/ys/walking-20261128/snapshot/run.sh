#!/bin/bash
set -u
cd /home/aril/rai-training/ys-walk-progress-pilot-20261128
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
date -u > started.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu --format=csv -l 10 > gpu.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=30s 3600 /home/aril/miniforge3/envs/unitree_mjlab_omen5060/bin/python -u pilot.py --iterations 500 --envs 512 --initialize-actor /home/aril/rai-training/ys-walk-restarts-refine-20261128/iteration_0500/checkpoint.pt > train.log 2>&1
rc=$?
echo "$rc" > exit.txt
date -u > ended.txt
python3 assess.py > assessment.log 2>&1
exit "$rc"
