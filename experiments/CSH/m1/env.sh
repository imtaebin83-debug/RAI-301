#!/usr/bin/env bash
# Source from any directory. Override activation hooks before importing custom.
source /home/aril/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301_v2
export HAIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$HAIC_ROOT/src:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages"
export MLSPACES_ASSETS_DIR="$HAIC_ROOT/src/environment/models/assets/molmo"
export PYTHONDONTWRITEBYTECODE=1
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=92 MUJOCO_GL=egl
cd "$HAIC_ROOT/src"
