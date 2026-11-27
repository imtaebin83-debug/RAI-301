#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${HAIC_ENV:-}"
if [[ ! "$ENV_NAME" =~ ^haic_[A-Za-z0-9_-]+$ ]]; then
    echo "set HAIC_ENV to haic_<team name> first, e.g.:" >&2
    echo "  export HAIC_ENV=haic_team1" >&2
    echo "  bash $0" >&2
    exit 1
fi
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAIC_ROOT="$(cd "$SETUP_DIR/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found -- install Miniconda first (README: Setup)" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$SETUP_DIR"                      # env.yml refers to requirements.txt relatively
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[haic] updating conda env '$ENV_NAME' from env.yml"
    conda env update -n "$ENV_NAME" -f env.yml
else
    echo "[haic] creating conda env '$ENV_NAME' from env.yml"
    conda env create -n "$ENV_NAME" -f env.yml
fi

PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"

cat > "$PREFIX/etc/conda/activate.d/haic.sh" <<EOF
# written by $SETUP_DIR/install_conda_env.sh -- rerun it after moving the checkout
export HAIC_ROOT="$HAIC_ROOT"
export _HAIC_OLD_PYTHONPATH="\${PYTHONPATH:-}"
if [ -n "\${BASH_VERSION:-}" ]; then
    source /opt/ros/humble/setup.bash
    if [ -f "\$HAIC_ROOT/install/setup.bash" ]; then
        source "\$HAIC_ROOT/install/setup.bash"
    fi
else
    echo "[haic] ROS setup scripts need bash; source /opt/ros/humble/setup.zsh yourself" >&2
fi
export PYTHONPATH="\$HAIC_ROOT/src:\${PYTHONPATH:-}"
export MLSPACES_ASSETS_DIR="\$HAIC_ROOT/src/environment/models/assets/molmo"
export ROS_LOCALHOST_ONLY=1
EOF

cat > "$PREFIX/etc/conda/deactivate.d/haic.sh" <<'EOF'
export PYTHONPATH="${_HAIC_OLD_PYTHONPATH:-}"
unset _HAIC_OLD_PYTHONPATH MLSPACES_ASSETS_DIR HAIC_ROOT
EOF

echo "[haic] checking the env"
set +u                               # ROS setup scripts read unset variables
conda activate "$ENV_NAME"
python - <<'PY'
import importlib
for mod in ("mujoco", "onnxruntime", "cv2", "PIL", "shapely", "scipy", "rclpy",
            "h5py", "teledex", "qrcode"):
    try:
        importlib.import_module(mod)
        print(f"  ok   {mod}")
    except Exception as exc:                       # noqa: BLE001
        print(f"  FAIL {mod}: {exc}")
PY
echo "[haic] done.  conda env '$ENV_NAME' -> $HAIC_ROOT"
echo "[haic] new terminals:  conda activate $ENV_NAME"
