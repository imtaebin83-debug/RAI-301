"""Optional development runtime adapter; final judging uses Mission5.run()."""
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from custom.mission_5.policy import create_policy  # noqa: E402,F401
