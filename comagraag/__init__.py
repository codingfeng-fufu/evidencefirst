import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent
_parent = _pkg_root.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
