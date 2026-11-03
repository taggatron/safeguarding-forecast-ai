"""Root-level module wrapper re-exporting backend.app.features."""
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.features import *  # noqa: F401,F403

