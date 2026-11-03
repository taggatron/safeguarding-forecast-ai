"""Root-level module wrapper re-exporting backend.app.evaluation.kaggle_metrics."""
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.evaluation.kaggle_metrics import *  # noqa: F401,F403

