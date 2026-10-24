"""Pytest configuration for tests package."""
import sys
from pathlib import Path

# Ensure root directory is always on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
