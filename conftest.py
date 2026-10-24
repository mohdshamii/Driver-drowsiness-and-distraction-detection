"""Pytest configuration and root sys.path setup."""
import sys
from pathlib import Path

# Ensure root directory is always on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
