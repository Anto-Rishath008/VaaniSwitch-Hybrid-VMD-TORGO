"""Launcher for Module 6 Phase 2-4 / final evaluation without installing the root project."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vaaniswitch.m6_reconstruction.phase234 import main

if __name__ == "__main__":
    main()
