"""Module 6 launcher without installing the root project or importing M0-M5."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vaaniswitch.m6_reconstruction.runtime import main

if __name__ == "__main__":
    main()
