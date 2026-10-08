"""Prepare a new immutable M6 freeze or execute/resume its once-only final panel."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vaaniswitch.m6_reconstruction.final_pipeline import final_evaluation, prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--freeze-sha256")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    else:
        if args.freeze is None or not args.freeze_sha256:
            parser.error("run requires --freeze and --freeze-sha256")
        path = args.freeze if args.freeze.is_absolute() else root / args.freeze
        final_evaluation(root, path, args.freeze_sha256)
