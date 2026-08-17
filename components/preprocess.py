"""Data preprocessing entry point for VaaniSwitch.

Runs manifest generation (Module 0), VMD caching (Module 1), and multirepresentation
extraction (Module 2) end-to-end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from components.data.config import TorgoConfig
from components.data.pipeline import run_module0, validate_generated_outputs
from components.vmd.config import Module1Config
from components.vmd.pipeline import run_module1, validate_module1
from components.representations.config import Module2Config
from components.representations.pipeline import run_module2, validate_module2


def main() -> int:
    """Parse command line arguments and execute the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="VaaniSwitch End-to-End Preprocessing Pipeline",
    )
    parser.add_argument(
        "action",
        choices=("run", "validate"),
        default="run",
        nargs="?",
        help="run preprocessing from scratch or validate existing cache outputs",
    )
    parser.add_argument(
        "--m0-config",
        type=Path,
        default=Path("configs/data/torgo.yaml"),
        help="Path to Module 0 (manifest) configuration",
    )
    parser.add_argument(
        "--m1-config",
        type=Path,
        default=Path("configs/vmd/default.yaml"),
        help="Path to Module 1 (VMD cache) configuration",
    )
    parser.add_argument(
        "--m2-config",
        type=Path,
        default=Path("configs/representations/default.yaml"),
        help="Path to Module 2 (representations) configuration",
    )
    parser.add_argument(
        "--subset",
        choices=("core", "extended"),
        default=None,
        help="Manifest subset to process (defaults to core)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test limit to process only a subset of records",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild cache even if it is valid on disk",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Explicit repository root directory",
    )

    args = parser.parse_args()

    try:
        if args.action == "run":
            print(">>> [1/3] Building TORGO manifest and auditing quality...")
            config0 = TorgoConfig.from_yaml(args.m0_config, repo_root=args.repo_root)
            run_module0(config0)

            print(">>> [2/3] Building VMD caches (requires raw TORGO database)...")
            config1 = Module1Config.from_yaml(args.m1_config)
            run_module1(config1, subset=args.subset, limit=args.limit, force=args.force)

            print(">>> [3/3] Generating Mel, Recurrence, and Visibility Graph representations...")
            config2 = Module2Config.from_yaml(args.m2_config)
            result = run_module2(config2, subset=args.subset, limit=args.limit, force=args.force)

            print(">>> Preprocessing complete!")
            print(json.dumps(result, indent=2, sort_keys=True))

        else:
            print(">>> Validating TORGO manifest and metadata...")
            config0 = TorgoConfig.from_yaml(args.m0_config, repo_root=args.repo_root)
            validate_generated_outputs(config0)

            print(">>> Validating VMD caches...")
            config1 = Module1Config.from_yaml(args.m1_config)
            validate_module1(config1, subset=args.subset, limit=args.limit)

            print(">>> Validating Multirepresentations...")
            config2 = Module2Config.from_yaml(args.m2_config)
            result = validate_module2(config2, subset=args.subset, limit=args.limit)

            print(">>> Validation successful!")
            print(json.dumps(result, indent=2, sort_keys=True))

    except Exception as exc:
        print(f"ERROR: Preprocessing failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
