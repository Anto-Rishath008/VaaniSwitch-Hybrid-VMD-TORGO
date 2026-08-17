"""Command-line interface for building and validating Module 0 outputs."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from vaaniswitch.m0_data.config import TorgoConfig
from vaaniswitch.m0_data.pipeline import run_module0, validate_generated_outputs


def build_parser() -> argparse.ArgumentParser:
    """Create the stable Module 0 CLI parser."""

    parser = argparse.ArgumentParser(
        prog="vaaniswitch-m0",
        description="Build TORGO manifests, audio audit, and speaker-disjoint folds.",
    )
    parser.add_argument(
        "command",
        choices=("build", "validate"),
        nargs="?",
        default="build",
        help="build all outputs or validate outputs already on disk",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/torgo.yaml"),
        help="path to the Module 0 YAML configuration",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="optional explicit repository root (normally inferred from config path)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return a process exit status."""

    args = build_parser().parse_args(argv)
    try:
        config = TorgoConfig.from_yaml(args.config, repo_root=args.repo_root)
        if args.command == "build":
            summary = run_module0(config)
            result = {
                "status": "complete",
                "logical_utterances": summary["logical_utterances"]["total"],
                "excluded": summary["logical_utterances"]["excluded_count"],
                "leakage_status": summary["leakage_and_duplicates"]["status"],
                "summary_path": config.relative_path(config.outputs["dataset_summary"]),
            }
        else:
            result = validate_generated_outputs(config)
    except Exception:
        logging.exception("Module 0 command failed")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
