"""Command-line interface for Module 1 VMD caching and validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vaaniswitch.m1_vmd.config import Module1Config
from vaaniswitch.m1_vmd.pipeline import Module1PipelineError, run_module1, validate_module1


def build_parser() -> argparse.ArgumentParser:
    """Build the Module 1 command parser."""

    parser = argparse.ArgumentParser(
        prog="vaaniswitch-m1",
        description="Shared VMD decomposition and all-mode reconstruction cache",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = subcommands.add_parser(name)
        command.add_argument(
            "--config",
            type=Path,
            default=Path("configs/vmd/default.yaml"),
            help="Module 1 YAML configuration",
        )
        command.add_argument(
            "--subset",
            choices=("core", "extended"),
            default=None,
            help="Manifest subset; defaults to the config's CORE subset",
        )
        command.add_argument(
            "--limit",
            type=_positive_integer,
            default=None,
            help="Deterministic condition-balanced smoke limit",
        )
    build = subcommands.choices["build"]
    build.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected entries even when valid caches exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a Module 1 build or validation command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = Module1Config.from_yaml(args.config)
        if args.command == "build":
            result = run_module1(
                config,
                subset=args.subset,
                limit=args.limit,
                force=args.force,
            )
        else:
            result = validate_module1(
                config,
                subset=args.subset,
                limit=args.limit,
            )
    except (OSError, ValueError, Module1PipelineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
