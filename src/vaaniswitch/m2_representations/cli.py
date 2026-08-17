"""Command-line interface for Module 2 representation caching and validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vaaniswitch.m2_representations.config import Module2Config
from vaaniswitch.m2_representations.pipeline import (
    Module2PipelineError,
    run_module2,
    validate_module2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaaniswitch-m2",
        description="Log-Mel, recurrence, and natural visibility representations",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = subcommands.add_parser(name)
        command.add_argument(
            "--config", type=Path, default=Path("configs/representations/default.yaml")
        )
        command.add_argument("--subset", choices=("core",), default=None)
        command.add_argument("--limit", type=_positive_integer, default=None)
    subcommands.choices["build"].add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = Module2Config.from_yaml(args.config)
        if args.command == "build":
            result = run_module2(config, subset=args.subset, limit=args.limit, force=args.force)
        else:
            result = validate_module2(config, subset=args.subset, limit=args.limit)
    except (OSError, ValueError, Module2PipelineError) as exc:
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
