"""CLI for strict Module 3 source validation and encoder smoke diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.pipeline import run_module3_smoke, validate_module3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaaniswitch-m3", description="Parallel Mel, visibility-graph, and RP encoders"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "validate"):
        command = subcommands.add_parser(name)
        command.add_argument(
            "--config", type=Path, default=Path("configs/models/encoders.yaml")
        )
        command.add_argument("--limit", type=_positive_integer, default=8)
    smoke = subcommands.choices["smoke"]
    smoke.add_argument("--batch-size", type=_positive_integer, default=2)
    smoke.add_argument("--timing-repetitions", type=_positive_integer, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = Module3Config.from_yaml(args.config)
        if args.command == "smoke":
            result = run_module3_smoke(
                config,
                limit=args.limit,
                batch_size=args.batch_size,
                timing_repetitions=args.timing_repetitions,
            )
        else:
            result = validate_module3(config, limit=args.limit)
    except (OSError, ValueError, RuntimeError) as exc:
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
