"""CLI that never offers an outer-test evaluation mode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from vaaniswitch.m4_fusion.config import Module4Config
from vaaniswitch.m4_fusion.pipeline import (
    audit_development_split,
    run_training,
    validate_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Module 4 development classifier; no test evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("split-audit", "smoke", "train", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/models/fusion.yaml"))
        command.add_argument("--outer-test-fold", type=int, required=True)
        command.add_argument("--seed", type=int, default=42, choices=[42])
        command.add_argument("--batch-size", type=int)
        if name == "train":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument(
                "--smoke", action="store_true", help="one epoch on tiny dev panels"
            )
            command.add_argument("--resume", type=Path)
        if name == "validate":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--smoke", action="store_true", help="validate a smoke checkpoint")
    args = parser.parse_args(argv)
    config: Module4Config | None = None
    try:
        config = Module4Config.from_yaml(args.config).with_batch_size(args.batch_size)
        result: dict[str, Any]
        if args.command == "split-audit":
            result = audit_development_split(config, args.outer_test_fold)
        elif args.command == "train":
            result = run_training(
                config,
                outer_test_fold=args.outer_test_fold,
                smoke=args.smoke,
                dry_run=args.dry_run,
                resume=args.resume,
            )
        elif args.command == "validate":
            result = validate_checkpoint(
                config, args.checkpoint, outer_test_fold=args.outer_test_fold, smoke=args.smoke
            )
        else:
            from vaaniswitch.m4_fusion.diagnostics import run_smoke

            result = run_smoke(config, outer_test_fold=args.outer_test_fold)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        if config is not None:
            path = config.output("logs", "module4_fusion_failures.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                if new:
                    writer.writerow(["command", "error_type", "message"])
                writer.writerow([args.command, type(exc).__name__, str(exc)])
        print(json.dumps({"status": "FAIL", "error": str(exc)}), flush=True)
        return 1
