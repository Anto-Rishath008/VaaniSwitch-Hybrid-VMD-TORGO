"""Module 5 phases are explicit; defaults never launch full training or a full cache."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m4_fusion.trainer import tiny_records
from vaaniswitch.m5_experiments.aggregation import aggregate, load_outer_results
from vaaniswitch.m5_experiments.config import ABLATIONS, ExperimentConfig
from vaaniswitch.m5_experiments.diagnostics import smoke_models
from vaaniswitch.m5_experiments.evaluator import evaluate_outer
from vaaniswitch.m5_experiments.ledger import Ledger
from vaaniswitch.m5_experiments.no_vmd import NoVMDCache
from vaaniswitch.m5_experiments.pipeline import prepare_ablations, retrain_fold, search_c2
from vaaniswitch.m5_experiments.reporting import write_json
from vaaniswitch.m5_experiments.robustness import robustness
from vaaniswitch.m5_experiments.source import (
    ExperimentContext,
    freeze_protocol,
    prepare_experiments,
    protocol_document,
    require_frozen,
)


def a0_records(context: ExperimentContext, *, smoke: bool) -> tuple[ManifestRecord, ...]:
    if not smoke:
        return tuple(
            sorted(context.predecessor.source.contract.records, key=lambda r: r.logical_id)
        )
    records = context.split(1).records(context.predecessor.source.contract, "inner_train")
    panel = list(tiny_records(records))
    predicates: tuple[Callable[[ManifestRecord], bool], ...] = (
        lambda r: r.duration_seconds is not None and r.duration_seconds > 8,
        lambda r: r.primary_microphone is not None and r.primary_microphone.value == "array",
    )
    for predicate in predicates:
        record = next((r for r in records if predicate(r)), None)
        if record is not None and record not in panel:
            panel.append(record)
    return tuple(panel)


def cache_phase(context: ExperimentContext, *, build: bool, smoke: bool) -> dict[str, Any]:
    if not smoke:
        require_frozen(context)
    cache = NoVMDCache(context)
    records = a0_records(context, smoke=smoke)
    counts = {"computed": 0, "cache_hit": 0, "validated": 0}
    failures = []
    start = time.perf_counter()
    ledger = Ledger(context.config, smoke=smoke)
    with ledger.lock("a0_cache"):
        for index, record in enumerate(records, 1):
            try:
                if build:
                    counts[cache.build(record)] += 1
                cache.load(record, check_source=True)
                counts["validated"] += 1
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append({"logical_id": record.logical_id, "error": str(exc)})
            if index % 250 == 0:
                print(f"A0 processed {index}/{len(records)}; failures={len(failures)}", flush=True)
        stage = "build" if build else "validate"
        mode = "smoke" if smoke else "full"
        report = {
            "status": "FAIL" if failures else "PASS",
            "scope": f"a0_{mode}_{stage}",
            "selected": len(records),
            **counts,
            "failed": len(failures),
            "a0_config_hash": cache.config_hash,
            "experiment_config_hash": context.config.config_hash,
            "namespace": str(cache.directory(records[0].logical_id).parents[2]),
            "elapsed_seconds": time.perf_counter() - start,
            "failures": failures,
            "outer_model_forwards": 0,
        }
        write_json(context.config.output(f"a0/{mode}_{stage}.json"), report)
        path = context.config.output(f"a0/{mode}_{stage}_failures.csv")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["logical_id", "error"])
            writer.writeheader()
            writer.writerows(failures)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", type=Path, default=Path("configs/experiments/module5/protocol.yaml")
    )
    sub = result.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol", help="metadata-only protocol audit/freeze")
    protocol.add_argument("--freeze", action="store_true")
    sub.add_parser("dry-run", help="metadata and candidate plan only; no model construction")
    for name in ("a0-build", "a0-validate"):
        cache = sub.add_parser(name)
        group = cache.add_mutually_exclusive_group(required=True)
        group.add_argument("--smoke", action="store_true")
        group.add_argument(
            "--full", action="store_true", help="explicit manual full CORE operation"
        )
    sub.add_parser("smoke", help="seven tiny development forwards and one tiny inner epoch")
    for name in ("search", "ablation-inner", "retrain"):
        command = sub.add_parser(name, help="expensive manual training phase")
        command.add_argument("--outer-fold", type=int, choices=[1, 2, 3], required=True)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--retry-failed", metavar="REASON")
    outer = sub.add_parser("outer", help="manual one-shot evaluation after all fold retraining")
    outer.add_argument("--outer-fold", type=int, choices=[1, 2, 3], required=True)
    outer.add_argument("--ablation", choices=ABLATIONS, required=True)
    sub.add_parser("aggregate")
    sub.add_parser("robustness")
    sub.add_parser("ledger")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path.cwd().resolve()
    try:
        config = ExperimentConfig.from_yaml(root / args.config, repo_root=root)
        context = prepare_experiments(config)
        if args.command in {"protocol", "dry-run"}:
            report = (
                freeze_protocol(context)
                if getattr(args, "freeze", False)
                else protocol_document(context)
            )
        elif args.command in {"a0-build", "a0-validate"}:
            report = cache_phase(context, build=args.command == "a0-build", smoke=args.smoke)
        elif args.command == "smoke":
            report = smoke_models(context)
        elif args.command in {"search", "ablation-inner", "retrain"}:
            function = {
                "search": search_c2,
                "ablation-inner": prepare_ablations,
                "retrain": retrain_fold,
            }[args.command]
            report = function(
                context, args.outer_fold, resume=args.resume, retry_reason=args.retry_failed
            )
        elif args.command == "outer":
            result = evaluate_outer(context, args.outer_fold, args.ablation)
            report = {k: v for k, v in result.items() if k != "predictions"}
        elif args.command in {"aggregate", "robustness"}:
            require_frozen(context)
            results = load_outer_results(config)
            report = (
                aggregate(results) if args.command == "aggregate" else robustness(context, results)
            )
            write_json(config.output(f"reports/{args.command}.json"), report)
        else:
            report = {"runs": Ledger(config).all_runs()}
        print(json.dumps(report, indent=2, allow_nan=False))
        return 1 if report.get("status") == "FAIL" else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
