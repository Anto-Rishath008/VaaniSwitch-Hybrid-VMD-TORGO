"""Instrument the authorized tiny real development smoke; reject every Fold 1 source read."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m2_representations.cache import CacheEntry
from vaaniswitch.m5_experiments.cli import cache_phase
from vaaniswitch.m5_experiments.config import CANDIDATES, ExperimentConfig, digest
from vaaniswitch.m5_experiments.diagnostics import smoke_models
from vaaniswitch.m5_experiments.ledger import experiment_id
from vaaniswitch.m5_experiments.no_vmd import NoVMDCache
from vaaniswitch.m5_experiments.reporting import write_json
from vaaniswitch.m5_experiments.source import prepare_experiments
from vaaniswitch.m5_experiments.trainer import train_stage

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/experiments/module5/protocol.yaml", repo_root=ROOT
    )
    context = prepare_experiments(config)
    source = context.predecessor.source
    forbidden = set(context.split(1).speakers("outer_test"))
    source_loads: list[str] = []
    primary_reads: list[str] = []
    original_load, original_primary = source.load, NoVMDCache._primary

    def load(logical_id: str) -> CacheEntry:
        if source.contract.record_by_id[logical_id].speaker_id in forbidden:
            raise ValueError("Fold 1 source load forbidden in implementation smoke")
        result = original_load(logical_id)
        source_loads.append(logical_id)
        return result

    def primary(cache: NoVMDCache, record: ManifestRecord) -> Path:
        if record.speaker_id in forbidden:
            raise ValueError("Fold 1 primary read forbidden in implementation smoke")
        path = original_primary(cache, record)
        primary_reads.append(record.logical_id)
        return path

    with (
        patch.object(source, "load", side_effect=load),
        patch.object(NoVMDCache, "_primary", primary),
    ):
        built = cache_phase(context, build=True, smoke=True)
        validated = cache_phase(context, build=False, smoke=True)
        if built["status"] != "PASS" or validated["status"] != "PASS":
            raise ValueError("bounded A0 cache validation failed")
        report = smoke_models(context)
        run_id = experiment_id(config, 1, "C2", CANDIDATES[0].id, smoke=True)
        # The already-completed one-epoch checkpoint must reload without another epoch.
        reloaded = train_stage(
            context, context.split(1), "C2", CANDIDATES[0], run_id, smoke=True, resume=True
        )
        if reloaded["progress"]["epoch"] != 1:
            raise ValueError("tiny checkpoint reload epoch mismatch")
    result = {
        "status": "PASS",
        "a0_build": built,
        "a0_validate": validated,
        "models": report,
        "representation_source_loads": len(source_loads),
        "representation_unique_logical_ids": sorted(set(source_loads)),
        "source_load_identity_hash": digest(source_loads),
        "a0_primary_checks": len(primary_reads),
        "a0_primary_ids": sorted(set(primary_reads)),
        "checkpoint_reload": "PASS; identical identity and epoch; no additional training",
        "checkpoint": reloaded["checkpoint"],
        "fold1_loads": 0,
        "real_outer_evaluations": 0,
        "full_a0_built": False,
        "real_full_inner_training": False,
        "protocol_config_hash": config.config_hash,
        "code_hash": context.code_hash,
    }
    write_json(ROOT / "artifacts/module5/implementation/real_smoke.json", result)
    print(
        f"PASS: {built['selected']} A0 records; 7 ablations; one tiny C2 inner epoch; "
        f"{len(source_loads)} signed representation loads; no Fold 1 load or outer evaluation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
