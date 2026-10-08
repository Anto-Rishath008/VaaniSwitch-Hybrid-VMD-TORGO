"""Verify signed predecessor identities before preparing metadata-only experiment roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vaaniswitch.m0_data.leakage import speaker_folds_fingerprint
from vaaniswitch.m1_vmd.config import Module1Config
from vaaniswitch.m4_fusion.config import Module4Config
from vaaniswitch.m4_fusion.source import DevelopmentContext, prepare_context
from vaaniswitch.m5_experiments.config import SIGNED_HASHES, ExperimentConfig, digest, file_hash
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.splits import NestedSplit, nested_splits, record_hash


@dataclass(frozen=True)
class ExperimentContext:
    config: ExperimentConfig
    predecessor: DevelopmentContext
    splits: tuple[NestedSplit, ...]
    audio_config: Module1Config

    def split(self, fold: int) -> NestedSplit:
        for split in self.splits:
            if type(fold) is int and split.outer_fold == fold:
                return split
        raise ValueError("unknown outer fold")

    @property
    def code_hash(self) -> str:
        root = self.config.repo_root
        files = sorted((root / "src/vaaniswitch").glob("m[012345]*/*.py"))
        return digest({p.relative_to(root).as_posix(): file_hash(p) for p in files})


def prepare_experiments(config: ExperimentConfig) -> ExperimentContext:
    root = config.repo_root
    m4 = Module4Config.from_yaml(root / "configs/models/fusion.yaml", repo_root=root)
    if m4.hashes != SIGNED_HASHES:
        raise ValueError("wrong signed Module 0-4 identities")
    audit = read_json(root / "artifacts/metrics/module4_full_signoff_audit.json")
    if (
        audit.get("status") != "PASS"
        or audit.get("module4_signed_off") is not True
        or audit.get("hashes") != SIGNED_HASHES
    ):
        raise ValueError("Module 4 must be signed off with exact predecessor hashes")
    m1 = Module1Config.from_yaml(root / "configs/vmd/default.yaml", repo_root=root)
    signoff1 = read_json(root / "artifacts/metrics/module1_full_signoff_audit.json")
    leakage = read_json(root / "artifacts/logs/module0_leakage_report.json")
    if (
        m1.config_hash != SIGNED_HASHES["module1_config_hash"]
        or signoff1.get("status") != "PASS"
        or signoff1.get("source", {}).get("config_hash") != m1.config_hash
        or signoff1.get("source", {}).get("manifest_sha256")
        != SIGNED_HASHES["manifest_fingerprint"]
        or leakage.get("status") != "PASS"
    ):
        raise ValueError("Module 0/1 source sign-off mismatch")
    predecessor = prepare_context(m4, 1)
    folds = read_json(predecessor.encoder_config.source.speaker_folds_path)
    fingerprint = speaker_folds_fingerprint(folds)
    if fingerprint != audit["split"]["folds_fingerprint"]:
        raise ValueError("signed outer fold document changed")
    splits = nested_splits(predecessor.source.contract, fingerprint)
    return ExperimentContext(config, predecessor, splits, m1)


def protocol_document(context: ExperimentContext) -> dict[str, Any]:
    contract = context.predecessor.source.contract
    splits = []
    for split in context.splits:
        roles = {}
        for role in ("inner_train", "inner_validation", "development", "outer_test"):
            records = split.records(contract, role)
            roles[role] = {
                "speakers": list(split.speakers(role)),
                "record_count": len(records),
                "record_hash": record_hash(records),
            }
        splits.append({"split_hash": split.split_hash, "identity": split.identity, "roles": roles})
    return {
        "protocol": context.config.identity,
        "config_hash": context.config.config_hash,
        "code_hash": context.code_hash,
        "splits": splits,
        "core_records": contract.signed_record_count,
        "outer_evaluation": "manual only, once after every ablation in fold "
        "is frozen and retrained",
        "prior_exposure": "Module 4 developed on Fold 2/3 speakers; nested results "
        "are not pristine "
        "external validation. Fold 1 was reserved from Module 4 evaluation.",
        "fair_ablation": "Each outer fold's inner-selected C2 hyperparameters apply to all seven "
        "architectures; each has its own inner-selected epoch count.",
        "module5_signed_off": False,
        "module6_started": False,
    }


def freeze_protocol(context: ExperimentContext) -> dict[str, Any]:
    path = context.config.output("protocol.json")
    expected = protocol_document(context)
    if path.exists():
        if read_json(path) != expected:
            raise ValueError("frozen protocol/code changed; preserve existing experiment evidence")
    else:
        write_json(path, expected, immutable=True)
    return expected


def require_frozen(context: ExperimentContext) -> None:
    if read_json(context.config.output("protocol.json")) != protocol_document(context):
        raise ValueError("protocol/code/splits differ from frozen experiment")
