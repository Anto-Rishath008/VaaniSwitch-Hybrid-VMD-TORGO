from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from vaaniswitch.m5_experiments import source
from vaaniswitch.m5_experiments.cli import main
from vaaniswitch.m5_experiments.config import SIGNED_HASHES, ExperimentConfig, canonical_path
from vaaniswitch.m5_experiments.data import ExperimentDataset, batches, pos_weight, selected_records
from vaaniswitch.m5_experiments.reporting import read_json
from vaaniswitch.m5_experiments.source import ExperimentContext, freeze_protocol, require_frozen
from vaaniswitch.m5_experiments.splits import nested_splits


def test_signed_identities_and_exact_outer_folds(real_context: ExperimentContext) -> None:
    assert real_context.predecessor.config.hashes == SIGNED_HASHES
    assert real_context.predecessor.source.contract.signed_record_count == 8673
    assert [s.speakers("outer_test") for s in real_context.splits] == [
        ("F03", "FC01", "M04", "MC03", "MC04"),
        ("F04", "FC02", "M01", "M02", "MC02"),
        ("F01", "FC03", "M03", "M05", "MC01"),
    ]
    assert [s.speakers("inner_validation") for s in real_context.splits] == [
        ("M03", "MC02"),
        ("F03", "FC03"),
        ("F03", "MC02"),
    ]


@pytest.mark.parametrize("key", list(SIGNED_HASHES))
def test_wrong_predecessor_hash_rejected(
    real_context: ExperimentContext, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    original = read_json

    def tamper(path: Path) -> dict[str, Any]:
        value = original(path)
        if path.name == "module4_full_signoff_audit.json":
            value["hashes"][key] = "0" * 64
        return value

    monkeypatch.setattr(source, "read_json", tamper)
    with pytest.raises(ValueError, match="Module 4 must be signed"):
        source.prepare_experiments(real_context.config)


@pytest.mark.parametrize("fold", [1, 2, 3])
def test_nested_disjoint_deterministic_classes(context: ExperimentContext, fold: int) -> None:
    split = context.split(fold)
    contract = context.predecessor.source.contract
    reversed_contract = replace(contract, records=tuple(reversed(contract.records)))
    repeated = nested_splits(reversed_contract, split.inner.folds_fingerprint)[fold - 1]
    assert split == repeated
    train, validation, outer = (
        set(split.speakers(r)) for r in ("inner_train", "inner_validation", "outer_test")
    )
    assert not train & validation and not train & outer and not validation & outer
    assert len(train) == 8 and len(validation) == 2 and len(outer) == 5
    for role in ("inner_train", "inner_validation", "development"):
        records = split.records(contract, role)
        assert {r.condition.value for r in records} == {"control", "dysarthric"}
        assert not {r.speaker_id for r in records} & outer


@pytest.mark.parametrize("fold", [1, 2, 3])
@pytest.mark.parametrize("role", ["inner_train", "development"])
def test_fitting_only_weights(context: ExperimentContext, fold: int, role: Any) -> None:
    split = context.split(fold)
    records = split.records(context.predecessor.source.contract, role)
    positive = sum(r.condition.value == "dysarthric" for r in records)
    assert pos_weight(context, split, role, smoke=False) == (len(records) - positive) / positive
    assert len(split.speakers("development")) == 10


@pytest.mark.parametrize("role", ["inner_validation", "outer_test"])
def test_validation_outer_weighting_blocked(context: ExperimentContext, role: Any) -> None:
    with pytest.raises(ValueError, match="fitting roles"):
        pos_weight(context, context.split(1), role, smoke=False)


def test_outer_dataset_blocked_before_selection_and_retrain(context: ExperimentContext) -> None:
    with pytest.raises(ValueError, match="outer dataset blocked"):
        ExperimentDataset(context, context.split(1), "outer_test", "C2")
    dataset = ExperimentDataset(context, context.split(1), "inner_train", "B1")
    batch = next(batches(dataset))
    outer = context.split(1).records(context.predecessor.source.contract, "outer_test")[0]
    forged = replace(batch.provenance[0], logical_id=outer.logical_id, speaker_id=outer.speaker_id)
    with pytest.raises(ValueError, match="violates"):
        dataset.guard(replace(batch, provenance=(forged, *batch.provenance[1:])))


def test_smoke_never_loads_fold1(context: ExperimentContext) -> None:
    with pytest.raises(ValueError, match="restricted"):
        selected_records(context, context.split(2), "inner_train", smoke=True)
    with pytest.raises(ValueError, match="restricted"):
        selected_records(context, context.split(1), "outer_test", smoke=True)
    panel = selected_records(context, context.split(1), "inner_train", smoke=True)
    assert len(panel) == 4
    assert not {r.speaker_id for r in panel} & set(context.split(1).speakers("outer_test"))


def test_protocol_freeze_and_code_drift(context: ExperimentContext) -> None:
    first = freeze_protocol(context)
    assert freeze_protocol(context) == first
    require_frozen(context)
    (context.config.repo_root / "src/vaaniswitch/m1_vmd/audio.py").write_text("changed\n")
    with pytest.raises(ValueError, match="differ"):
        require_frozen(context)
    with pytest.raises(ValueError, match="changed"):
        freeze_protocol(context)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 7),
        ("ablations", ["C3"]),
        ("candidates", []),
        ("training", {"threshold": 0.7}),
        ("extra", True),
    ],
)
def test_protocol_settings_locked(context: ExperimentContext, field: str, value: Any) -> None:
    path = context.config.repo_root / "configs/experiments/module5/protocol.yaml"
    raw = yaml.safe_load(path.read_text())
    raw[field] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(path, repo_root=context.config.repo_root)


def test_namespace_cannot_redirect_or_escape(context: ExperimentContext) -> None:
    with pytest.raises(ValueError):
        canonical_path(context.config.repo_root, "artifacts/module5/../../data/raw/overwrite")
    assert context.config.output("reports/report.json").is_relative_to(
        context.config.repo_root / "artifacts/module5"
    )


def test_cli_dry_run_does_not_instantiate_model(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(context.config.repo_root)
    monkeypatch.setattr(
        "vaaniswitch.m5_experiments.cli.prepare_experiments", lambda config: context
    )
    assert main(["dry-run"]) == 0
    assert '"module5_signed_off": false' in capsys.readouterr().out
    assert not context.config.artifact_root.exists()
    assert main(["protocol", "--freeze"]) == 0
    assert read_json(context.config.output("protocol.json"))["core_records"] == 15
