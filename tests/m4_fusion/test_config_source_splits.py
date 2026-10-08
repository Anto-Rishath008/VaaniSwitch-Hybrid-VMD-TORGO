"""Config, signed-source, deterministic splitting, and pre-forward leakage rejection."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tests.m4_fusion.helpers import (
    REPO,
    install_memory_source,
    make_context,
    mutate_yaml,
    write_fusion_config,
)
from vaaniswitch.m4_fusion.config import FusionConfigError, Module4Config
from vaaniswitch.m4_fusion.losses import label_for_condition, training_pos_weight
from vaaniswitch.m4_fusion.reporting import read_json, write_json
from vaaniswitch.m4_fusion.source import load_encoder_contract, prepare_context
from vaaniswitch.m4_fusion.splits import (
    Partition,
    create_inner_split,
    guard_batch,
    partition_records,
    split_audit,
)
from vaaniswitch.m4_fusion.trainer import batches


def test_signed_production_config_and_architecture_accepted() -> None:
    config = Module4Config.from_yaml(REPO / "configs/models/fusion.yaml")
    encoder = load_encoder_contract(config)
    assert encoder.config_hash == "7c8e44eddb11704d08e45a1b63bc9cc54b28d679f5e3a3ae1a474b61e3a40383"
    assert encoder.token_dimension == 128


@pytest.mark.parametrize(
    "key",
    [
        "encoder_config_hash",
        "representation_config_hash",
        "module1_config_hash",
        "manifest_fingerprint",
    ],
)
def test_wrong_signed_hash_rejected(tmp_path: Path, key: str) -> None:
    context = make_context(tmp_path)
    overrides: dict[str, Any] = {key: "f" * 64}
    config = replace(context.config, source=replace(context.config.source, **overrides))
    with pytest.raises(ValueError, match=r"hash|fingerprint"):
        load_encoder_contract(config)


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("token_dim",), 192),
        (("attention", "query_tokens"), 1),
        (("attention", "key_value_tokens"), 16),
        (("attention", "heads"), 8),
        (("attention", "query"), "nonlinear_tokens"),
        (("attention", "dropout"), 0.2),
        (("attention", "residual"), False),
        (("classifier", "output_dim"), 2),
        (("pooling", "mel"), "flatten"),
        (("training", "threshold"), 0.4),
        (("training", "batch_size"), 0),
        (("training", "max_epochs"), 31),
        (("training", "learning_rate"), float("nan")),
        (("training", "seed"), 43),
        (("training", "freeze_mel_backbone"), False),
        (("training", "resampling"), "weighted"),
        (("source", "encoder_config_path"), "../encoders.yaml"),
        (("source", "encoder_config_hash"), "bad"),
        (("source", "core_record_count"), False),
        (("execution", "cpu_threads"), 0),
    ],
)
def test_invalid_configuration_rejected(tmp_path: Path, keys: tuple[str, ...], value: Any) -> None:
    path = write_fusion_config(make_context(tmp_path).config)
    mutate_yaml(path, keys, value)
    with pytest.raises((FusionConfigError, ValueError)):
        Module4Config.from_yaml(path)


def test_hash_deterministic_semantic_and_not_path_bound(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    path = write_fusion_config(context.config)
    loaded = Module4Config.from_yaml(path)
    assert loaded.config_hash == context.config.config_hash
    raw = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert Module4Config.from_yaml(path).config_hash == loaded.config_hash
    assert replace(loaded, repo_root=tmp_path / "elsewhere").config_hash == loaded.config_hash
    assert loaded.with_batch_size(8).config_hash != loaded.config_hash
    assert replace(loaded, training=replace(loaded.training, learning_rate=1e-4)).config_hash != (
        loaded.config_hash
    )


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("metrics", "module3_summary.json"),
        ("data", "module4.json"),
        ("metrics", "module4/../../raw.json"),
    ],
)
def test_output_paths_protect_prior_modules(tmp_path: Path, kind: str, name: str) -> None:
    with pytest.raises(ValueError):
        make_context(tmp_path).config.output(kind, name)


@pytest.mark.parametrize("field", ["status", "module3_signed_off", "configuration", "source"])
def test_tampered_signoff_rejected(tmp_path: Path, field: str) -> None:
    context = make_context(tmp_path)
    document = read_json(context.config.source.module3_signoff_path)
    document[field] = {} if field in {"configuration", "source"} else False
    write_json(context.config.source.module3_signoff_path, document)
    with pytest.raises(ValueError):
        load_encoder_contract(context.config)


@pytest.mark.parametrize("fold", [1, 2, 3])
def test_all_outer_folds_deterministic_disjoint_both_classes(tmp_path: Path, fold: int) -> None:
    contract = make_context(tmp_path).source.contract
    split = create_inner_split(contract, outer_test_fold=fold, seed=42, folds_fingerprint="a" * 64)
    repeat = create_inner_split(
        replace(contract, records=tuple(reversed(contract.records))),
        outer_test_fold=fold,
        seed=42,
        folds_fingerprint="a" * 64,
    )
    assert split == repeat
    assert split.split_hash == repeat.split_hash
    train, validation, test = map(
        set, (split.train_speakers, split.validation_speakers, split.test_speakers)
    )
    assert not train & validation and not train & test and not validation & test
    assert len(validation) == 2
    assert train | validation | test == set(contract.speaker_to_evaluation_fold)
    for part in ("train", "validation"):
        records = partition_records(contract, split, part)
        assert {r.condition.value for r in records} == {"control", "dysarthric"}
        assert all(contract.speaker_to_evaluation_fold[r.speaker_id] != fold for r in records)


def test_fold_map_tamper_rejected_before_dataset_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = make_context(tmp_path)
    install_memory_source(monkeypatch, context)
    path = context.encoder_config.source.speaker_folds_path
    document = read_json(path)
    document["folds"][0]["evaluation_speakers"].append(context.split.train_speakers[0])
    write_json(path, document)
    with pytest.raises(RuntimeError):
        prepare_context(context.config, 1)


def test_split_audit_and_no_test_partition(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    audit = split_audit(context.source.contract, context.split)
    assert audit["speaker_overlap_count"] == 0
    assert audit["outer_test_forward_count"] == 0
    with pytest.raises(ValueError, match="only train and validation"):
        partition_records(context.source.contract, context.split, cast(Partition, "test"))
    with pytest.raises(ValueError):
        create_inner_split(
            context.source.contract, outer_test_fold=9, seed=42, folds_fingerprint="x"
        )


def test_batch_guard_rejects_test_or_mislabeled_record(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    batch = next(batches(context, "train", smoke=True))
    guard_batch(batch, context.source.contract, context.split, "train")
    invalid = replace(batch.provenance[0], speaker_id=context.split.test_speakers[0])
    with pytest.raises(ValueError, match="forward forbidden"):
        guard_batch(
            replace(batch, provenance=(invalid, *batch.provenance[1:])),
            context.source.contract,
            context.split,
            "train",
        )
    with pytest.raises(ValueError, match="forward forbidden"):
        guard_batch(batch, context.source.contract, context.split, "validation")


def test_label_mapping_and_train_only_weight(tmp_path: Path) -> None:
    assert label_for_condition("control") == 0
    assert label_for_condition("dysarthric") == 1
    with pytest.raises(ValueError):
        label_for_condition("unknown")
    context = make_context(tmp_path)
    train = partition_records(context.source.contract, context.split, "train")
    val = partition_records(context.source.contract, context.split, "validation")
    positive = sum(r.condition.value == "dysarthric" for r in train)
    assert (
        training_pos_weight(train, context.split.train_speakers)
        == (len(train) - positive) / positive
    )
    with pytest.raises(ValueError, match="only nonempty training"):
        training_pos_weight((*train, *val), context.split.train_speakers)
    controls = tuple(r for r in train if r.condition.value == "control")
    with pytest.raises(ValueError, match="both classes"):
        training_pos_weight(controls, context.split.train_speakers)
