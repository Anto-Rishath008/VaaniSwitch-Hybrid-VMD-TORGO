"""CLI integration, checkpoint integrity, resume determinism, and test-fold sentinels."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tests.m4_fusion.helpers import (
    MemorySource,
    install_memory_source,
    make_context,
    shorter,
    write_fusion_config,
)
from vaaniswitch.m0_data.reporting import snapshot_raw_tree
from vaaniswitch.m4_fusion import pipeline
from vaaniswitch.m4_fusion.checkpoint import checkpoint_directory, load_checkpoint
from vaaniswitch.m4_fusion.cli import main
from vaaniswitch.m4_fusion.diagnostics import run_smoke
from vaaniswitch.m4_fusion.model import DysarthriaClassifier
from vaaniswitch.m4_fusion.pipeline import run_training, selection_identity, validate_checkpoint
from vaaniswitch.m4_fusion.reproducibility import capture_rng, restore_rng, seed_everything
from vaaniswitch.m4_fusion.source import DevelopmentContext
from vaaniswitch.m4_fusion.trainer import make_optimizer, run_epoch


@pytest.fixture
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DevelopmentContext, Path]:
    context = make_context(tmp_path)
    install_memory_source(monkeypatch, context)
    result = run_training(context.config, outer_test_fold=1, smoke=True)
    return context, Path(result["best_checkpoint"])


def resign(path: Path, document: dict[str, Any]) -> None:
    torch.save(document, path)
    path.with_suffix(".pt.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_checkpoint_roundtrip_identity_optimizer_and_no_dataset_arrays(
    trained: tuple[DevelopmentContext, Path],
) -> None:
    context, path = trained
    document = torch.load(path, weights_only=True)
    assert document["split"] == context.split.identity
    assert document["hashes"] == context.config.hashes
    assert document["seed"] == 42
    assert document["optimizer_state"]["state"]
    assert document["selection"] == selection_identity(context, smoke=True)
    assert not set(document) & {"data", "dataset", "mel", "recurrence", "graph_nodes", "tokens"}
    assert all(not key.endswith(".fc.weight") for key in document["model_state"])
    model = DysarthriaClassifier(context.encoder_config, load_pretrained=False)
    optimizer = make_optimizer(model, context)
    loaded = load_checkpoint(
        path,
        model,
        optimizer,
        context.config,
        context.split,
        selection=selection_identity(context, smoke=True),
        smoke=True,
    )
    assert loaded["best_epoch"] == 1
    assert all(
        torch.equal(value, model.state_dict()[key])
        for key, value in document["model_state"].items()
    )
    validated = validate_checkpoint(context.config, path, outer_test_fold=1, smoke=True)
    assert validated["saved_validation_metrics_match"] is True
    assert validated["outer_test_forward_count"] == 0
    assert isinstance(context.source, MemorySource)
    assert context.source.loaded
    assert all(
        logical.split("/")[0] not in context.split.test_speakers
        for logical in context.source.loaded
    )


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "hashes",
        "split",
        "model_nan",
        "model_extra",
        "optimizer_nan",
        "epoch",
        "rng",
        "dataset_array",
        "mode",
        "pos_weight",
        "history",
    ],
)
def test_corrupted_checkpoint_rejected(
    trained: tuple[DevelopmentContext, Path],
    damage: str,
) -> None:
    context, path = trained
    document = torch.load(path, weights_only=True)
    if damage == "checksum":
        path.write_bytes(path.read_bytes() + b"corruption")
    else:
        if damage == "hashes":
            document["hashes"]["encoder_config_hash"] = "f" * 64
        elif damage == "split":
            document["split"]["train_speakers"].append(context.split.test_speakers[0])
        elif damage == "model_nan":
            document["model_state"]["fusion.classifier.0.weight"].fill_(float("nan"))
        elif damage == "model_extra":
            document["model_state"]["dataset.mel"] = torch.zeros(1, 224, 224)
        elif damage == "optimizer_nan":
            next(iter(document["optimizer_state"]["state"].values()))["exp_avg"].fill_(float("nan"))
        elif damage == "epoch":
            document["epoch"] = 50
        elif damage == "rng":
            document["rng"] = {}
        elif damage == "dataset_array":
            document["dataset"] = torch.zeros(1, 224, 224)
        elif damage == "mode":
            document["mode"] = "baseline"
        elif damage == "pos_weight":
            document["pos_weight"] = float("nan")
        elif damage == "history":
            document["history"] = []
        resign(path, document)
    model = DysarthriaClassifier(context.encoder_config, load_pretrained=False)
    with pytest.raises(ValueError, match="checkpoint rejected"):
        load_checkpoint(
            path,
            model,
            make_optimizer(model, context),
            context.config,
            context.split,
            selection=selection_identity(context, smoke=True),
            smoke=True,
        )


def test_rng_roundtrip_and_seed_repeat() -> None:
    seed_everything(42, 2)
    saved = capture_rng()
    expected = random.random(), np.random.random(), torch.rand(3)
    restore_rng(saved)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    seed_everything(42, 2)
    assert random.random() == expected[0]
    with pytest.raises(ValueError):
        seed_everything(43, 2)


@pytest.mark.parametrize(
    "protected",
    ["data/raw/torgo/extracted", "data/processed/vmd", "data/processed/representations"],
)
def test_training_preserves_protected_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected: str,
) -> None:
    context = make_context(tmp_path)
    install_memory_source(monkeypatch, context)
    root = tmp_path / protected
    root.mkdir(parents=True, exist_ok=True)
    (root / "signed-input.bin").write_bytes(b"immutable source sentinel")
    before = snapshot_raw_tree(root)
    run_training(context.config, outer_test_fold=1, smoke=True)
    assert snapshot_raw_tree(root) == before
    assert (root / "signed-input.bin").read_bytes() == b"immutable source sentinel"


def test_cli_split_dry_run_smoke_train_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context(tmp_path)
    install_memory_source(monkeypatch, context)
    path = write_fusion_config(context.config)
    common = ["--config", str(path), "--outer-test-fold", "1", "--seed", "42"]
    assert main(["split-audit", *common]) == 0
    assert main(["train", *common, "--dry-run"]) == 0
    assert isinstance(context.source, MemorySource)
    assert context.source.loaded == []
    assert main(["train", *common, "--smoke"]) == 0
    checkpoint = checkpoint_directory(context.config, context.split, smoke=True) / "best.pt"
    assert main(["validate", *common, "--checkpoint", str(checkpoint), "--smoke"]) == 0
    assert main(["validate", *common, "--checkpoint", str(checkpoint)]) == 1
    # Existing runs cannot be silently overwritten.
    assert main(["train", *common, "--smoke"]) == 1
    assert main(["train", *common, "--smoke", "--resume", str(checkpoint)]) == 1
    assert (
        main(["train", *common, "--smoke", "--resume", str(checkpoint.with_name("last.pt"))]) == 0
    )
    assert (tmp_path / "artifacts/logs/module4_fusion_failures.csv").is_file()


def test_cli_smoke_diagnostics_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = make_context(tmp_path)
    install_memory_source(monkeypatch, context)
    path = write_fusion_config(context.config)
    assert main(["smoke", "--config", str(path), "--outer-test-fold", "1"]) == 0


def test_smoke_reports_shapes_freeze_roundtrip_and_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context(tmp_path)
    install_memory_source(monkeypatch, context)
    result = run_smoke(context.config, outer_test_fold=1)
    assert result["shapes"]["attention_weights"] == [4, 4, 16, 17]
    assert result["checkpoint_roundtrip_exact"]
    assert result["frozen_backbone_weights_and_buffers_unchanged"]
    assert result["manual_runtime_ESTIMATE"]["maximum_run_hours"] > 0
    assert result["long_baseline_training_run"] is False


def test_resume_matches_uninterrupted_training_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = shorter(make_context(tmp_path / "full"))
    interrupted = shorter(make_context(tmp_path / "interrupted"))
    contexts = {c.config.repo_root: c for c in (full, interrupted)}

    def prepare(config: Any, fold: int) -> DevelopmentContext:
        return contexts[config.repo_root]

    def offline_model(
        context: DevelopmentContext, *, load_pretrained: bool
    ) -> DysarthriaClassifier:
        model = DysarthriaClassifier(context.encoder_config, load_pretrained=False)
        # Synthetic test only: mimic the production initialization identity without a download.
        model.encoders.mel_encoder.pretrained_weights_identifier = "ResNet18_Weights.IMAGENET1K_V1"
        return model

    monkeypatch.setattr(pipeline, "prepare_context", prepare)
    monkeypatch.setattr(pipeline, "build_model", offline_model)
    complete = run_training(full.config, outer_test_fold=1)
    original_epoch = run_epoch

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("epoch") == 2:
            raise RuntimeError("simulated interruption")
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(pipeline, "run_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_training(interrupted.config, outer_test_fold=1)
    monkeypatch.setattr(pipeline, "run_epoch", original_epoch)
    last = checkpoint_directory(interrupted.config, interrupted.split, smoke=False) / "last.pt"
    resumed = run_training(interrupted.config, outer_test_fold=1, resume=last)
    expected = torch.load(complete["last_checkpoint"], weights_only=True)
    actual = torch.load(resumed["last_checkpoint"], weights_only=True)
    assert expected["epoch"] == actual["epoch"] == 2
    assert all(
        torch.equal(value, actual["model_state"][key])
        for key, value in expected["model_state"].items()
    )
    assert (
        expected["history"][-1]["validation"]["macro_f1"]
        == actual["history"][-1]["validation"]["macro_f1"]
    )


def test_early_stopping_uses_validation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = shorter(make_context(tmp_path), epochs=5, patience=1)
    monkeypatch.setattr(pipeline, "prepare_context", lambda *_: context)

    def model_factory(
        current: DevelopmentContext, *, load_pretrained: bool
    ) -> DysarthriaClassifier:
        model = DysarthriaClassifier(current.encoder_config, load_pretrained=False)
        model.encoders.mel_encoder.pretrained_weights_identifier = "ResNet18_Weights.IMAGENET1K_V1"
        return model

    monkeypatch.setattr(pipeline, "build_model", model_factory)
    original = run_epoch

    def constant_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        if args[2] == "validation":
            result["macro_f1"] = 0.25
        return result

    monkeypatch.setattr(pipeline, "run_epoch", constant_validation)
    result = run_training(context.config, outer_test_fold=1)
    assert result["early_stopped"] and result["epochs_completed"] == 2
    assert result["best_epoch"] == 1
    assert result["outer_test_forward_count"] == 0
    assert sorted(p.name for p in Path(result["best_checkpoint"]).parent.glob("*.pt")) == [
        "best.pt",
        "last.pt",
    ]
