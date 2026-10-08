from __future__ import annotations

import hashlib
import io
from typing import Any

import pytest
import torch

from vaaniswitch.m5_experiments import trainer
from vaaniswitch.m5_experiments.checkpoint import load_checkpoint
from vaaniswitch.m5_experiments.config import CANDIDATES
from vaaniswitch.m5_experiments.data import ExperimentDataset
from vaaniswitch.m5_experiments.ledger import Ledger, experiment_id
from vaaniswitch.m5_experiments.pipeline import run_inner
from vaaniswitch.m5_experiments.reporting import write_json
from vaaniswitch.m5_experiments.source import ExperimentContext, freeze_protocol


def constant_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    original = trainer.run_epoch

    def run(model: Any, dataset: ExperimentDataset, **kwargs: Any) -> dict[str, Any]:
        result = original(model, dataset, **kwargs)
        if dataset.role == "inner_validation":
            result.update(macro_f1=0.8, balanced_accuracy=0.8, roc_auc=0.9, loss=0.4)
        return result

    monkeypatch.setattr(trainer, "run_epoch", run)


def test_inner_stopping_final_fresh_all_development_and_exact_epochs(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    constant_validation(monkeypatch)
    split, candidate = context.split(1), CANDIDATES[0]
    inner = trainer.train_stage(
        context, split, "B1", candidate, "fixture", smoke=False, resume=False
    )
    assert inner["progress"]["epoch"] == 6
    assert inner["progress"]["best_inner"]["epoch"] == 1
    selection = inner["progress"]["best_inner"]
    final = trainer.train_stage(
        context, split, "B1", candidate, "fixture", smoke=False, resume=False, selection=selection
    )
    assert final["progress"]["epoch"] == 1
    assert final["progress"]["history"][0]["training"]["record_count"] == 10
    assert final["progress"]["history"][0]["training"]["role"] == "development"
    assert "validation" not in final["progress"]["history"][0]
    assert final["identity"]["all_development_speakers"] == list(split.speakers("development"))
    assert final["identity"]["selected_inner_configuration"] == selection
    records = split.records(context.predecessor.source.contract, "development")
    positive = sum(r.condition.value == "dysarthric" for r in records)
    assert final["progress"]["pos_weight"] == (10 - positive) / positive
    # Train another new model without consulting inner weights; final states must be exact.
    independent = trainer.train_stage(
        context,
        split,
        "B1",
        candidate,
        "fresh_reference",
        smoke=False,
        resume=False,
        selection=selection,
    )

    def state(run_id: str, report: dict[str, Any]) -> dict[str, Any]:
        path = context.config.output(
            f"experiments/runs/{run_id}/final/{report['checkpoint']['filename']}"
        )
        return dict(torch.load(path, weights_only=True)["model"])

    left, right = state("fixture", final), state("fresh_reference", independent)
    assert all(torch.equal(t, right[n]) for n, t in left.items())


def test_interrupted_resume_matches_uninterrupted(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    constant_validation(monkeypatch)
    split, candidate = context.split(1), CANDIDATES[0]
    full = trainer.train_stage(context, split, "B1", candidate, "full", smoke=False, resume=False)
    original = trainer.run_epoch

    def interrupt(model: Any, dataset: ExperimentDataset, **kwargs: Any) -> dict[str, Any]:
        if kwargs["epoch"] == 2 and dataset.role == "inner_train":
            raise RuntimeError("simulated process interruption")
        return original(model, dataset, **kwargs)

    monkeypatch.setattr(trainer, "run_epoch", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        trainer.train_stage(
            context, split, "B1", candidate, "interrupted", smoke=False, resume=False
        )
    monkeypatch.setattr(trainer, "run_epoch", original)
    resumed = trainer.train_stage(
        context, split, "B1", candidate, "interrupted", smoke=False, resume=True
    )
    assert resumed["progress"]["epoch"] == full["progress"]["epoch"] == 6
    for group in ("model", "optimizer"):
        saved = []
        for run_id, report in (("full", full), ("interrupted", resumed)):
            path = context.config.output(
                f"experiments/runs/{run_id}/inner/{report['checkpoint']['filename']}"
            )
            saved.append(torch.load(path, weights_only=True)[group])
        if group == "model":
            assert all(torch.equal(t, saved[1][n]) for n, t in saved[0].items())
        else:
            assert saved[0]["param_groups"] == saved[1]["param_groups"]
            assert all(
                torch.equal(t, saved[1]["state"][i][name])
                for i, state in saved[0]["state"].items()
                for name, t in state.items()
            )
    with pytest.raises(ValueError, match="explicit resume"):
        trainer.train_stage(
            context, split, "B1", candidate, "interrupted", smoke=False, resume=False
        )


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "pickle",
        "identity",
        "tensor",
        "optimizer",
        "epoch",
        "extra_field",
        "rng",
        "history",
    ],
)
def test_checkpoint_corruption_rejected(context: ExperimentContext, damage: str) -> None:
    candidate = CANDIDATES[0]
    report = trainer.train_stage(
        context, context.split(1), "B1", candidate, "checkpoint", smoke=True, resume=False
    )
    directory = context.config.output("smoke/runs/checkpoint/inner")
    pointer = report["checkpoint"]
    path = directory / pointer["filename"]
    if damage == "checksum":
        path.write_bytes(b"corrupt")
    else:
        payload = torch.load(path, weights_only=True)
        if damage == "identity":
            payload["identity"]["seed"] = 7
        elif damage == "tensor":
            next(iter(payload["model"].values())).fill_(float("nan"))
        elif damage == "optimizer":
            payload["optimizer"]["state"][100000] = {"hidden_array": torch.ones(7)}
        elif damage == "epoch":
            payload["progress"]["epoch"] = 99
        elif damage == "extra_field":
            payload["representations"] = torch.ones(224, 224)
        elif damage == "rng":
            payload["rng"] = {}
        elif damage == "history":
            payload["progress"]["history"] = []
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        content = b"invalid restricted pickle" if damage == "pickle" else buffer.getvalue()
        checksum = hashlib.sha256(content).hexdigest()
        (directory / f"{checksum}.pt").write_bytes(content)
        write_json(directory / "latest.json", {"filename": f"{checksum}.pt", "sha256": checksum})
    model = trainer.build_model(context, "B1", candidate, load_pretrained=False)
    optimizer = trainer.make_optimizer(model, candidate)
    with pytest.raises(ValueError, match="checkpoint rejected"):
        load_checkpoint(directory, model, optimizer, report["identity"])


def test_checkpoint_required_provenance(context: ExperimentContext) -> None:
    report = trainer.train_stage(
        context, context.split(1), "B1", CANDIDATES[0], "fields", smoke=True, resume=False
    )
    identity = report["identity"]
    assert {
        "experiment_id",
        "ablation",
        "outer_fold",
        "split_hash",
        "seed",
        "config_hash",
        "code_hash",
        "signed_hashes",
        "selection_status",
        "selected_inner_configuration",
        "all_development_speakers",
        "outer_test_speakers",
    } <= set(identity)
    directory = context.config.output("smoke/runs/fields/inner")
    payload = torch.load(directory / report["checkpoint"]["filename"], weights_only=True)
    assert {"model", "optimizer", "rng", "progress", "identity"} <= set(payload)
    assert payload["progress"]["best_inner"]["epoch"] == 1
    assert len(list(directory.glob("*.pt"))) == 1


def test_failed_inner_requires_explicit_retry(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_protocol(context)
    ledger = Ledger(context.config)
    candidate = CANDIDATES[0]
    run_id = experiment_id(context.config, 1, "B1", candidate.id)
    original = trainer.run_epoch

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(trainer, "run_epoch", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        run_inner(context, ledger, 1, "B1", candidate)
    assert ledger.get(run_id)["state"] == "FAILED"
    with pytest.raises(ValueError, match="disposition"):
        run_inner(context, ledger, 1, "B1", candidate, resume=True)
    monkeypatch.setattr(trainer, "run_epoch", original)
    constant_validation(monkeypatch)
    recovered = run_inner(
        context,
        ledger,
        1,
        "B1",
        candidate,
        resume=True,
        retry_reason="transient synthetic failure cleared",
    )
    assert recovered.experiment_id == run_id
    assert run_inner(context, ledger, 1, "B1", candidate) == recovered
    assert ledger.get(run_id)["state"] == "INNER_COMPLETE"


def test_early_stopping_rejects_outer_metrics(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = trainer.run_epoch

    def forged(model: Any, dataset: ExperimentDataset, **kwargs: Any) -> dict[str, Any]:
        result = original(model, dataset, **kwargs)
        if dataset.role == "inner_validation":
            result["role"] = "outer_test"
        return result

    monkeypatch.setattr(trainer, "run_epoch", forged)
    with pytest.raises(ValueError, match="inner-validation evidence"):
        trainer.train_stage(
            context, context.split(1), "B1", CANDIDATES[0], "forged", smoke=True, resume=False
        )
