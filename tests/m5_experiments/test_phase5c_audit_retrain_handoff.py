"""Read-only audit checks and mocked final-training control flow; no model forwards."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.audit_module5_phase5c import COMMON, summaries, validate_bundle, validate_rng
from tests.m5_experiments.test_workflow_outer_robustness import fake_training
from vaaniswitch.m4_fusion.reproducibility import capture_rng
from vaaniswitch.m5_experiments import pipeline, trainer
from vaaniswitch.m5_experiments.checkpoint import load_checkpoint
from vaaniswitch.m5_experiments.config import ABLATIONS, CANDIDATES, file_hash
from vaaniswitch.m5_experiments.data import pos_weight
from vaaniswitch.m5_experiments.ledger import Ledger, experiment_id
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.source import ExperimentContext, freeze_protocol
from vaaniswitch.m5_experiments.splits import record_hash


@pytest.mark.parametrize("damage", [None, "python", "numpy_keys", "numpy_position", "torch"])
def test_rng_validation_uses_isolated_generators(damage: str | None) -> None:
    before = capture_rng()
    state = deepcopy(before)
    if damage == "python":
        state["python"] = (0, (), None)
    elif damage == "numpy_keys":
        state["numpy"][1][0] = -1
    elif damage == "numpy_position":
        state["numpy"][2] = 10000
    elif damage == "torch":
        state["torch"] = torch.zeros(2)
    if damage:
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            validate_rng(state)
    else:
        validate_rng(state)
    after = capture_rng()
    assert before["python"] == after["python"] and before["numpy"] == after["numpy"]
    assert torch.equal(before["torch"], after["torch"])


@pytest.mark.parametrize("damage", [None, "common_candidate", "missing", "c2", "epoch", "identity"])
def test_bundle_is_bound_to_c2_and_selected_ledger(
    context: ExperimentContext, damage: str | None
) -> None:
    selected = {
        a: {
            "experiment_id": experiment_id(context.config, 1, a, COMMON[1]),
            "candidate_id": COMMON[1],
            "epoch": i + 1,
        }
        for i, a in enumerate(ABLATIONS)
    }
    runs = {
        s["experiment_id"]: {
            "state": "SELECTED",
            "data": {"selection": deepcopy(s), "inner_result": deepcopy(s)},
        }
        for s in selected.values()
    }
    bundle = {
        "outer_fold": 1,
        "config_hash": context.config.config_hash,
        "common_candidate": COMMON[1],
        "selections": selected,
    }
    c2 = deepcopy(selected["C2"])
    if damage == "common_candidate":
        bundle["common_candidate"] = "baseline"
    elif damage == "missing":
        del selected["B2"]
    elif damage == "c2":
        selected["C2"]["epoch"] = 22
    elif damage == "epoch":
        selected["A0"]["epoch"] = 22
    elif damage == "identity":
        selected["A0"]["experiment_id"] = selected["A1"]["experiment_id"]
    if damage:
        with pytest.raises(ValueError):
            validate_bundle(context, 1, bundle, runs, c2)
    else:
        validate_bundle(context, 1, bundle, runs, c2)


def test_descriptive_statistics_use_sample_sd_and_directional_differences() -> None:
    bundles = [
        {
            "outer_fold": f,
            "selections": {
                a: {
                    "metrics": dict.fromkeys(
                        ("macro_f1", "balanced_accuracy", "roc_auc"), 0.1 * f + 0.01 * i
                    )
                }
                for i, a in enumerate(ABLATIONS)
            },
        }
        for f in (1, 2, 3)
    ]
    report = summaries(bundles)
    assert report["summary"]["A0"]["macro_f1"]["mean"] == pytest.approx(0.2)
    assert report["summary"]["A0"]["macro_f1"]["sample_sd"] == pytest.approx(0.1)
    assert report["differences"]["A1-A0"]["macro_f1"] == pytest.approx([0.01] * 3)
    assert report["differences"]["B3-B1"]["macro_f1"] == pytest.approx([0.02] * 3)
    assert report["macro_f1_fold_winners"] == ["C2"] * 3


@pytest.mark.parametrize("interrupt", [False, True])
def test_final_stage_fresh_then_final_only_resume_no_validation(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    split, candidate = context.split(1), CANDIDATES[0]
    selection = {"epoch": 2}
    run_id = "mocked_final_stage"
    directory = context.config.output(f"experiments/runs/{run_id}/final")
    inner = directory.parent / "inner/latest.json"
    write_json(inner, {"must_never_be_read": True})
    flags, epochs, loaded, restored = [], [], [], []
    saved: dict[str, Any] = {}
    interrupted = False
    model = SimpleNamespace(parameter_counts=lambda: {"total": 1, "trainable": 1, "classifier": 1})
    optimizer = object()

    def build(*args: Any, load_pretrained: bool, **kwargs: Any) -> Any:
        flags.append(load_pretrained)
        return model

    def dataset(ctx: Any, assigned: Any, role: str, *args: Any, **kwargs: Any) -> Any:
        assert role == "development"
        assert len(assigned.speakers(role)) == 10
        return SimpleNamespace(
            role=role, records=assigned.records(ctx.predecessor.source.contract, role)
        )

    def epoch(m: Any, data: Any, *, weight: float, epoch: int, **kwargs: Any) -> dict[str, Any]:
        nonlocal interrupted
        assert m is model and kwargs["optimizer"] is optimizer
        assert data.role == "development"
        assert weight == pos_weight(context, split, "development", smoke=False)
        if interrupt and epoch == 2 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("synthetic unsaved epoch")
        epochs.append(epoch)
        return {
            "role": data.role,
            "record_count": len(data.records),
            "record_hash": record_hash(data.records),
            "loss": 0.5,
            "elapsed_seconds": 0.01,
        }

    def save(path: Path, m: Any, o: Any, identity: Any, progress: Any) -> dict[str, str]:
        assert path == directory and identity["stage"] == "final"
        assert identity["selected_inner_configuration"] == selection
        saved.update(
            identity=deepcopy(identity), progress=deepcopy(progress), rng={"fixture": True}
        )
        pointer = {"filename": "mocked.pt", "sha256": "mocked"}
        write_json(path / "latest.json", pointer)
        return pointer

    def load(path: Path, m: Any, o: Any, identity: Any) -> dict[str, Any]:
        assert path == directory and path.name == "final"
        assert identity == saved["identity"]
        loaded.append(path)
        result = deepcopy(saved)
        # A final stage must not use the inner early-stopping condition.
        result["progress"]["bad_epochs"] = 5
        return result

    monkeypatch.setattr(trainer, "build_model", build)
    monkeypatch.setattr(trainer, "make_optimizer", lambda *a: optimizer)
    monkeypatch.setattr(trainer, "ExperimentDataset", dataset)
    monkeypatch.setattr(trainer, "run_epoch", epoch)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    monkeypatch.setattr(trainer, "load_checkpoint", load)
    monkeypatch.setattr(trainer, "restore_rng", lambda value: restored.append(value))
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            trainer.train_stage(
                context,
                split,
                "B1",
                candidate,
                run_id,
                smoke=False,
                resume=True,
                selection=selection,
            )
    result = trainer.train_stage(
        context, split, "B1", candidate, run_id, smoke=False, resume=True, selection=selection
    )
    assert epochs == [1, 2] and result["progress"]["epoch"] == 2
    assert flags == ([True, False] if interrupt else [True])
    assert len(loaded) == len(restored) == int(interrupt)
    assert all(set(h) == {"epoch", "training"} for h in result["progress"]["history"])
    assert result["identity"]["all_development_speakers"] == list(split.speakers("development"))
    assert read_json(inner) == {"must_never_be_read": True}


@pytest.mark.parametrize("failure", [False, True])
def test_whole_fold_final_reuses_completed_runs_and_explicit_failed_recovery(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    freeze_protocol(context)
    monkeypatch.setattr(pipeline, "train_stage", fake_training)
    pipeline.search_c2(context, 1)
    bundle = pipeline.prepare_ablations(context, 1)
    bundle_path = context.config.output("experiments/selection/fold1_ablations.json")
    before = bundle_path.read_bytes()
    calls: list[str] = []
    failed = False

    def final(*args: Any, selection: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        nonlocal failed
        ablation = args[2]
        assert selection == bundle["selections"][ablation] and not kwargs["smoke"]
        calls.append(ablation)
        if failure and ablation == "A1" and not failed:
            failed = True
            raise RuntimeError("synthetic recoverable interruption")
        report = fake_training(*args, selection=selection, **kwargs)
        assert report["progress"]["epoch"] == selection["epoch"]
        return report

    monkeypatch.setattr(pipeline, "train_stage", final)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic"):
            pipeline.retrain_fold(context, 1, resume=True)
        with pytest.raises(ValueError, match="immutable inner selection"):
            pipeline.retrain_fold(context, 1, resume=True)
    result = pipeline.retrain_fold(
        context, 1, resume=True, retry_reason="synthetic fixture cleared" if failure else None
    )
    assert result == {"outer_fold": 1, "status": "FINAL_TRAIN_COMPLETE", "outer_evaluations": 0}
    assert calls.count("A0") == 1 and len(calls) == (8 if failure else 7)
    completed = list(calls)
    assert pipeline.retrain_fold(context, 1, resume=True) == result and calls == completed
    assert bundle_path.read_bytes() == before
    runs = Ledger(context.config).all_runs()
    assert sum(r["state"] == "FINAL_TRAIN_COMPLETE" for r in runs) == 7
    assert sum(r["state"] == "INNER_COMPLETE" for r in runs) == 3
    assert all(r["claim"] is None for r in runs)


def test_checksum_valid_checkpoint_still_rejects_identity_drift(tmp_path: Path) -> None:
    payload = {
        "schema": "module5_checkpoint_v1",
        "identity": {"seed": 99},
        "progress": {},
        "model": {},
        "optimizer": {},
        "rng": {},
    }
    temporary = tmp_path / "fixture.pt"
    torch.save(payload, temporary)
    checksum = file_hash(temporary)
    temporary.rename(tmp_path / f"{checksum}.pt")
    write_json(tmp_path / "latest.json", {"filename": f"{checksum}.pt", "sha256": checksum})
    with pytest.raises(ValueError, match="schema/provenance mismatch"):
        load_checkpoint(tmp_path, None, None, {"seed": 42})
