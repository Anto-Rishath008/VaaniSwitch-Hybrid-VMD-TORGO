from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from vaaniswitch.m1_vmd.audio import standardize_audio
from vaaniswitch.m2_representations.mel import generate_log_mel
from vaaniswitch.m3_encoders.schemas import EncoderBatch
from vaaniswitch.m4_fusion.losses import batch_labels, make_loss
from vaaniswitch.m4_fusion.model import TokenFusionClassifier
from vaaniswitch.m5_experiments.ablation_models import AblationModel
from vaaniswitch.m5_experiments.cli import cache_phase
from vaaniswitch.m5_experiments.config import ABLATIONS, CANDIDATES, file_hash
from vaaniswitch.m5_experiments.data import ExperimentDataset, batches
from vaaniswitch.m5_experiments.no_vmd import NoVMDCache
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.source import ExperimentContext


@pytest.mark.parametrize(
    "ablation,branches,width",
    [
        ("A0", {"mel"}, 128),
        ("A1", {"mel"}, 128),
        ("B1", {"graph"}, 128),
        ("B2", {"recurrence"}, 128),
        ("B3", {"graph", "recurrence"}, 256),
        ("C1", {"mel", "graph", "recurrence"}, 256),
        ("C2", {"mel", "graph", "recurrence"}, 256),
    ],
)
def test_ablation_shapes_branches_finite_and_gradients(
    context: ExperimentContext, ablation: str, branches: set[str], width: int
) -> None:
    dataset = ExperimentDataset(context, context.split(1), "inner_train", "A1", smoke=True)
    batch = next(batches(dataset, smoke=True))
    model = AblationModel(
        context.predecessor.encoder_config, ablation, CANDIDATES[0], load_pretrained=False
    )
    assert set(model.branches) == branches
    assert any(isinstance(m, nn.MultiheadAttention) for m in model.modules()) == (ablation == "C2")
    frozen = {n: t.clone() for n, t in model.state_dict().items() if ".backbone." in n}
    output = model(batch)
    assert output.logits.shape == (4,) and output.pooled.shape == (4, width)
    assert all(torch.isfinite(t).all() for t in output.tokens.values())
    loss = make_loss(1.0)(output.logits, batch_labels(batch))
    loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    assert all(torch.equal(t, model.state_dict()[n]) for n, t in frozen.items())
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(batch).logits, model(batch).logits)
    if ablation == "B3":
        assert torch.equal(
            output.pooled,
            torch.cat(
                (output.tokens["graph"].mean(1), output.tokens["recurrence"].mean(1)), dim=-1
            ),
        )
    if ablation == "C1":
        nonlinear = torch.cat((output.tokens["graph"], output.tokens["recurrence"]), 1)
        assert torch.equal(
            output.pooled, torch.cat((output.tokens["mel"].mean(1), nonlinear.mean(1)), -1)
        )


def test_c2_matches_signed_fusion_and_qkv_direction(context: ExperimentContext) -> None:
    model = AblationModel(
        context.predecessor.encoder_config, "C2", CANDIDATES[0], load_pretrained=False
    )
    batch = next(
        batches(
            ExperimentDataset(context, context.split(1), "inner_train", "C2", smoke=True),
            smoke=True,
        )
    )
    assert model.fusion is not None
    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.update({k: kwargs[k] for k in ("query", "key", "value")})

    handle = model.fusion.cross_attention.attention.register_forward_pre_hook(
        hook, with_kwargs=True
    )
    model.eval()
    with torch.no_grad():
        output = model(batch)
        handle.remove()
        signed = TokenFusionClassifier().eval()
        signed.load_state_dict(model.fusion.state_dict())
        nonlinear = torch.cat((output.tokens["graph"], output.tokens["recurrence"]), 1)
        assert torch.equal(captured["query"], output.tokens["mel"])
        assert torch.equal(captured["key"], nonlinear) and torch.equal(captured["value"], nonlinear)
        assert torch.equal(output.logits, signed(output.tokens["mel"], nonlinear).logits)


def test_overlapping_initial_weights_identical_and_head_counts(context: ExperimentContext) -> None:
    models = {
        a: AblationModel(
            context.predecessor.encoder_config, a, CANDIDATES[0], load_pretrained=False
        )
        for a in ABLATIONS
    }
    for ablation, model in models.items():
        for name, branch in model.branches.items():
            reference = models["C2"].branches[name].state_dict()
            assert all(torch.equal(t, reference[k]) for k, t in branch.state_dict().items())
        assert model.parameter_counts()["classifier"] == (
            33025 if ablation in {"B3", "C1", "C2"} else 16641
        )
    assert models["C2"].parameter_counts()["trainable"] == 291921
    assert models["C1"].parameter_counts()["trainable"] == 225617
    with pytest.raises(ValueError, match="design-only"):
        AblationModel(
            context.predecessor.encoder_config, "C3", CANDIDATES[0], load_pretrained=False
        )


def test_a0_conditioning_mel_and_selected_primary_exact(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = NoVMDCache(context)
    record = context.split(1).records(context.predecessor.source.contract, "inner_train")[0]
    source = context.config.repo_root / str(record.primary_audio_path)
    before = source.read_bytes()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("A0 accessed a VMD or Module 2 source")

    monkeypatch.setattr(context.predecessor.source, "load", forbidden)
    assert cache.build(record) == "computed"
    expected = generate_log_mel(standardize_audio(source).waveform, cache.mel).image
    assert np.array_equal(cache.load(record, check_source=True), expected)
    assert cache.build(record) == "cache_hit"
    assert source.read_bytes() == before
    assert "module5_ablation" in cache.directory(record.logical_id).parts
    assert "representations" not in cache.directory(record.logical_id).parts
    data = ExperimentDataset(context, context.split(1), "inner_train", "A0")
    sample = data[0]
    assert np.array_equal(sample.mel.numpy()[0], expected)
    assert sample.provenance.logical_id == record.logical_id


@pytest.mark.parametrize("damage", ["checksum", "binding", "missing", "extra", "nan", "shape"])
def test_a0_corrupt_cache_rejected_and_preserved(context: ExperimentContext, damage: str) -> None:
    cache = NoVMDCache(context)
    record = context.split(1).records(context.predecessor.source.contract, "inner_train")[0]
    cache.build(record)
    directory = cache.directory(record.logical_id)
    metadata = read_json(directory / "metadata.json")
    if damage == "checksum":
        metadata["mel_sha256"] = "0" * 64
    elif damage == "binding":
        metadata["binding"]["config_hash"] = "0" * 64
    elif damage == "missing":
        (directory / "mel.npy").unlink()
    elif damage == "extra":
        (directory / "extra.txt").write_text("unexpected")
    else:
        values = np.full((224, 224) if damage == "nan" else (10, 10), np.nan, dtype=np.float32)
        np.save(directory / "mel.npy", values)
        metadata["mel_sha256"] = file_hash(directory / "mel.npy")
    write_json(directory / "metadata.json", metadata)
    snapshot = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises((ValueError, OSError)):
        cache.build(record)
    assert snapshot == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_a0_changed_primary_rejected(context: ExperimentContext) -> None:
    cache = NoVMDCache(context)
    record = context.split(1).records(context.predecessor.source.contract, "inner_train")[0]
    (context.config.repo_root / str(record.primary_audio_path)).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        cache.build(record)


def test_a0_forged_selected_primary_rejected(context: ExperimentContext) -> None:
    cache = NoVMDCache(context)
    records = context.split(1).records(context.predecessor.source.contract, "inner_train")
    forged = replace(
        records[0],
        primary_audio_path=records[1].primary_audio_path,
        primary_audio_sha256=records[1].primary_audio_sha256,
    )
    with pytest.raises(ValueError, match="signed record"):
        cache.build(forged)


def test_a0_cli_smoke_build_validate_resume(context: ExperimentContext) -> None:
    first = cache_phase(context, build=True, smoke=True)
    assert first["status"] == "PASS" and first["computed"] <= 6
    second = cache_phase(context, build=True, smoke=True)
    assert second["computed"] == 0 and second["cache_hit"] == first["computed"]
    assert cache_phase(context, build=False, smoke=True)["status"] == "PASS"


def test_unused_branch_values_do_not_affect_mel(context: ExperimentContext) -> None:
    dataset = ExperimentDataset(context, context.split(1), "inner_train", "A1", smoke=True)
    batch = next(batches(dataset, smoke=True))
    model = AblationModel(
        context.predecessor.encoder_config, "A1", CANDIDATES[0], load_pretrained=False
    )
    model.eval()
    corrupted: EncoderBatch = replace(
        batch,
        recurrence=torch.full_like(batch.recurrence, 17),
        graph_nodes=torch.full_like(batch.graph_nodes, float("nan")),
    )
    with torch.no_grad():
        assert torch.equal(model(batch).logits, model(corrupted).logits)
