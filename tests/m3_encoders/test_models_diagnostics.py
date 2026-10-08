"""Encoder output, determinism, gradient, parameter, and state tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from tests.m3_encoders.helpers import make_batch, make_model_config
from vaaniswitch.m3_encoders.diagnostics import (
    deterministic_eval,
    gradient_sanity,
    measure_cpu_timings,
    parameter_audit,
    state_dict_roundtrip,
    token_statistics,
)
from vaaniswitch.m3_encoders.graph_encoder import VisibilityGraphTokenEncoder
from vaaniswitch.m3_encoders.mel_encoder import MelTokenEncoder
from vaaniswitch.m3_encoders.models import ParallelTokenEncoders
from vaaniswitch.m3_encoders.recurrence_encoder import RecurrenceTokenEncoder
from vaaniswitch.m3_encoders.schemas import EncoderBatch, EncoderTokenOutput


def test_mel_encoder_uses_feature_trunk_and_returns_deterministic_tokens(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    encoder = MelTokenEncoder(
        config.mel, token_dimension=128, load_pretrained=False
    ).eval()
    mel = make_batch().mel
    first = encoder(mel)
    second = encoder(mel)

    assert first.shape == (1, 16, 128)
    assert torch.equal(first, second)
    assert not any(isinstance(module, torch.nn.Linear) for module in encoder.backbone.modules())
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert all(parameter.requires_grad for parameter in encoder.projection.parameters())


def test_mel_backbone_supports_controlled_unfreezing(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    encoder = MelTokenEncoder(config.mel, token_dimension=128, load_pretrained=False)
    encoder.set_backbone_trainable(True)
    assert all(parameter.requires_grad for parameter in encoder.backbone.parameters())
    encoder.set_backbone_trainable(False)
    encoder.train()
    assert encoder.backbone.training is False


def test_recurrence_encoder_shape_and_eval_determinism(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    encoder = RecurrenceTokenEncoder(
        config.recurrence, token_dimension=128, dropout=config.dropout
    ).eval()
    recurrence = make_batch().recurrence
    first = encoder(recurrence)
    second = encoder(recurrence)
    assert first.shape == (1, 16, 128)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_all_token_contracts_and_label_independent_forward(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    model = ParallelTokenEncoders(config, load_pretrained=False).eval()
    batch = make_batch()
    changed_provenance = tuple(
        replace(item, condition="control", evaluation_fold=3) for item in batch.provenance
    )
    relabelled = EncoderBatch(
        mel=batch.mel,
        recurrence=batch.recurrence,
        graph_nodes=batch.graph_nodes,
        graph_node_mask=batch.graph_node_mask,
        graph_edges=batch.graph_edges,
        provenance=changed_provenance,
    )
    with torch.inference_mode():
        original = model(batch)
        changed = model(relabelled)
    original.validate()
    assert original.mel_tokens.shape == (1, 16, 128)
    assert original.graph_token.shape == (1, 1, 128)
    assert original.recurrence_tokens.shape == (1, 16, 128)
    assert original.nonlinear_tokens.shape == (1, 17, 128)
    assert torch.equal(original.mel_tokens, changed.mel_tokens)
    assert torch.equal(original.graph_token, changed.graph_token)
    assert torch.equal(original.recurrence_tokens, changed.recurrence_tokens)


def test_output_contract_rejects_wrong_shape_and_nonfinite_values() -> None:
    valid = EncoderTokenOutput(
        mel_tokens=torch.zeros((1, 16, 128)),
        graph_token=torch.zeros((1, 1, 128)),
        recurrence_tokens=torch.zeros((1, 16, 128)),
    )
    valid.validate()
    wrong = replace(valid, graph_token=torch.zeros((1, 2, 128)))
    try:
        wrong.validate()
    except ValueError as exc:
        assert "shape" in str(exc)
    else:  # pragma: no cover - explicit assertion branch
        raise AssertionError("wrong graph shape was accepted")
    nonfinite = replace(valid, mel_tokens=valid.mel_tokens.clone())
    nonfinite.mel_tokens[0, 0, 0] = torch.nan
    try:
        nonfinite.validate()
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:  # pragma: no cover - explicit assertion branch
        raise AssertionError("non-finite tokens were accepted")


def test_gradient_sanity_freezes_backbone_and_reaches_all_trainable_branches(
    tmp_path: Path,
) -> None:
    config = make_model_config(tmp_path)
    model = ParallelTokenEncoders(config, load_pretrained=False)
    result = gradient_sanity(model, make_batch())
    assert result["optimizer_step_performed"] is False
    assert all(result["frozen_mel_backbone"].values())
    for branch in ("mel_projection", "recurrence_cnn", "visibility_graph_gcn"):
        assert result[branch]["all_gradients_present"]
        assert result[branch]["all_gradients_finite"]


def test_state_roundtrip_parameter_audit_timing_and_statistics(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    model = ParallelTokenEncoders(config, load_pretrained=False).eval()
    batch = make_batch()
    roundtrip = state_dict_roundtrip(model, config, batch)
    parameters = parameter_audit(model)
    timing = measure_cpu_timings(model, batch, repetitions=1)
    with torch.inference_mode():
        stats = token_statistics(model(batch).mel_tokens)

    assert roundtrip["exact_match"]
    assert parameters["total_parameters"] == (
        parameters["initial_trainable_parameters"] + parameters["initial_frozen_parameters"]
    )
    assert parameters["mel_backbone"]["trainable"] == 0
    assert parameters["initial_frozen_parameters"] > parameters["initial_trainable_parameters"]
    assert timing["device"] == "cpu"
    assert timing["combined"]["count"] == 1
    assert stats["l2_norm"]["count"] == 16


def test_eval_determinism_helper_and_standalone_graph_encoder(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    model = ParallelTokenEncoders(config, load_pretrained=False)
    assert all(deterministic_eval(model, make_batch()).values())
    graph = VisibilityGraphTokenEncoder(config.graph, token_dimension=128, dropout=0.2)
    batch = make_batch()
    output = graph(batch.graph_nodes, batch.graph_edges, batch.graph_node_mask)
    assert output.shape == (
        1,
        1,
        128,
    )
