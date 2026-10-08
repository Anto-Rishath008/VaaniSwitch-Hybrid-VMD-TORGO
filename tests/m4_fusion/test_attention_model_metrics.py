"""Hand-computable metrics, attention direction, residual, trainability, and optimization."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from tests.m4_fusion.helpers import make_context
from vaaniswitch.m4_fusion.attention import MelCrossAttention
from vaaniswitch.m4_fusion.diagnostics import gradient_audit, parameter_audit, tiny_batch_overfit
from vaaniswitch.m4_fusion.losses import make_loss
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m4_fusion.model import DysarthriaClassifier, TokenFusionClassifier
from vaaniswitch.m4_fusion.trainer import batches


def test_attention_direction_residual_pooling_and_optional_weights() -> None:
    model = TokenFusionClassifier().eval()
    mel = torch.randn(2, 16, 128)
    nonlinear = torch.randn(2, 17, 128)
    observed: list[dict[str, Any]] = []

    def inspect(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        observed.append(kwargs)

    hook = model.cross_attention.attention.register_forward_pre_hook(inspect, with_kwargs=True)
    output = model(mel, nonlinear)
    diagnostic = model(mel, nonlinear, return_attention=True)
    hook.remove()
    assert observed[0]["query"] is mel
    assert observed[0]["key"] is nonlinear and observed[0]["value"] is nonlinear
    assert observed[0]["need_weights"] is False and observed[1]["need_weights"] is True
    assert observed[0]["query"].shape[1] == 16 and observed[0]["key"].shape[1] == 17
    assert output.attention_weights is None
    assert diagnostic.attention_weights is not None
    assert diagnostic.attention_weights.shape == (2, 4, 16, 17)
    torch.testing.assert_close(diagnostic.attention_weights.sum(-1), torch.ones(2, 4, 16))
    assert output.attention_output.shape == output.attended_mel_tokens.shape == (2, 16, 128)
    torch.testing.assert_close(
        output.attended_mel_tokens, model.cross_attention.layer_norm(mel + output.attention_output)
    )
    assert output.mel_pool.shape == output.nonlinear_pool.shape == (2, 128)
    torch.testing.assert_close(output.mel_pool, output.attended_mel_tokens.mean(1))
    torch.testing.assert_close(output.nonlinear_pool, nonlinear.mean(1))
    assert output.fusion_vector.shape == (2, 256)
    torch.testing.assert_close(
        output.fusion_vector, torch.cat((output.mel_pool, output.nonlinear_pool), -1)
    )
    assert output.logits.shape == (2,) and output.probabilities.shape == (2,)
    assert ((output.probabilities >= 0) & (output.probabilities <= 1)).all()
    assert torch.isfinite(output.attended_mel_tokens).all()


def test_nonlinear_perturbation_affects_attention_with_mel_fixed() -> None:
    module = MelCrossAttention().eval()
    mel, nonlinear = torch.randn(1, 16, 128), torch.randn(1, 17, 128)
    first = module(mel, nonlinear)[1]
    second = module(mel, nonlinear + torch.randn_like(nonlinear))[1]
    assert not torch.allclose(first, second)


def test_zero_attention_keeps_original_mel_residual() -> None:
    module = MelCrossAttention().eval()
    nn.init.zeros_(module.attention.out_proj.weight)
    nn.init.zeros_(module.attention.out_proj.bias)
    mel = torch.randn(1, 16, 128)
    attended, attention, _ = module(mel, torch.randn(1, 17, 128))
    assert torch.count_nonzero(attention) == 0
    torch.testing.assert_close(attended, module.layer_norm(mel))


@pytest.mark.parametrize(
    ("mel_shape", "nonlinear_shape"),
    [
        ((1, 1, 128), (1, 17, 128)),
        ((1, 16, 192), (1, 17, 192)),
        ((1, 16, 128), (1, 16, 128)),
        ((1, 16, 128), (2, 17, 128)),
        ((0, 16, 128), (0, 17, 128)),
        ((16, 128), (17, 128)),
    ],
)
def test_invalid_token_shape_rejected(
    mel_shape: tuple[int, ...], nonlinear_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        MelCrossAttention()(torch.zeros(mel_shape), torch.zeros(nonlinear_shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_tokens_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        MelCrossAttention()(torch.full((1, 16, 128), value), torch.zeros(1, 17, 128))


def test_training_logit_path_has_no_sigmoid() -> None:
    model = TokenFusionClassifier()
    assert not any(isinstance(m, nn.Sigmoid) for m in model.modules())
    head = model.classifier[-1]
    assert isinstance(head, nn.Linear)
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, 1000.0)
    output = model(torch.randn(1, 16, 128), torch.randn(1, 17, 128))
    criterion = make_loss(1.0)
    assert isinstance(criterion, nn.BCEWithLogitsLoss)
    assert output.logits.item() == 1000.0
    assert criterion(output.logits, torch.zeros(1)).item() == 1000.0


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("accuracy", 0.5),
        ("balanced_accuracy", 0.5),
        ("macro_f1", 17 / 35),
        ("precision", 1 / 3),
        ("sensitivity", 0.5),
        ("specificity", 0.5),
        ("roc_auc", 0.75),
    ],
)
def test_hand_computed_metrics(key: str, expected: float) -> None:
    result = binary_metrics([0, 0, 0, 0, 1, 1], [0.1, 0.2, 0.6, 0.8, 0.3, 0.9])
    assert result[key] == pytest.approx(expected)
    assert result["confusion_matrix"] == [[2, 2], [1, 1]]
    assert [result[k] for k in ("TN", "FP", "FN", "TP")] == [2, 2, 1, 1]


def test_auc_ranking_ties_and_threshold_boundary() -> None:
    assert binary_metrics([0, 1], [0.5, 0.5])["roc_auc"] == 0.5
    assert binary_metrics([1, 0], [0.9, 0.1])["roc_auc"] == 1.0
    assert binary_metrics([1, 0], [0.1, 0.9])["roc_auc"] == 0.0
    result = binary_metrics([0, 1], [0.499, 0.5])
    assert result["threshold"] == 0.5 and result["accuracy"] == 1.0


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_auc_single_class_unavailable_and_json_safe(labels: list[int]) -> None:
    result = binary_metrics(labels, [0.1, 0.8])
    assert result["roc_auc"] is None
    assert result["roc_auc_unavailable_reason"] == "only one true class present"
    assert (result["sensitivity"] is None) == (labels[0] == 0)
    assert (result["specificity"] is None) == (labels[0] == 1)


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([], []),
        ([0], []),
        ([2], [0.5]),
        ([0], [float("nan")]),
        ([0], [1.1]),
        ([1], [-0.1]),
    ],
)
def test_invalid_metric_inputs_rejected(labels: list[int], scores: list[float]) -> None:
    with pytest.raises(ValueError):
        binary_metrics(labels, scores)


def test_no_positive_predictions_zero_precision() -> None:
    result = binary_metrics([0, 1], [0.1, 0.2])
    assert result["precision"] == 0.0
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.1, 0.2], threshold=0.4)
    with pytest.raises(ValueError):
        make_loss(-1)


def test_gradients_freeze_eval_determinism_and_parameter_counts(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    batch = next(batches(context, "train", smoke=True))
    model = DysarthriaClassifier(context.encoder_config, load_pretrained=False)
    audit = gradient_audit(model, batch)
    assert audit["status"] == "PASS"
    assert all(
        row["finite"] and row["absolute_gradient_sum"] > 0 for row in audit["groups"].values()
    )
    model.verify_frozen_backbone()
    assert all(p.grad is None for p in model.encoders.mel_encoder.backbone.parameters())
    model.eval()
    assert torch.equal(model(batch).logits, model(batch).logits)
    counts = parameter_audit(model)
    assert counts["module3"]["total_parameters"] == 11_369_104
    assert counts["module3"]["initial_trainable_parameters"] == 192_592
    assert counts["combined_frozen"] == 11_176_512
    assert counts["cross_attention"] == 66_048
    assert counts["residual_layer_norm"] == 256
    assert counts["classifier"] == 33_025
    assert counts["combined_total"] == 11_468_433
    assert counts["combined_trainable"] == 291_921
    model.encoders.mel_encoder.set_backbone_trainable(True)
    with pytest.raises(ValueError, match="frozen"):
        model(batch)


def test_tiny_batch_loss_decreases_and_backbone_buffers_stay_frozen(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    batch = next(batches(context, "train", smoke=True))
    model = DysarthriaClassifier(context.encoder_config, load_pretrained=False)
    before = {k: v.clone() for k, v in model.encoders.mel_encoder.backbone.state_dict().items()}
    result, _ = tiny_batch_overfit(model, batch, context)
    assert result["final_eval_bce"] < result["initial_eval_bce"] * 0.8
    assert result["scientific_performance_result"] is False
    assert all(
        torch.equal(before[k], v)
        for k, v in model.encoders.mel_encoder.backbone.state_dict().items()
    )
