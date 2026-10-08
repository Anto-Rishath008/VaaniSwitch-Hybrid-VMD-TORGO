"""Reproductions for concrete checkpoint and output-boundary defects found in review."""

import hashlib
from pathlib import Path

import pytest
import torch

from tests.m4_fusion.helpers import make_context
from vaaniswitch.m4_fusion.checkpoint import _validate_optimizer, load_checkpoint
from vaaniswitch.m4_fusion.model import DysarthriaClassifier
from vaaniswitch.m4_fusion.pipeline import selection_identity
from vaaniswitch.m4_fusion.trainer import make_optimizer


def test_matching_checksum_invalid_pickle_is_controlled_rejection(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    path = tmp_path / "invalid.pt"
    content = b"not a pytorch archive"
    path.write_bytes(content)
    path.with_suffix(".pt.sha256").write_text(hashlib.sha256(content).hexdigest())
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


def test_optimizer_state_cannot_hide_extra_dataset_arrays() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state_dict()
    state["state"][99999] = {"dataset": torch.zeros(1, 224, 224)}
    with pytest.raises(ValueError, match="optimizer"):
        _validate_optimizer(state, optimizer)


def test_resolved_output_directory_cannot_redirect_to_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_context(tmp_path).config
    original = Path.resolve
    redirected = tmp_path / "artifacts/metrics"

    def resolve(path: Path, strict: bool = False) -> Path:
        if path == redirected:
            return tmp_path / "data/raw"
        return original(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="output"):
        config.output("metrics", "module4_summary.json")
