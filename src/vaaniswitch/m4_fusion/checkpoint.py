"""Checksummed tensor-only checkpoints bound to exact config, split, and record selection."""

from __future__ import annotations

import hashlib
import io
import math
import pickle
from pathlib import Path
from typing import Any

import torch

from vaaniswitch.m4_fusion.config import Module4Config
from vaaniswitch.m4_fusion.model import DysarthriaClassifier
from vaaniswitch.m4_fusion.reproducibility import capture_rng
from vaaniswitch.m4_fusion.splits import SpeakerSplit


def checkpoint_directory(config: Module4Config, split: SpeakerSplit, *, smoke: bool) -> Path:
    root = config.output("checkpoints", "module4")
    suffix = "_smoke" if smoke else ""
    path = root / f"fold{split.outer_test_fold}_seed{split.seed}_{config.config_hash[:12]}{suffix}"
    if not path.resolve().is_relative_to(root):
        raise ValueError("checkpoint directory escapes Module 4")
    return path


def save_checkpoint(
    path: Path,
    model: DysarthriaClassifier,
    optimizer: torch.optim.Optimizer,
    config: Module4Config,
    split: SpeakerSplit,
    *,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    bad_epochs: int,
    pos_weight: float,
    selection: dict[str, Any],
    smoke: bool,
    history: list[dict[str, Any]],
) -> None:
    model.verify_frozen_backbone()
    payload = {
        "schema_version": "1.0",
        "module": "4",
        "mode": "smoke" if smoke else "baseline",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "pos_weight": pos_weight,
        "hashes": config.hashes,
        "processing_identity": config.processing_identity,
        "split": split.identity,
        "split_hash": split.split_hash,
        "selection": selection,
        "seed": split.seed,
        "rng": capture_rng(),
        "history": history,
        "pretrained_weights": model.encoders.mel_encoder.pretrained_weights_identifier,
        "runtime": {"torch": str(torch.__version__), "cpu_threads": torch.get_num_threads()},
        "outer_test_forward_count": 0,
    }
    _validate_metadata(payload, config, split, selection, smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    temporary_digest = digest_path.with_suffix(".tmp")
    temporary_digest.write_text(hashlib.sha256(content).hexdigest() + "\n", encoding="ascii")
    temporary_digest.replace(digest_path)


def load_checkpoint(
    path: Path,
    model: DysarthriaClassifier,
    optimizer: torch.optim.Optimizer,
    config: Module4Config,
    split: SpeakerSplit,
    *,
    selection: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    """Validate before forward; weights_only=True never permits arbitrary pickle globals."""
    try:
        content = path.read_bytes()
        digest = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("checkpoint checksum mismatch")
        payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must contain a mapping")
        _validate_metadata(payload, config, split, selection, smoke)
        state = payload["model_state"]
        expected = model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("checkpoint model keys differ from architecture")
        for key, value in state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected[key].shape
                or (value.dtype != expected[key].dtype or not torch.isfinite(value).all())
            ):
                raise ValueError(f"invalid model tensor: {key}")
        _validate_optimizer(payload["optimizer_state"], optimizer)
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        model.encoders.mel_encoder.pretrained_weights_identifier = payload["pretrained_weights"]
        model.verify_frozen_backbone()
        return payload
    except (
        OSError,
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
        IndexError,
    ) as exc:
        raise ValueError(f"checkpoint rejected: {exc}") from exc


def _validate_metadata(
    payload: dict[str, Any],
    config: Module4Config,
    split: SpeakerSplit,
    selection: dict[str, Any],
    smoke: bool,
) -> None:
    keys = {
        "schema_version",
        "module",
        "mode",
        "model_state",
        "optimizer_state",
        "epoch",
        "best_metric",
        "best_epoch",
        "bad_epochs",
        "pos_weight",
        "hashes",
        "processing_identity",
        "split",
        "split_hash",
        "selection",
        "seed",
        "rng",
        "history",
        "pretrained_weights",
        "runtime",
        "outer_test_forward_count",
    }
    if set(payload) != keys:
        raise ValueError("checkpoint fields do not match schema; dataset arrays are forbidden")
    required = {
        "schema_version": "1.0",
        "module": "4",
        "mode": "smoke" if smoke else "baseline",
        "hashes": config.hashes,
        "processing_identity": config.processing_identity,
        "split": split.identity,
        "split_hash": split.split_hash,
        "selection": selection,
        "seed": config.training.seed,
        "outer_test_forward_count": 0,
    }
    if any(payload[key] != value for key, value in required.items()):
        raise ValueError("checkpoint config, split, mode, or source identity mismatch")
    for key in ("epoch", "best_epoch", "bad_epochs"):
        if type(payload[key]) is not int or not 0 <= payload[key] <= config.training.max_epochs:
            raise ValueError(f"invalid checkpoint {key}")
    if not 1 <= payload["best_epoch"] <= payload["epoch"]:
        raise ValueError("invalid best epoch")
    for key in ("best_metric", "pos_weight"):
        if type(payload[key]) not in (float, int) or not math.isfinite(payload[key]):
            raise ValueError(f"invalid checkpoint {key}")
    if not 0 <= payload["best_metric"] <= 1 or payload["pos_weight"] <= 0:
        raise ValueError("invalid checkpoint metric or class weight")
    if not isinstance(payload["rng"], dict) or set(payload["rng"]) != {"python", "numpy", "torch"}:
        raise ValueError("checkpoint RNG state missing")
    if not isinstance(payload["history"], list) or len(payload["history"]) != payload["epoch"]:
        raise ValueError("checkpoint epoch history missing")
    if not smoke and payload["pretrained_weights"] != "ResNet18_Weights.IMAGENET1K_V1":
        raise ValueError("baseline checkpoint must use signed ImageNet initialization")


def _validate_optimizer(state: Any, optimizer: torch.optim.Optimizer) -> None:
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("invalid optimizer state")
    groups = state["param_groups"]
    expected_groups = optimizer.state_dict()["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(expected_groups):
        raise ValueError("optimizer groups mismatch")
    expected_indices = {index for group in expected_groups for index in group["params"]}
    if not isinstance(state["state"], dict) or set(state["state"]) != expected_indices:
        raise ValueError("optimizer state indices must cover exactly the trainable parameters")
    for group, expected, live in zip(groups, expected_groups, optimizer.param_groups, strict=True):
        if group != expected:
            raise ValueError("optimizer settings differ from baseline")
        for index, parameter in zip(group["params"], live["params"], strict=True):
            saved = state["state"].get(index, {})
            if set(saved) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("optimizer state is incomplete")
            for name, value in saved.items():
                shape = () if name == "step" else parameter.shape
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != shape
                    or not (torch.isfinite(value).all())
                ):
                    raise ValueError("optimizer state has invalid tensor")
