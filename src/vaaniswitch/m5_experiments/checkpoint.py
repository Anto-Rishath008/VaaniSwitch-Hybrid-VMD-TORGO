"""Atomic checkpoint generations with checksum, schema, optimizer and provenance validation."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from vaaniswitch.m4_fusion.checkpoint import _validate_optimizer
from vaaniswitch.m4_fusion.reproducibility import capture_rng
from vaaniswitch.m5_experiments.ablation_models import AblationModel
from vaaniswitch.m5_experiments.config import file_hash
from vaaniswitch.m5_experiments.reporting import read_json, write_json


def validate_progress(progress: dict[str, Any]) -> None:
    json.dumps(progress, allow_nan=False)
    epoch = progress.get("epoch")
    if type(epoch) is not int or not 1 <= epoch <= 30:
        raise ValueError("invalid checkpoint epoch")
    history = progress.get("history")
    if not isinstance(history, list) or [h.get("epoch") for h in history] != list(
        range(1, epoch + 1)
    ):
        raise ValueError("checkpoint history does not match epoch")
    if type(progress.get("bad_epochs")) is not int or not 0 <= progress["bad_epochs"] <= 30:
        raise ValueError("invalid checkpoint patience")
    if not isinstance(progress.get("pos_weight"), (int, float)) or progress["pos_weight"] <= 0:
        raise ValueError("invalid checkpoint class weight")


def save_checkpoint(
    directory: Path,
    model: AblationModel,
    optimizer: torch.optim.Optimizer,
    identity: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, str]:
    model.verify_frozen_backbone()
    validate_progress(progress)
    payload = {
        "schema": "module5_checkpoint_v1",
        "identity": identity,
        "progress": progress,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    checksum = hashlib.sha256(content).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / "latest.json"
    previous = read_json(pointer) if pointer.exists() else None
    path = directory / f"{checksum}.pt"
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(content)
    if file_hash(path) != checksum:
        raise ValueError("checkpoint generation checksum mismatch")
    result = {"filename": path.name, "sha256": checksum}
    write_json(pointer, result)
    # Only the superseded resumable generation is removed. The scientific inner best
    # is metadata; final retraining starts fresh, so no best-inner weights are needed.
    if previous and previous != result:
        old = checked_generation(directory, previous)
        old.unlink()
    return result


def checked_generation(directory: Path, pointer: dict[str, Any]) -> Path:
    checksum = pointer.get("sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(c not in "0123456789abcdef" for c in checksum)
        or pointer.get("filename") != f"{checksum}.pt"
    ):
        raise ValueError("invalid checkpoint generation pointer")
    path = directory / f"{checksum}.pt"
    if path.resolve() != path or file_hash(path) != checksum:
        raise ValueError("checkpoint checksum or path mismatch")
    return path


def load_checkpoint(
    directory: Path,
    model: AblationModel,
    optimizer: torch.optim.Optimizer,
    identity: dict[str, Any],
    *,
    pointer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        path = checked_generation(directory, pointer or read_json(directory / "latest.json"))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "identity", "progress", "model", "optimizer", "rng"}
            or payload["schema"] != "module5_checkpoint_v1"
            or payload["identity"] != identity
        ):
            raise ValueError("checkpoint schema/provenance mismatch")
        validate_progress(payload["progress"])
        expected = model.state_dict()
        if set(payload["model"]) != set(expected):
            raise ValueError("checkpoint architecture keys mismatch")
        for name, value in payload["model"].items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("invalid model tensor")
        _validate_optimizer(payload["optimizer"], optimizer)
        if not isinstance(payload["rng"], dict) or set(payload["rng"]) != {
            "python",
            "numpy",
            "torch",
        }:
            raise ValueError("invalid RNG state")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        model.verify_frozen_backbone()
        return dict(payload)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
        IndexError,
    ) as exc:
        raise ValueError(f"Module 5 checkpoint rejected: {exc}") from exc
