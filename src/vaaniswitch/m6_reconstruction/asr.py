"""Frozen audio-only Whisper adapter; acquisition is a separate explicit action."""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
import urllib.request
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .artifacts import output_path, verify_checksum
from .config import DECODING, MODEL_SHA, MODEL_SOURCE, RuntimeConfig


def acquire_model(config: RuntimeConfig) -> dict[str, Any]:
    target = config.checkpoint
    if target.exists():
        return {"path": str(target), "sha256": verify_checksum(target, MODEL_SHA), "reused": True}
    partial = output_path(config.root, target.with_suffix(".pt.part"))
    partial.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # An interrupted partial is preserved; no automatic overwrite or restart.
    with partial.open("xb") as stream, urllib.request.urlopen(MODEL_SOURCE, timeout=60) as reply:
        while chunk := reply.read(1024 * 1024):
            stream.write(chunk)
    actual = verify_checksum(partial, MODEL_SHA)
    partial.rename(target)
    return {
        "path": str(target),
        "sha256": actual,
        "reused": False,
        "source": MODEL_SOURCE,
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": target.stat().st_size,
    }


def load_frozen_model(config: RuntimeConfig) -> tuple[Any, dict[str, Any]]:
    import torch
    import whisper  # type: ignore[import-untyped]

    verify_checksum(config.checkpoint, MODEL_SHA)
    if importlib.metadata.version("openai-whisper") != "20250625":
        raise ValueError("Whisper package version drift")
    torch.set_num_threads(config.values["execution"]["cpu_threads"])
    torch.manual_seed(42)
    np.random.seed(42)
    probe: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": config.device,
        "dtype": config.values["execution"]["dtype"],
        "cpu_threads": torch.get_num_threads(),
        "cuda_allocation": False,
    }
    if config.device == "cuda":
        if not torch.version.cuda or not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicit documented fallback required")
        name = torch.cuda.get_device_name(0)
        if name != "NVIDIA GeForce RTX 4070 Laptop GPU":
            raise RuntimeError(f"unexpected GPU: {name}")
        tensor = torch.ones(8, device="cuda")
        torch.cuda.synchronize()
        probe.update(
            cuda_device_name=name,
            cuda_allocation=bool(tensor.sum().item() == 8),
            device_capability=list(torch.cuda.get_device_capability(0)),
        )
        del tensor
    start = time.perf_counter()
    model = whisper.load_model(str(config.checkpoint), device=config.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if str(model.device).split(":")[0] != config.device:
        raise RuntimeError("loaded model device mismatch")
    if config.device == "cuda":
        torch.cuda.synchronize()
    probe["model_load_seconds"] = time.perf_counter() - start
    probe["model_device"] = str(model.device)
    probe["parameter_storage_dtype"] = str(next(model.parameters()).dtype)
    probe["all_parameters_frozen"] = all(not p.requires_grad for p in model.parameters())
    return model, probe


def transcribe_audio(model: Any, audio: NDArray[np.float32], *, device: str) -> dict[str, Any]:
    """Only waveform and explicit device enter; no references, arbitrary options or prompts."""
    import torch

    if device not in ("cuda", "cpu") or str(model.device).split(":")[0] != device:
        raise ValueError("explicit device must match the loaded model")
    if audio.dtype != np.float32 or audio.ndim != 1 or not len(audio):
        raise ValueError("expected nonempty float32 mono waveform")
    if not np.isfinite(audio).all():
        raise ValueError("nonfinite audio")
    options = {**DECODING, "fp16": device == "cuda"}
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        backend = model.transcribe(audio, **options)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not isinstance(backend, dict) or not isinstance(backend.get("text"), str):
        raise ValueError("backend did not return raw text")
    return {
        "backend_result": backend,
        "raw_text": backend["text"],
        "device": device,
        "dtype": "float16" if device == "cuda" else "float32",
        "decoding": options,
        "elapsed_seconds": elapsed,
        "input_frames": len(audio),
        "input_float32_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "reference_context_used": False,
    }
