"""Synthetic retained-ASR validation, including Windows locale-independent JSON."""

import json
from pathlib import Path
from typing import Any

import pytest

from vaaniswitch.m6_reconstruction import runtime
from vaaniswitch.m6_reconstruction.artifacts import sha256, verify_checksum, write_bytes, write_json
from vaaniswitch.m6_reconstruction.config import MODEL_SHA, RuntimeConfig


def test_resume_unicode_and_hash_validation(
    tmp_path: Path, config_values: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RuntimeConfig(tmp_path, config_values)
    previous = tmp_path / "artifacts/module6/failed"
    job = {"logical_id": "M03/Session2/0013", "primary_audio_sha256": "fixture"}
    write_json(tmp_path, previous / "FAILURE.json", {"tts": "timeout"})
    write_json(tmp_path, previous / "config.resolved.json", config_values)
    raw = write_bytes(tmp_path, previous / "raw_asr.txt", "  Exact “raw”.\n".encode())
    backend = {"text": "  Exact “raw”.\n", "segments": []}
    backend_path = write_json(tmp_path, previous / "backend_result.json", backend)
    write_json(
        tmp_path,
        previous / "asr_smoke.json",
        {
            "source": job,
            "device": "cuda",
            "checkpoint_sha256": MODEL_SHA,
            "raw_text": backend["text"],
            "backend_result": backend,
            "raw_text_sha256": sha256(raw),
            "backend_json_sha256": sha256(backend_path),
            "decoding": config_values["asr"]["decoding"],
        },
    )
    write_json(tmp_path, previous / "cuda_model_probe.json", {"device": "cuda"})
    write_json(
        tmp_path,
        previous / "reproducibility.json",
        {
            "source_sha256": {},
            "identity": {"lock": "fixture-lock"},
        },
    )
    monkeypatch.setattr(runtime, "resolve_smoke_audio", lambda root: job)

    def verify(path: Path, expected: str) -> str:
        if path in (raw, backend_path):
            return verify_checksum(path, expected)
        return expected

    monkeypatch.setattr(runtime, "verify_checksum", verify)
    # Force a legacy Windows locale regardless of the test host's UTF-8 settings.
    original = Path.read_text

    def legacy_read(path: Path, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("encoding", "cp1252")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", legacy_read)
    retained = runtime.retained_asr(config, previous)
    assert retained["raw_asr.txt"] == raw.read_bytes()
    assert json.loads(retained["backend_result.json"]) == backend
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.retained_asr(config, previous)
