import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from vaaniswitch.m6_reconstruction.artifacts import (
    complete_json,
    digest,
    output_path,
    sha256,
    verify_checksum,
    write_bytes,
    write_json,
)
from vaaniswitch.m6_reconstruction.config import MODEL_SHA, VOICE_ID, load_config


def test_frozen_config(project_root: Path) -> None:
    config = load_config(project_root / "configs/module6/pipeline.yaml", project_root)
    assert config.device == "cuda"
    assert config.values["asr"]["sha256"] == MODEL_SHA
    assert config.values["tts"]["voice_id"] == VOICE_ID
    assert config.identity == digest(config.values)
    assert config.checkpoint == project_root / "artifacts/module6/models/whisper/small.en.pt"


@pytest.mark.parametrize("kind", ["prompt", "model", "voice", "device", "unknown", "boolean"])
def test_config_drift_fails(tmp_path: Path, config_values: dict[str, Any], kind: str) -> None:
    if kind == "prompt":
        config_values["asr"]["decoding"]["initial_prompt"] = "REFERENCE_SENTINEL"
    elif kind == "model":
        config_values["asr"]["sha256"] = "0" * 64
    elif kind == "voice":
        config_values["tts"]["voice_id"] = "David"
    elif kind == "device":
        config_values["execution"]["device"] = "auto"
    elif kind == "boolean":
        config_values["smoke"]["metrics_enabled"] = 0
    else:
        config_values["reference"] = "REFERENCE_SENTINEL"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_values), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path, tmp_path)


def test_checksum_fails_closed_and_success(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"synthetic checkpoint bytes")
    assert verify_checksum(path, sha256(path)) == sha256(path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checksum(path, MODEL_SHA)


@pytest.mark.parametrize("bad_execution", [None, [], "cuda"])
def test_malformed_execution_fails_cleanly(
    tmp_path: Path, config_values: dict[str, Any], bad_execution: Any
) -> None:
    config_values["execution"] = bad_execution
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_values), encoding="utf-8")
    with pytest.raises(ValueError, match="execution"):
        load_config(path, tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/x.wav",
        "data/manifests/x.json",
        "data/splits/x.json",
        ".venv/a",
        "artifacts/module5/a",
        "src/vaaniswitch/m5_experiments/a",
        "artifacts/module6/../../data/raw/a",
        "uv.lock",
        "pyproject.toml",
    ],
)
def test_protected_output_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError):
        write_bytes(tmp_path, path, b"forbidden")
    assert list(tmp_path.iterdir()) == []


def test_serialization_and_immutable_publication(tmp_path: Path) -> None:
    value = {
        "device": "cuda",
        "dtype": "float16",
        "raw": "  Raw text.\n",
        "python": "3.12.2",
        "cuda": "12.6",
        "versions": {"pyttsx3": "2.99"},
    }
    path = write_json(tmp_path, "artifacts/module6/run/environment.json", value)
    assert json.loads(path.read_text()) == value
    with pytest.raises(FileExistsError):
        write_json(tmp_path, path, {"changed": True})
    with pytest.raises(ValueError):
        write_json(tmp_path, "artifacts/module6/run/nonfinite.json", {"bad": float("nan")})
    complete = complete_json(tmp_path, "artifacts/module6/run/COMPLETE.json", value)
    assert json.loads(complete.read_text()) == value
    with pytest.raises(FileExistsError):
        complete_json(tmp_path, complete, {"changed": True})
    assert json.loads(complete.read_text()) == value


def test_junction_cannot_redirect_outputs(tmp_path: Path) -> None:
    protected = tmp_path / "data/raw"
    protected.mkdir(parents=True)
    parent = tmp_path / "artifacts/module6"
    parent.mkdir(parents=True)
    junction = parent / "redirect"
    if os.name == "nt":
        command = f"New-Item -ItemType Junction -Path '{junction}' -Target '{protected}'"
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", command], check=True, capture_output=True
        )
    else:
        junction.symlink_to(protected, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="symlink/junction"):
            output_path(tmp_path, junction / "bad.wav")
        assert not (protected / "bad.wav").exists()
    finally:
        if os.name == "nt":
            junction.rmdir()
        else:
            junction.unlink()


def test_hardlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "raw.wav"
    source.write_bytes(b"protected")
    target = tmp_path / "artifacts/module6/run/audio.wav"
    target.parent.mkdir(parents=True)
    os.link(source, target)
    with pytest.raises(ValueError, match="hardlinked"):
        output_path(tmp_path, target)
    assert source.read_bytes() == b"protected"
