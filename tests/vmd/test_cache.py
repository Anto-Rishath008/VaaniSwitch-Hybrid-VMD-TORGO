"""Regression tests for the portable, resumable Module 1 cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from components.vmd.cache import (
    METADATA_FILENAME,
    MODES_FILENAME,
    RECONSTRUCTION_FILENAME,
    CacheIdentity,
    CacheValidationError,
    VMDCache,
    sha256_file,
)


def _identity(*, config_hash: str = "a" * 64) -> CacheIdentity:
    return CacheIdentity(
        logical_id="F01/Session1/0067",
        source_primary_audio_path=(
            "data/raw/torgo/extracted/F/F01/Session1/wav_arrayMic/0067.wav"
        ),
        source_audio_sha256="b" * 64,
        config_hash=config_hash,
        source_manifest_fingerprint="c" * 64,
        modes_k=2,
        sample_count=8,
    )


def _arrays() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    modes = np.vstack(
        (
            np.linspace(-0.5, 0.5, 8, dtype=np.float64),
            np.linspace(0.25, -0.25, 8, dtype=np.float64),
        )
    )
    return modes, modes.sum(axis=0)


def _write_valid(cache: VMDCache, identity: CacheIdentity) -> None:
    modes, reconstruction = _arrays()
    cache.write(
        identity,
        modes,
        reconstruction,
        {
            "mode_metadata": [{"mode_index": np.int64(1), "relative_energy": 0.75}],
            "reconstruction_metrics": {"mse": np.float64(1.0e-12)},
        },
    )


def test_cache_path_is_deterministic_and_logical_id_derived(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "data" / "processed" / "vmd")
    identity = _identity()

    assert cache.path_for(identity) == (
        tmp_path
        / "data"
        / "processed"
        / "vmd"
        / identity.config_hash
        / "F01"
        / "Session1"
        / "0067"
    )


def test_cache_round_trip_preserves_arrays_identity_metadata_and_hashes(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    modes, reconstruction = _arrays()

    written = cache.write(
        identity,
        modes,
        reconstruction,
        {
            "mode_metadata": [{"mode_index": np.int64(1), "relative_energy": 0.75}],
            "reconstruction_metrics": {"mse": np.float64(1.0e-12)},
        },
    )
    loaded = cache.load(identity)

    assert written.identity == identity
    assert loaded.identity == identity
    np.testing.assert_array_equal(loaded.modes, modes)
    np.testing.assert_array_equal(loaded.reconstruction, reconstruction)
    assert loaded.metadata == {
        "mode_metadata": [{"mode_index": 1, "relative_energy": 0.75}],
        "reconstruction_metrics": {"mse": 1.0e-12},
    }
    document = json.loads(
        (loaded.directory / METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert document["logical_id"] == "F01/Session1/0067"
    assert document["source_primary_audio_path"].endswith("wav_arrayMic/0067.wav")
    assert document["source_manifest_fingerprint"] == "c" * 64
    assert document["config_hash"] == "a" * 64
    assert document["artifacts"]["modes"]["sha256"] == sha256_file(
        loaded.directory / MODES_FILENAME
    )
    assert document["artifacts"]["reconstruction"]["sha256"] == sha256_file(
        loaded.directory / RECONSTRUCTION_FILENAME
    )
    assert str(tmp_path) not in (loaded.directory / METADATA_FILENAME).read_text(
        encoding="utf-8"
    )


def test_valid_cache_can_be_detected_for_resume_skip(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    assert not cache.is_valid(identity)

    _write_valid(cache, identity)

    result = cache.validate(identity)
    assert result.valid
    assert result.reason is None
    assert cache.is_valid(identity)


def test_partial_cache_without_completion_metadata_is_rejected(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    directory = cache.path_for(identity)
    directory.mkdir(parents=True)
    modes, _ = _arrays()
    np.save(directory / MODES_FILENAME, modes, allow_pickle=False)

    result = cache.validate(identity)

    assert not result.valid
    assert result.reason is not None
    assert "completion marker is missing" in result.reason
    with pytest.raises(CacheValidationError, match="completion marker is missing"):
        cache.load(identity)


def test_corrupted_array_is_rejected_by_file_hash(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    _write_valid(cache, identity)
    modes_path = cache.path_for(identity) / MODES_FILENAME
    content = bytearray(modes_path.read_bytes())
    content[-1] ^= 1
    modes_path.write_bytes(content)

    result = cache.validate(identity)

    assert not result.valid
    assert result.reason == f"cache artifact SHA-256 mismatch: {MODES_FILENAME}"


def test_missing_and_wrong_sized_array_artifacts_are_rejected(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    _write_valid(cache, identity)
    reconstruction_path = cache.path_for(identity) / RECONSTRUCTION_FILENAME
    original = reconstruction_path.read_bytes()
    reconstruction_path.unlink()
    assert cache.validate(identity).reason == (
        f"cache artifact is missing: {RECONSTRUCTION_FILENAME}"
    )

    reconstruction_path.write_bytes(original[:-1])
    assert cache.validate(identity).reason == (
        f"cache artifact size mismatch: {RECONSTRUCTION_FILENAME}"
    )


def test_metadata_shape_tampering_is_rejected(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    _write_valid(cache, identity)
    metadata_path = cache.path_for(identity) / METADATA_FILENAME
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["artifacts"]["modes"]["shape"] = [1, 8]
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    result = cache.validate(identity)

    assert not result.valid
    assert result.reason is not None
    assert "stored modes.npy shape" in result.reason


def test_cache_metadata_schema_and_identity_tampering_are_rejected(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    _write_valid(cache, identity)
    metadata_path = cache.path_for(identity) / METADATA_FILENAME
    original = json.loads(metadata_path.read_text(encoding="utf-8"))

    changed = dict(original)
    changed["unexpected"] = True
    metadata_path.write_text(json.dumps(changed), encoding="utf-8")
    schema_result = cache.validate(identity)
    assert schema_result.reason is not None
    assert "cache metadata fields do not match schema" in schema_result.reason

    changed = dict(original)
    changed["logical_id"] = "F01/Session1/9999"
    metadata_path.write_text(json.dumps(changed), encoding="utf-8")
    assert cache.validate(identity).reason == "cache identity does not match expected inputs"


def test_config_change_uses_a_different_cache_identity_and_path(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    first = _identity(config_hash="a" * 64)
    second = _identity(config_hash="d" * 64)
    _write_valid(cache, first)

    assert cache.path_for(first) != cache.path_for(second)
    assert cache.is_valid(first)
    assert not cache.is_valid(second)


@pytest.mark.parametrize(
    ("modes_change", "reconstruction_change", "message"),
    [
        (lambda modes: modes[:1], lambda reconstruction: reconstruction, "modes shape"),
        (
            lambda modes: modes,
            lambda reconstruction: reconstruction[:-1],
            "reconstruction shape",
        ),
        (
            lambda modes: np.full_like(modes, np.nan),
            lambda reconstruction: reconstruction,
            "modes contain non-finite",
        ),
        (
            lambda modes: modes,
            lambda reconstruction: np.full_like(reconstruction, np.inf),
            "reconstruction contains non-finite",
        ),
    ],
)
def test_write_rejects_wrong_shapes_and_nonfinite_values(
    tmp_path: Path,
    modes_change: object,
    reconstruction_change: object,
    message: str,
) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    modes, reconstruction = _arrays()
    change_modes = modes_change
    change_reconstruction = reconstruction_change
    assert callable(change_modes)
    assert callable(change_reconstruction)

    with pytest.raises(CacheValidationError, match=message):
        cache.write(
            identity,
            change_modes(modes),
            change_reconstruction(reconstruction),
            {},
        )


@pytest.mark.parametrize(
    "identity",
    [
        CacheIdentity(
            logical_id="../Session1/0067",
            source_primary_audio_path="data/raw/audio.wav",
            source_audio_sha256="b" * 64,
            config_hash="a" * 64,
            source_manifest_fingerprint="c" * 64,
            modes_k=2,
            sample_count=8,
        ),
        CacheIdentity(
            logical_id="F01/Session1/0067",
            source_primary_audio_path="C:/Users/example/audio.wav",
            source_audio_sha256="b" * 64,
            config_hash="a" * 64,
            source_manifest_fingerprint="c" * 64,
            modes_k=2,
            sample_count=8,
        ),
    ],
)
def test_cache_identity_rejects_unsafe_paths(identity: CacheIdentity, tmp_path: Path) -> None:
    with pytest.raises(CacheValidationError):
        VMDCache(tmp_path / "vmd").path_for(identity)


def test_custom_metadata_rejects_absolute_paths(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    modes, reconstruction = _arrays()

    with pytest.raises(CacheValidationError, match="non-portable absolute path"):
        cache.write(
            _identity(),
            modes,
            reconstruction,
            {"diagnostic_path": "C:/Users/example/result.txt"},
        )


def test_atomic_replacements_publish_metadata_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replaced_destinations: list[str] = []
    original_replace = os.replace

    def tracking_replace(source: str | Path, destination: str | Path) -> None:
        replaced_destinations.append(Path(destination).name)
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", tracking_replace)
    cache = VMDCache(tmp_path / "vmd")
    _write_valid(cache, _identity())

    assert replaced_destinations == [
        MODES_FILENAME,
        RECONSTRUCTION_FILENAME,
        METADATA_FILENAME,
    ]


def test_invalid_completion_marker_and_malformed_json_are_rejected(tmp_path: Path) -> None:
    cache = VMDCache(tmp_path / "vmd")
    identity = _identity()
    _write_valid(cache, identity)
    metadata_path = cache.path_for(identity) / METADATA_FILENAME
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["complete"] = False
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    assert cache.validate(identity).reason == "cache metadata is not marked complete"

    metadata_path.write_text("{not-json", encoding="utf-8")
    result = cache.validate(identity)
    assert not result.valid
    assert result.reason is not None
    assert "cannot read cache metadata" in result.reason
