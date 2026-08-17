"""WAV header and signal-quality regression tests."""

from pathlib import Path

import pytest

from tests.data.helpers import make_config, write_wav
from components.data import audio_audit
from components.data.audio_audit import audit_wav


def test_valid_wav_header_and_duration(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    path = config.raw_root / "F/F01/Session1/wav_headMic/0001.wav"
    write_wav(path, duration_seconds=0.2)

    audit = audit_wav(path, config)

    assert audit.readable
    assert audit.sample_rate == 16_000
    assert audit.channels == 1
    assert audit.sample_width_bytes == 2
    assert audit.bit_depth == 16
    assert audit.frame_count == 3_200
    assert audit.duration_seconds == 0.2
    assert audit.sha256 is not None


def test_corrupt_wav_is_reported_not_raised(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    path = config.raw_root / "F/F01/Session1/wav_headMic/0001.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a wave file")

    audit = audit_wav(path, config)

    assert not audit.readable
    assert audit.error is not None and audit.error.startswith("wav_read_error")
    assert "unreadable" in audit.quality_flags


def test_truncated_wav_payload_is_reported_as_unreadable(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    path = config.raw_root / "F/F01/Session1/wav_headMic/0001.wav"
    write_wav(path, duration_seconds=0.2)
    payload = path.read_bytes()
    path.write_bytes(payload[:-100])

    audit = audit_wav(path, config)

    assert not audit.readable
    assert audit.error is not None and "truncated WAV payload" in audit.error


def test_zero_byte_wav_is_reported(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    path = config.raw_root / "F/F01/Session1/wav_headMic/0001.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")

    audit = audit_wav(path, config)

    assert audit.zero_byte
    assert not audit.readable
    assert audit.sha256 is not None


def test_hash_read_failure_is_reported_not_raised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    path = config.raw_root / "F/F01/Session1/wav_headMic/0001.wav"
    write_wav(path)

    def fail_hash(_path: Path) -> str:
        raise OSError("synthetic permission failure")

    monkeypatch.setattr(audio_audit, "_sha256", fail_hash)
    audit = audit_wav(path, config)

    assert not audit.readable
    assert audit.error is not None and audit.error.startswith("file_hash_error")


def test_silence_shortness_and_clipping_indicators(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    silent = config.raw_root / "F/F01/Session1/wav_headMic/0001.wav"
    clipped = config.raw_root / "F/F01/Session1/wav_headMic/0002.wav"
    write_wav(silent, samples=[0] * 100)
    write_wav(clipped, samples=[32_767] * 200)

    silent_audit = audit_wav(silent, config)
    clipped_audit = audit_wav(clipped, config)

    assert silent_audit.extremely_short
    assert silent_audit.near_silent
    assert clipped_audit.possible_clipping
    assert clipped_audit.clipped_sample_fraction == 1.0
