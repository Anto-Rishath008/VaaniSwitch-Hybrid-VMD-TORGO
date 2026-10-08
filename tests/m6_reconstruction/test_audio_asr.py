import inspect
import io
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from vaaniswitch.m6_reconstruction.asr import transcribe_audio
from vaaniswitch.m6_reconstruction.audio import decode_pcm, validate_generated_wav
from vaaniswitch.m6_reconstruction.config import DECODING


def wav_bytes(
    frames: int = 1600, rate: int = 16000, channels: int = 1, width: int = 2, silent: bool = False
) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((channels, width, rate, frames, "NONE", "not compressed"))
        wav.writeframes((b"\0" if silent else b"\1") * frames * channels * width)
    return stream.getvalue()


def test_complete_audio_beyond_30_seconds() -> None:
    audio, metadata = decode_pcm(wav_bytes(frames=16000 * 31 + 17), source=True)
    assert len(audio) == 16000 * 31 + 17
    assert audio.dtype == np.float32
    assert metadata["duration_seconds"] > 31


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not wave",
        wav_bytes()[:-10],
        wav_bytes(frames=0),
        wav_bytes(rate=8000),
        wav_bytes(channels=2),
        wav_bytes(width=1),
    ],
)
def test_bad_or_unsupported_source(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_pcm(payload, source=True)


@pytest.mark.parametrize("payload", [b"", b"corrupt", wav_bytes(frames=0), wav_bytes(silent=True)])
def test_invalid_generated_audio(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "clear.wav"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        validate_generated_wav(path)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_generated_pcm_metadata(width: int) -> None:
    _, meta = decode_pcm(wav_bytes(width=width, channels=2, rate=22050), source=False)
    assert meta["nonzero_samples"] > 0
    assert meta["finite_samples"]
    assert meta["bit_depth"] == width * 8
    assert meta["channels"] == 2


def test_audio_only_invocation_and_exact_raw() -> None:
    import torch

    calls = []

    class Backend:
        device = "cpu"

        def transcribe(self, audio: NDArray[np.float32], **kwargs: Any) -> dict[str, Any]:
            assert torch.is_inference_mode_enabled()
            calls.append((len(audio), kwargs))
            return {"text": "  Exact RAW output.\n", "segments": [], "language": "en"}

    result = transcribe_audio(Backend(), np.ones(32, dtype=np.float32), device="cpu")
    assert result["raw_text"] == "  Exact RAW output.\n"
    assert result["device"] == "cpu" and result["dtype"] == "float32"
    assert result["elapsed_seconds"] >= 0
    assert result["reference_context_used"] is False
    assert calls == [(32, {**DECODING, "fp16": False})]
    assert set(inspect.signature(transcribe_audio).parameters) == {"model", "audio", "device"}
    with pytest.raises(TypeError):
        transcribe_audio(
            Backend(), np.ones(32, dtype=np.float32), device="cpu", reference="REFERENCE_SENTINEL"
        )  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="device"):
        transcribe_audio(Backend(), np.ones(32, dtype=np.float32), device="cuda")
