"""Frozen Phase-1 policy; no reference or user prompt interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .artifacts import digest, output_path

MODEL_SHA = "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872"
MODEL_SOURCE = f"https://openaipublic.azureedge.net/main/whisper/models/{MODEL_SHA}/small.en.pt"
VOICE_ID = (
    r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens"
    r"\TTS_MS_EN-US_ZIRA_11.0"
)
RECORD_ID = "M03/Session2/0013"
DECODING: dict[str, Any] = {
    "task": "transcribe",
    "language": "en",
    "temperature": 0.0,
    "beam_size": None,
    "best_of": None,
    "initial_prompt": None,
    "prefix": None,
    "condition_on_previous_text": False,
    "carry_initial_prompt": False,
    "word_timestamps": False,
    "clip_timestamps": "0",
    "verbose": None,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "hallucination_silence_threshold": None,
    "patience": None,
    "length_penalty": None,
    "sample_len": None,
    "suppress_tokens": "-1",
    "suppress_blank": True,
    "without_timestamps": False,
    "max_initial_timestamp": 1.0,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",  # noqa: RUF001 -- exact Whisper default
}


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    values: dict[str, Any]

    @property
    def identity(self) -> str:
        return digest(self.values)

    @property
    def checkpoint(self) -> Path:
        return output_path(self.root, self.values["asr"]["checkpoint"])

    @property
    def device(self) -> str:
        return str(self.values["execution"]["device"])


def load_config(path: Path, root: Path) -> RuntimeConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    if not isinstance(value.get("execution"), dict):
        raise ValueError("execution must be a mapping")
    expected: dict[str, Any] = {
        "schema": "module6_phase1_v1",
        "seed": 42,
        "asr": {
            "backend": "openai-whisper",
            "package_version": "20250625",
            "model": "small.en",
            "checkpoint": "artifacts/module6/models/whisper/small.en.pt",
            "sha256": MODEL_SHA,
            "source": MODEL_SOURCE,
            "decoding": DECODING,
        },
        "execution": {
            "device": "cuda",
            "dtype": "float16",
            "cpu_threads": 8,
            "asr_timeout_seconds": 300,
            "cpu_fallback": {
                "device": "cpu",
                "dtype": "float32",
                "cpu_threads": 8,
                "automatic": False,
            },
        },
        "tts": {
            "backend": "pyttsx3",
            "package_version": "2.99",
            "driver": "sapi5",
            "voice_id": VOICE_ID,
            "rate": 175,
            "volume": 1.0,
            "timeout_seconds": 60,
            "smoke_text": "This is the VaaniSwitch English speech synthesis runtime test.",
        },
        "smoke": {"record_id": RECORD_ID, "metrics_enabled": False, "correction_enabled": False},
    }
    # CPU fallback requires an explicit versioned config and a documented failed CUDA attempt.
    if value.get("execution", {}).get("device") == "cpu":
        expected["execution"]["device"] = "cpu"
        expected["execution"]["dtype"] = "float32"
    if digest(value) != digest(expected):
        raise ValueError("configuration differs from the frozen Phase-1 schema/policy")
    result = RuntimeConfig(root.resolve(), value)
    _ = result.checkpoint
    return result
