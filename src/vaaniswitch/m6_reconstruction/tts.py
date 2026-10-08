"""Bounded SAPI worker; the required voice is checked before saving any audio."""

from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from typing import Any

from .artifacts import output_path, sha256, write_bytes, write_json
from .audio import validate_generated_wav
from .config import VOICE_ID, RuntimeConfig


def select_required_voice(voices: list[Any]) -> Any:
    matches = [voice for voice in voices if voice.id == VOICE_ID]
    if len(matches) != 1 or "Zira" not in matches[0].name:
        raise ValueError("required exact Microsoft Zira voice unavailable or ambiguous")
    return matches[0]


def voice_record(voice: Any) -> dict[str, Any]:
    return {
        "id": voice.id,
        "name": voice.name,
        "languages": [
            v.decode("utf-8", errors="backslashreplace") if isinstance(v, bytes) else str(v)
            for v in voice.languages
        ],
        "gender": voice.gender,
        "age": voice.age,
    }


def synthesize_smoke(config: RuntimeConfig, run: Path) -> dict[str, Any]:
    import pyttsx3  # type: ignore[import-untyped]

    if importlib.metadata.version("pyttsx3") != "2.99":
        raise ValueError("pyttsx3 package version drift")
    settings = config.values["tts"]
    started = time.perf_counter()
    engine = pyttsx3.init(driverName="sapi5")
    voices = engine.getProperty("voices")
    selected = select_required_voice(voices)
    engine.setProperty("voice", VOICE_ID)
    engine.setProperty("rate", settings["rate"])
    engine.setProperty("volume", settings["volume"])
    # pyttsx3 2.99 leaves the proxy idle after runAndWait. An immediate synchronous
    # save then finishes before the next driver loop starts, stranding that loop.
    # Flush settings through the public external loop, then end it to restore the
    # queue barrier before save_to_file + runAndWait. Never patch the installed driver.
    engine.startLoop(False)
    try:
        engine.iterate()
    finally:
        engine.endLoop()
    actual = {
        "voice_id": engine.getProperty("voice"),
        "rate": engine.getProperty("rate"),
        "volume": engine.getProperty("volume"),
    }
    if actual != {"voice_id": VOICE_ID, "rate": 175, "volume": 1.0}:
        raise ValueError(f"SAPI settings mismatch: {actual}")
    write_json(
        config.root,
        run / "tts/voice.json",
        {
            "selected": voice_record(selected),
            "actual": actual,
            "enumerated": [voice_record(v) for v in voices],
        },
    )
    text = settings["smoke_text"]
    write_bytes(config.root, run / "tts/synthesis_text.txt", text.encode("utf-8"))
    temporary = output_path(config.root, run / "tts/clear.pending.wav")
    final = output_path(config.root, run / "tts/clear.wav")
    if temporary.exists() or final.exists():
        raise FileExistsError("TTS output already exists")
    engine.save_to_file(text, str(temporary))
    engine.runAndWait()
    engine.stop()
    metadata = validate_generated_wav(temporary)
    temporary.rename(final)
    result = {
        "backend": "pyttsx3",
        "package_version": "2.99",
        "driver": "sapi5",
        "selected_voice": voice_record(selected),
        "actual_settings": actual,
        "synthesis_text": text,
        "wav_path": final.relative_to(config.root).as_posix(),
        "wav_sha256": sha256(final),
        "validation": metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "listening_check": "NOT PERFORMED",
    }
    write_json(config.root, run / "tts/synthesis.json", result)
    return result
