"""Generate offline demo audio samples for the VaaniSwitch demonstration."""

import math
import struct
import wave
from pathlib import Path

import numpy as np


def generate_speech_like_waveform(
    duration: float = 2.5,
    sample_rate: int = 16000,
    f0_mean: float = 130.0,
    dysarthric: bool = False,
) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if dysarthric:
        # Neuromotor instability: Tremor (4-6 Hz jitter/shimmer), irregular pitch modulation, breathiness
        f0_mod = f0_mean + 18.0 * np.sin(2 * np.pi * 5.2 * t) + 10.0 * np.sin(2 * np.pi * 2.1 * t)
        phase = 2 * np.pi * np.cumsum(f0_mod) / sample_rate
        glottal = (
            0.6 * np.sin(phase)
            + 0.3 * np.sin(2 * phase)
            + 0.2 * np.sin(3 * phase)
            + 0.15 * np.sin(4 * phase)
        )
        # Formant frequencies with articulatory blurring (vowel centralization)
        formants = [500.0, 1500.0, 2500.0]
        resonances = np.zeros_like(t)
        for f in formants:
            resonances += np.sin(2 * np.pi * f * t) * np.exp(-t % 0.08 * 30.0)
        
        envelope = 0.5 * (1.0 - np.cos(2 * np.pi * t / duration))
        envelope *= 1.0 + 0.35 * np.sin(2 * np.pi * 4.5 * t)  # Asthenia & tremor
        noise = np.random.normal(0, 0.08, size=t.shape)  # Breathiness noise
        signal = (0.6 * glottal + 0.3 * resonances + noise) * envelope
    else:
        # Stable control speech
        f0_mod = f0_mean + 5.0 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(f0_mod) / sample_rate
        glottal = (
            0.7 * np.sin(phase)
            + 0.35 * np.sin(2 * phase)
            + 0.2 * np.sin(3 * phase)
            + 0.1 * np.sin(4 * phase)
        )
        formants = [650.0, 1850.0, 2750.0]
        resonances = np.zeros_like(t)
        for f in formants:
            resonances += np.sin(2 * np.pi * f * t) * np.exp(-t % 0.05 * 40.0)
        
        envelope = 0.5 * (1.0 - np.cos(2 * np.pi * t / duration))
        signal = (0.7 * glottal + 0.3 * resonances) * envelope

    # RMS normalization to 0.1
    rms = np.sqrt(np.mean(signal**2))
    if rms > 0:
        signal = signal * (0.1 / rms)
    # Peak clamp
    signal = np.clip(signal, -0.95, 0.95)
    return signal


def save_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_samples.tobytes())


def main() -> None:
    sample_dir = Path(__file__).parent / "sample_audio"
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    # 1. Control sample
    control_audio = generate_speech_like_waveform(duration=2.2, dysarthric=False)
    save_wav(sample_dir / "control_sample_MC02.wav", control_audio)

    # 2. Dysarthric sample
    dys_audio = generate_speech_like_waveform(duration=2.5, dysarthric=True)
    save_wav(sample_dir / "dysarthric_sample_M03.wav", dys_audio)

    # 3. Clear reconstructed sample (crisp synthesis)
    clear_audio = generate_speech_like_waveform(duration=2.0, f0_mean=180.0, dysarthric=False)
    save_wav(sample_dir / "reconstructed_clear_speech.wav", clear_audio)

    print("Created demo samples successfully in", sample_dir)


if __name__ == "__main__":
    main()
