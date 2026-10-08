"""Opt-in evidence checks; never downloads, invokes a model, or synthesizes in pytest."""

import json
import os
from pathlib import Path

import pytest

from vaaniswitch.m6_reconstruction.artifacts import sha256
from vaaniswitch.m6_reconstruction.audio import validate_generated_wav
from vaaniswitch.m6_reconstruction.config import MODEL_SHA, RECORD_ID, VOICE_ID


@pytest.mark.runtime
@pytest.mark.skipif(
    not os.environ.get("M6_RUNTIME_EVIDENCE"), reason="explicit runtime evidence opt-in"
)
def test_retained_real_runtime(project_root: Path) -> None:
    run = Path(os.environ["M6_RUNTIME_EVIDENCE"])
    complete = json.loads((run / "RUNTIME_COMPLETE.json").read_bytes())
    for relative, expected in complete["artifact_sha256"].items():
        assert sha256(run / relative) == expected
    asr = json.loads((run / "asr_smoke.json").read_bytes())
    assert asr["record_id"] == RECORD_ID
    assert asr["checkpoint_sha256"] == MODEL_SHA
    assert asr["raw_text"].encode("utf-8") == (run / "raw_asr.txt").read_bytes()
    assert asr["input_frames"] == asr["source"]["frame_count"]
    assert asr["elapsed_seconds"] > 0
    assert asr["reference_context_used"] is False
    if complete["status"] == "PASS":
        assert asr["device"] == "cuda"
        assert asr["model_probe"]["cuda_allocation"]
        assert asr["model_probe"]["cuda_available"]
        assert asr["model_probe"]["torch_cuda"]
        assert asr["model_probe"]["cuda_device_name"] == "NVIDIA GeForce RTX 4070 Laptop GPU"
        assert asr["model_probe"]["model_device"].startswith("cuda")
    tts = json.loads((run / "tts/synthesis.json").read_bytes())
    assert tts["selected_voice"]["id"] == VOICE_ID
    assert tts["validation"] == validate_generated_wav(project_root / tts["wav_path"])
    job = json.loads((run / "recognition_audio.json").read_bytes())
    assert set(job) == {"logical_id", "primary_audio_path", "primary_audio_sha256", "frame_count"}
    assert complete["evaluation_records"] == 0
    assert complete["wer_cer_computed"] is False
