"""Read-only production metadata and tiny synthetic WAV/cache contexts."""

from __future__ import annotations

import shutil
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.m4_fusion.helpers import MemorySource
from vaaniswitch.m3_encoders.source import SignedSourceContract
from vaaniswitch.m5_experiments.config import ExperimentConfig, file_hash
from vaaniswitch.m5_experiments.source import ExperimentContext, prepare_experiments
from vaaniswitch.m5_experiments.splits import nested_splits

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def real_context() -> ExperimentContext:
    config = ExperimentConfig.from_yaml(
        REPO / "configs/experiments/module5/protocol.yaml", repo_root=REPO
    )
    return prepare_experiments(config)


@pytest.fixture
def context(tmp_path: Path, real_context: ExperimentContext) -> ExperimentContext:
    torch.set_num_threads(2)
    root = tmp_path.resolve()
    for relative in (
        "configs/representations/default.yaml",
        "configs/experiments/module5/protocol.yaml",
        "src/vaaniswitch/m1_vmd/audio.py",
        "src/vaaniswitch/m2_representations/mel.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, path)
    predecessor = real_context.predecessor
    contract = predecessor.source.contract
    records = []
    for index, speaker in enumerate(sorted(contract.speaker_to_evaluation_fold)):
        original = next(r for r in contract.records if r.speaker_id == speaker)
        relative = f"data/raw/torgo/extracted/{speaker}/Session1/wav_headMic/0001.wav"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        signal = (np.sin(np.arange(3200) * (0.02 + index * 0.003)) * 10000).astype("<i2")
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(signal.tobytes())
        records.append(
            replace(
                original,
                primary_audio_path=relative,
                primary_audio_sha256=file_hash(path),
                duration_seconds=0.2,
                frame_count=3200,
                sample_rate=16000,
            )
        )
    synthetic = SignedSourceContract(
        tuple(records),
        {r.logical_id: r for r in records},
        contract.speaker_to_evaluation_fold.copy(),
        contract.manifest_fingerprint,
        contract.representation_config_hash,
        contract.module1_config_hash,
        len(records),
    )
    encoder = replace(predecessor.encoder_config, repo_root=root)
    source = MemorySource(encoder, synthetic)
    source.forbidden_speakers = set(real_context.split(1).speakers("outer_test"))
    m1 = replace(
        real_context.audio_config,
        repo_root=root,
        dataset=replace(
            real_context.audio_config.dataset, raw_root=root / "data/raw/torgo/extracted"
        ),
    )
    return ExperimentContext(
        ExperimentConfig(root),
        replace(predecessor, source=source, encoder_config=encoder),
        nested_splits(synthetic, real_context.split(1).inner.folds_fingerprint),
        m1,
    )
