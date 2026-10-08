"""Synthetic signed contracts and an in-memory source with an outer-test read sentinel."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.m0_data.helpers import balanced_speaker_records, make_config
from tests.m3_encoders.helpers import make_model_config, write_module3_yaml
from vaaniswitch.m0_data.leakage import speaker_folds_fingerprint
from vaaniswitch.m0_data.splits import create_speaker_folds
from vaaniswitch.m2_representations.cache import CacheEntry
from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.source import SignedRepresentationSource, SignedSourceContract
from vaaniswitch.m4_fusion.config import (
    ARCHITECTURE,
    FusionSourceConfig,
    Module4Config,
    TrainingConfig,
)
from vaaniswitch.m4_fusion.model import DysarthriaClassifier
from vaaniswitch.m4_fusion.reporting import write_json
from vaaniswitch.m4_fusion.source import DevelopmentContext
from vaaniswitch.m4_fusion.splits import create_inner_split

REPO = Path(__file__).resolve().parents[2]


class MemorySource(SignedRepresentationSource):
    def __init__(self, config: Module3Config, contract: SignedSourceContract) -> None:
        self.config = config
        self.contract = contract
        self.loaded: list[str] = []
        self.forbidden_speakers: set[str] = set()

    def load(self, logical_id: str) -> CacheEntry:
        record = self.contract.record_by_id[logical_id]
        assert record.speaker_id not in self.forbidden_speakers, "reserved test data loaded"
        self.loaded.append(logical_id)
        positive = record.condition.value == "dysarthric"
        mel = np.linspace(-100, 0, 224 * 224, dtype=np.float32).reshape(224, 224)
        mel = mel if positive else np.flip(mel, axis=0).copy()
        recurrence = (
            np.eye(224, dtype=np.uint8) if positive else np.ones((224, 224), dtype=np.uint8)
        )
        return CacheEntry(
            self.config.repo_root / "data/processed/test",
            mel,
            recurrence,
            np.array([[0, -1], [0.5, 0], [1, 1]], dtype=np.float32),
            np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int16),
            {},
        )


def make_context(root: Path) -> DevelopmentContext:
    encoder = make_model_config(root)
    records = tuple(balanced_speaker_records())
    folds = create_speaker_folds(
        records, make_config(root), manifest_sha256=encoder.source.manifest_fingerprint
    )
    write_json(encoder.source.speaker_folds_path, folds)
    contract = SignedSourceContract(
        records,
        {r.logical_id: r for r in records},
        folds["speaker_to_evaluation_fold"],
        encoder.source.manifest_fingerprint,
        encoder.source.representation_config_hash,
        encoder.source.module1_config_hash,
        len(records),
    )
    source = MemorySource(encoder, contract)
    config = Module4Config(
        root.resolve(),
        FusionSourceConfig(
            root / "configs/models/encoders.yaml",
            root / "artifacts/metrics/module3_signoff.json",
            encoder.config_hash,
            contract.representation_config_hash,
            contract.module1_config_hash,
            contract.manifest_fingerprint,
            len(records),
        ),
        TrainingConfig(),
    )
    write_module3_yaml(encoder, config.source.encoder_config_path)
    write_json(
        config.source.module3_signoff_path,
        {
            "status": "PASS",
            "module3_signed_off": True,
            "configuration": {
                "encoder_config_hash": encoder.config_hash,
                "token_dimension": 128,
                "gcn_hidden_dimension": 96,
                "recurrence_channels": [16, 32, 64],
            },
            "source": {
                "signed_core_records": len(records),
                **{k: v for k, v in config.source.hashes.items() if k != "encoder_config_hash"},
            },
        },
    )
    split = create_inner_split(
        contract, outer_test_fold=1, seed=42, folds_fingerprint=speaker_folds_fingerprint(folds)
    )
    source.forbidden_speakers = set(split.test_speakers)
    return DevelopmentContext(config, encoder, source, split)


def write_fusion_config(config: Module4Config) -> Path:
    source = asdict(config.source)
    for key in ("encoder_config_path", "module3_signoff_path"):
        source[key] = source[key].relative_to(config.repo_root).as_posix()
    raw = {
        "schema_version": "1.0",
        "implementation": "vaaniswitch_pytorch_fusion_v1",
        "source": source,
        **ARCHITECTURE,
        "training": asdict(config.training),
        "execution": {"cpu_threads": config.cpu_threads},
    }
    path = config.repo_root / "configs/models/fusion.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def install_memory_source(monkeypatch: pytest.MonkeyPatch, context: DevelopmentContext) -> None:
    def source_factory(config: Module3Config) -> SignedRepresentationSource:
        assert config.config_hash == context.encoder_config.config_hash
        return context.source

    def model_factory(
        current: DevelopmentContext, *, load_pretrained: bool
    ) -> DysarthriaClassifier:
        return DysarthriaClassifier(current.encoder_config, load_pretrained=False)

    monkeypatch.setattr("vaaniswitch.m4_fusion.source.SignedRepresentationSource", source_factory)
    monkeypatch.setattr("vaaniswitch.m4_fusion.pipeline.build_model", model_factory)
    monkeypatch.setattr("vaaniswitch.m4_fusion.diagnostics.build_model", model_factory)


def shorter(context: DevelopmentContext, epochs: int = 2, patience: int = 1) -> DevelopmentContext:
    return replace(
        context,
        config=replace(
            context.config,
            training=replace(
                context.config.training,
                max_epochs=epochs,
                patience=patience,
            ),
        ),
    )


def mutate_yaml(path: Path, keys: tuple[str, ...], value: Any) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = raw
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
