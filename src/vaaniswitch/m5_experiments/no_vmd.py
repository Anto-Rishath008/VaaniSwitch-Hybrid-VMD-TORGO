"""A0: signed primary -> signed conditioning -> signed Mel, with no VMD array reads."""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m1_vmd.audio import AudioLoadConfig, NormalizationConfig, standardize_audio
from vaaniswitch.m1_vmd.manifest import resolve_primary_audio
from vaaniswitch.m2_representations.config import Module2Config
from vaaniswitch.m2_representations.mel import generate_log_mel
from vaaniswitch.m5_experiments.config import canonical_path, digest, file_hash
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.source import ExperimentContext


class NoVMDCache:
    def __init__(self, context: ExperimentContext) -> None:
        self.context = context
        root = context.config.repo_root
        self.mel = Module2Config.from_yaml(
            root / "configs/representations/default.yaml", repo_root=root
        ).mel
        self.identity = {
            "implementation": "module5_conditioned_primary_no_vmd_mel_v1",
            "audio": asdict(context.audio_config.audio),
            "mel": asdict(self.mel),
            "manifest": context.predecessor.source.contract.manifest_fingerprint,
            "vmd": "bypassed",
            "primary": "signed_Module0_primary_only",
            "algorithms": {
                p: file_hash(root / p)
                for p in (
                    "src/vaaniswitch/m1_vmd/audio.py",
                    "src/vaaniswitch/m2_representations/mel.py",
                )
            },
        }
        self.config_hash = digest(self.identity)

    def directory(self, logical_id: str) -> Path:
        if logical_id not in self.context.predecessor.source.contract.record_by_id:
            raise ValueError("A0 accepts signed CORE logical IDs only")
        if any(p in {"", ".", ".."} for p in logical_id.split("/")):
            raise ValueError("invalid logical ID")
        return canonical_path(
            self.context.config.repo_root,
            f"data/processed/module5_ablation/no_vmd_mel/{self.config_hash}/{logical_id}",
        )

    def binding(self, record: ManifestRecord) -> dict[str, Any]:
        self._require_record(record)
        return {
            "logical_id": record.logical_id,
            "primary_audio_path": record.primary_audio_path,
            "primary_audio_sha256": record.primary_audio_sha256,
            "manifest_fingerprint": self.context.predecessor.source.contract.manifest_fingerprint,
            "config_hash": self.config_hash,
            "processing_identity": self.identity,
        }

    def load(self, record: ManifestRecord, *, check_source: bool = False) -> NDArray[np.float32]:
        directory = self.directory(record.logical_id)
        if {p.name for p in directory.iterdir()} != {"mel.npy", "metadata.json"}:
            raise ValueError("A0 cache is incomplete or has unexpected files")
        metadata = read_json(directory / "metadata.json")
        if (
            metadata.get("binding") != self.binding(record)
            or metadata.get("complete") is not True
            or metadata.get("mel_sha256") != file_hash(directory / "mel.npy")
        ):
            raise ValueError("A0 cache provenance/checksum mismatch")
        if check_source:
            self._primary(record)
        values = np.load(directory / "mel.npy", allow_pickle=False)
        if (
            values.shape != (224, 224)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError("A0 Mel shape/dtype/numerical contract failed")
        return np.asarray(values, dtype=np.float32)

    def _primary(self, record: ManifestRecord) -> Path:
        self._require_record(record)
        path = resolve_primary_audio(record, self.context.config.repo_root)
        if not path.is_relative_to(self.context.audio_config.dataset.raw_root):
            raise ValueError("A0 primary must be in signed raw TORGO")
        if file_hash(path) != record.primary_audio_sha256:
            raise ValueError("A0 selected primary checksum mismatch")
        return path

    def _require_record(self, record: ManifestRecord) -> None:
        if self.context.predecessor.source.contract.record_by_id.get(record.logical_id) != record:
            raise ValueError("A0 requires the exact signed record and selected primary")

    def build(self, record: ManifestRecord) -> str:
        directory = self.directory(record.logical_id)
        if directory.exists():
            # Invalid existing evidence is never silently replaced by a fresh transform.
            self.load(record, check_source=True)
            return "cache_hit"
        audio = self.context.audio_config.audio
        normalized = standardize_audio(
            self._primary(record),
            AudioLoadConfig(audio.target_sample_rate, audio.target_channels),
            NormalizationConfig(
                "rms", audio.target_rms, audio.max_gain, audio.peak_limit, audio.silence_epsilon
            ),
        )
        image = generate_log_mel(normalized.waveform, self.mel)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.partial")
        temporary.mkdir()
        np.save(temporary / "mel.npy", image.image, allow_pickle=False)
        write_json(
            temporary / "metadata.json",
            {
                "binding": self.binding(record),
                "complete": True,
                "mel_sha256": file_hash(temporary / "mel.npy"),
                "sample_count": len(normalized.waveform),
                "sample_rate": normalized.sample_rate,
                "normalization": asdict(normalized.normalization),
                "downmixed": normalized.downmixed,
                "resample_method": normalized.resample_method,
                "time_handling": image.time_handling,
                "padded_seconds": image.padded_seconds,
                "cropped_seconds": image.cropped_seconds,
            },
        )
        os.rename(temporary, directory)
        self.load(record)
        return "computed"
