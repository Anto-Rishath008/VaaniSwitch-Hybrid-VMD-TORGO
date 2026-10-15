"""Read-only TORGO CORE identities, inherited partitions and signed RP loading."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import (FOLDS_SHA, MANIFEST_SHA, ORIGINAL, RP_HASH, VMD_HASH, digest,
                     progress, read, require, sha)


@dataclass(frozen=True)
class Record:
    logical_id: str
    speaker_id: str
    condition: str
    primary_audio_path: str
    primary_audio_sha256: str
    duration_seconds: float
    frame_count: int
    sample_rate: int
    primary_microphone: str

    @property
    def label(self):
        require(self.condition in {"control", "dysarthric"}, "Unknown frozen condition")
        return int(self.condition == "dysarthric")


def validate_microphone(row):
    if row["head_audio_status"] == "usable":
        expected = row["head_mic_path"], "head", "head_structurally_usable"
    else:
        require(row["array_audio_status"] == "usable", "Neither microphone usable")
        expected = (row["array_mic_path"], "array", "head_unusable_array_fallback"
                    if row["head_audio_status"] == "unusable"
                    else "array_structurally_usable_head_missing")
    require((row["primary_audio_path"], row["primary_microphone"],
             row["primary_selection_reason"]) == expected, "Primary microphone policy violated")


def safe_original(relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe source path")
    resolved = (ORIGINAL / path).resolve()
    require(resolved.is_relative_to(ORIGINAL.resolve()), "Source escapes original project")
    return resolved


class Corpus:
    def __init__(self):
        manifest = ORIGINAL / "data/manifests/torgo_manifest.jsonl"
        folds = ORIGINAL / "data/splits/speaker_folds.json"
        require(sha(manifest) == MANIFEST_SHA, "Frozen manifest changed")
        require(sha(folds) == FOLDS_SHA, "Frozen fold file changed")
        import json
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        require(len(rows) == 9494 and len({r["logical_id"] for r in rows}) == 9494,
                "Manifest identity/count mismatch")
        selected = [r for r in rows if r["normal_speech_eligible"]
                    and r["usability_status"] == "usable" and r["prompt_type"] in {"word", "sentence"}]
        records = []
        for row in selected:
            require(row["logical_id"] == "/".join(row[k] for k in
                    ("speaker_id", "session_id", "prompt_id")), "Logical ID mismatch")
            validate_microphone(row)
            safe_original(row["primary_audio_path"])
            records.append(Record(**{k: row[k] for k in Record.__dataclass_fields__}))
        self.records = tuple(sorted(records, key=lambda r: r.logical_id))
        self.by_id = {r.logical_id: r for r in records}
        self.folds = read(folds)
        self.mapping = self.folds["speaker_to_evaluation_fold"]
        require(len(records) == 8673 and len(self.mapping) == 15, "CORE count mismatch")
        require({r.speaker_id for r in records} == set(self.mapping), "Speaker mapping mismatch")
        signoff = read(ORIGINAL / "artifacts/metrics/module2_full_signoff_audit.json")
        require(signoff["status"] == "PASS" and signoff["module2_signed_off"] is True,
                "Missing RP signoff")
        require(signoff["identity"]["representation_config_hash"] == RP_HASH, "Wrong RP identity")
        require(signoff["completion"]["validated"] == len(records), "RP count mismatch")
        self.splits = tuple(Split(self, fold) for fold in (1, 2, 3))
        require([len(self.select(s.test)) for s in self.splits] == [2819, 2873, 2981],
                "Original outer record counts changed")
        require(sum(len(self.select(s.test)) for s in self.splits) == len(records), "Coverage mismatch")

    def select(self, speakers):
        return tuple(r for r in self.records if r.speaker_id in speakers)

    def audit(self):
        return {"manifest_sha256": MANIFEST_SHA, "folds_file_sha256": FOLDS_SHA,
                "folds_canonical_sha256": digest(self.folds), "core_count": len(self.records),
                "speakers": len(self.mapping), "rp_config_hash": RP_HASH, "vmd_config_hash": VMD_HASH,
                "primary_microphones": dict(Counter(r.primary_microphone for r in self.records)),
                "cropped_recordings": sum(r.frame_count > 128000 for r in self.records),
                "splits": [s.identity for s in self.splits]}


class Split:
    def __init__(self, corpus, fold):
        import hashlib
        self.fold = fold
        self.test = tuple(sorted(s for s, f in corpus.mapping.items() if f == fold))
        dev = set(corpus.mapping) - set(self.test)
        conditions = {r.speaker_id: r.condition for r in corpus.records}
        val = []
        for condition in ("control", "dysarthric"):
            candidates = [s for s in dev if conditions[s] == condition]
            val.append(min(candidates, key=lambda s: (hashlib.sha256(
                f"42|{fold}|{condition}|{s}".encode("ascii")).hexdigest(), s)))
        self.validation = tuple(sorted(val))
        self.train = tuple(sorted(dev - set(val)))
        self.development = tuple(sorted(dev))
        require([len(self.train), len(self.validation), len(self.test)] == [8, 2, 5], "Bad roles")
        roles = [corpus.select(s) for s in (self.train, self.validation, self.test)]
        for attribute in ("speaker_id", "logical_id", "primary_audio_path", "primary_audio_sha256"):
            groups = [{getattr(r, attribute) for r in role} for role in roles]
            require(all(not groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))),
                    f"Cross-role leakage: {attribute}")
        for role in roles:
            require({r.label for r in role} == {0, 1}, "A role lost a class")

    @property
    def identity(self):
        return {"outer_fold": self.fold, "inner_train": list(self.train),
                "inner_validation": list(self.validation), "outer_test": list(self.test)}


def guard_records(records, allowed, forbidden):
    require(records and all(r.speaker_id in allowed and r.speaker_id not in forbidden
                            for r in records), "Attempted access outside fitting role")
    require(len({r.logical_id for r in records}) == len(records), "Duplicate records")


def training_weights(records, allowed, forbidden):
    guard_records(records, allowed, forbidden)
    counts = Counter(r.speaker_id for r in records)
    # Equal total weight per training speaker. No class_weight or pos_weight is also applied.
    weights = np.array([1.0 / counts[r.speaker_id] for r in records], dtype=np.float64)
    return weights / weights.mean()


def rp_path(record):
    return safe_original(f"data/processed/representations/{RP_HASH}/{record.logical_id}/recurrence.npy")


def load_rp(record):
    path = rp_path(record)
    doc = read(path.parent / "metadata.json")
    require(doc.get("complete") is True and doc.get("logical_id") == record.logical_id,
            "Incomplete/misbound RP")
    require(doc.get("manifest_fingerprint") == MANIFEST_SHA
            and doc.get("module1_config_hash") == VMD_HASH
            and doc.get("representation_config_hash") == RP_HASH, "Stale RP provenance")
    require(digest(doc["processing_identity"]) == RP_HASH, "RP processing configuration changed")
    vmd = read(safe_original(f"data/processed/vmd/{VMD_HASH}/{record.logical_id}/metadata.json"))
    require(vmd.get("complete") is True and vmd["logical_id"] == record.logical_id,
            "VMD source metadata incomplete")
    require(vmd["source_audio_sha256"] == record.primary_audio_sha256
            and vmd["source_primary_audio_path"] == record.primary_audio_path
            and vmd["source_manifest_fingerprint"] == MANIFEST_SHA
            and vmd["config_hash"] == VMD_HASH, "VMD source identity does not match frozen primary")
    require(doc["source_reconstruction_path"] ==
            f"data/processed/vmd/{VMD_HASH}/{record.logical_id}/reconstruction.npy", "Wrong RP reconstruction path")
    require(doc["source_reconstruction_sha256"] == vmd["artifacts"]["reconstruction"]["sha256"],
            "RP/VMD source binding changed")
    source = doc["representation_metadata"]["source"]
    require(source["reconstruction_policy"] == "all_modes"
            and source["sample_rate"] == 16000, "Wrong RP source semantics")
    spec = doc["artifacts"]["recurrence"]
    require(spec["filename"] == "recurrence.npy" and sha(path) == spec["sha256"], "RP checksum mismatch")
    values = np.load(path, allow_pickle=False)
    require(values.shape == (224, 224) and values.dtype == np.uint8
            and np.isin(values, [0, 1]).all(), "Invalid binary RP")
    return values


def verify_inputs(records):
    import time
    started = time.perf_counter()
    for i, record in enumerate(records, 1):
        require(sha(safe_original(record.primary_audio_path)) == record.primary_audio_sha256,
                f"Primary audio changed: {record.logical_id}")
        load_rp(record)
        if i % 250 == 0 or i == len(records):
            progress("Primary audio + signed RP integrity", i, len(records), started)
    return {"status": "PASS", "selected_primary_and_rp_sha256_checked": len(records)}
