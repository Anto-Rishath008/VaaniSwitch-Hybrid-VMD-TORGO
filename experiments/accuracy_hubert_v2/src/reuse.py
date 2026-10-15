"""Verified read-only V1 feature/model references; never extract or repair V1 entries."""
from pathlib import Path

from .common import ROOT, digest, read, require, sha, write_json, progress
from .features import FeatureCache, extraction_identity, validate_cache

V1 = ROOT.with_name("accuracy_hubert")


class ReadOnlyFeatureCache(FeatureCache):
    def save(self, *args, **kwargs):
        raise RuntimeError("Read-only V1 feature reference: writes and repairs forbidden")

    def load(self, record):
        values = super().load(record)
        require(values is not None, f"Missing V1 feature; no automatic repair: {record.logical_id}")
        return values


def verify_lineage():
    lineage = read(ROOT/"V1_LINEAGE.json")
    for name, checksum in lineage["pinned_artifacts"].items():
        require(sha(V1/name) == checksum, f"V1 lineage artifact changed: {name}")
    frozen = read(V1/"logs/frozen_run.json")
    for name, checksum in frozen["source_files"].items():
        require(sha(V1/name) == checksum, f"V1 frozen source changed: {name}")
    require(not any(p.is_file() for p in (V1/"results").rglob("*")),
            "V1 has outer/result artifacts; review lineage before proceeding")
    return frozen


def acquire_readonly(cfg):
    frozen = verify_lineage()
    require(cfg["features"] == frozen["config"]["features"], "V1 preprocessing identity differs")
    lock, spec = frozen["model_lock"], cfg["features"]
    require(lock["model"] == spec["model"] and lock["revision"] == spec["revision"],
            "V1 checkpoint revision differs")
    directory = V1/"cache/checkpoint"/spec["revision"]
    require(read(directory/"verified_model.json") == lock, "V1 model lock differs")
    pins = {spec["weight_file"]: spec["weight_sha256"], "config.json": spec["config_sha256"],
            "preprocessor_config.json": spec["preprocessor_sha256"]}
    for name, checksum in pins.items():
        require(lock["files"][name] == checksum and sha(directory/name) == checksum,
                f"V1 pinned checkpoint configuration checksum mismatch: {name}")
    return directory, lock


def reference_cache(cfg, lock):
    return ReadOnlyFeatureCache(V1/"cache/features", extraction_identity(cfg, lock))


def verify_reuse(corpus, cfg, cache):
    progress("Verifying all V1 feature bytes, logical identities and preprocessing bindings")
    require(len(corpus.records) == len({r.logical_id for r in corpus.records}) == 8673,
            "Expected all 8673 logical identities")
    frozen = verify_lineage()
    require(cache.identity == extraction_identity(frozen["config"], frozen["model_lock"]),
            "Feature implementation/dependency identity changed")
    expected_ids = {r.logical_id for r in corpus.records}
    actual_ids = {p.parent.relative_to(cache.root).as_posix()
                  for p in cache.root.rglob("complete.json")}
    require(actual_ids == expected_ids, "V1 cache logical identity inventory mismatch")
    validation = validate_cache(corpus.records, cache)
    require(validation == read(V1/"logs/feature_validation.json"),
            "V1 feature validation bindings changed")
    for record in corpus.records:
        require(record.sample_rate == 16000, "Reassess segment identity for resampled data")
        n = record.frame_count
        start = max(0, (n-128000)//2)
        end = min(n, start+128000)
        expected = {"source_samples_16k": n, "start_sample_16k": start, "end_sample_16k": end,
                    "cropped": n > 128000, "valid_samples": end-start, "right_padding_samples": 0,
                    "source_rate": 16000, "target_rate": 16000}
        require(read(cache.location(record)/"complete.json")["segment"] == expected,
                f"Feature preprocessing interval differs: {record.logical_id}")
    report = {"status": "PASS", "record_count": len(expected_ids), "newly_extracted": 0,
              "cache_hits": len(expected_ids), "access": "read-only references; save forbidden",
              "cache_root": str(cache.root), "cache_identity": cache.identity,
              "validation_sha256": sha(V1/"logs/feature_validation.json"),
              "logical_ids_sha256": digest(sorted(expected_ids)),
              "cropped_recordings": validation["cropped_recordings"]}
    path = ROOT/"logs/feature_reuse.json"
    if path.exists():
        require(read(path) == report, "V2 reuse evidence changed")
    else:
        write_json(path, report)
    return report, [path]
