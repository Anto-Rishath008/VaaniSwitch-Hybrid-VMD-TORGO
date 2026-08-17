"""Reproducible final audit of the completed Module 1 production cache.

This script is intentionally read-only with respect to TORGO and the VMD cache.  It
writes only compact sign-off evidence beneath ``artifacts/``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vaaniswitch.m1_vmd.audio import (
    AudioLoadConfig,
    NormalizationConfig,
    standardize_audio,
)
from vaaniswitch.m1_vmd.cache import CacheIdentity, VMDCache
from vaaniswitch.m1_vmd.config import Module1Config
from vaaniswitch.m1_vmd.manifest import (
    load_manifest_source,
    resolve_primary_audio,
    select_subset,
)
from vaaniswitch.m1_vmd.metrics import compute_reconstruction_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/vmd/default.yaml"
SUMMARY_PATH = REPO_ROOT / "artifacts/metrics/module1_vmd_summary.json"
AUDIT_PATH = REPO_ROOT / "artifacts/metrics/module1_full_signoff_audit.json"
NONCONVERGED_PATH = REPO_ROOT / "artifacts/logs/module1_nonconverged_records.csv"
CACHE_INVENTORY_PATH = REPO_ROOT / "artifacts/logs/module1_cache_inventory.json"
EXPECTED_CACHE_FILES = {"metadata.json", "modes.npy", "reconstruction.npy"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _inventory_namespace(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    names = Counter(path.name for path in files)
    leaf_directories = sorted(
        path.parent.relative_to(root).as_posix()
        for path in files
        if path.name == "metadata.json"
    )
    unexpected = sorted(
        path.relative_to(root).as_posix()
        for path in files
        if path.name not in EXPECTED_CACHE_FILES
    )
    temporary = sorted(
        path.relative_to(root).as_posix()
        for path in files
        if ".tmp" in path.name.lower() or path.name.lower().endswith(".partial")
    )
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "root": root.relative_to(REPO_ROOT).as_posix(),
        "logical_cache_directories": len(leaf_directories),
        "modes_npy_count": names["modes.npy"],
        "reconstruction_npy_count": names["reconstruction.npy"],
        "metadata_json_count": names["metadata.json"],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024**3),
        "temporary_files": temporary,
        "unexpected_files": unexpected,
    }


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13):
        raise AssertionError(f"{label} differs: {actual!r} != {expected!r}")


def main() -> None:
    config = Module1Config.from_yaml(CONFIG_PATH, repo_root=REPO_ROOT)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    source = load_manifest_source(config.dataset.manifest_path)
    core_records = select_subset(source.records, "core")
    records_by_id = {record.logical_id: record for record in core_records}
    result_by_id = {record["logical_id"]: record for record in summary["records"]}

    expected_ids = set(records_by_id)
    result_ids = set(result_by_id)
    if len(core_records) != 8673 or expected_ids != result_ids:
        raise AssertionError("summary record IDs do not exactly match the CORE manifest subset")

    production_root = config.cache.root / config.config_hash
    inventory = _inventory_namespace(production_root)
    if inventory["logical_cache_directories"] != len(core_records):
        raise AssertionError("production cache directory count does not match CORE")
    if any(inventory[key] != len(core_records) for key in (
        "modes_npy_count",
        "reconstruction_npy_count",
        "metadata_json_count",
    )):
        raise AssertionError("production cache component counts are incomplete")
    if inventory["temporary_files"] or inventory["unexpected_files"]:
        raise AssertionError("production cache contains temporary or unexpected files")

    cache = VMDCache(config.cache.root)
    nonconverged_rows: list[dict[str, Any]] = []
    bound_primary_count = 0
    parameter_mismatches: list[str] = []
    incomplete_directories: list[str] = []
    expected_vmd = {
        "modes_k": config.vmd.modes_k,
        "alpha": float(config.vmd.alpha),
        "tau": float(config.vmd.tau),
        "dc": config.vmd.dc,
        "init": config.vmd.init,
        "tolerance": float(config.vmd.tolerance),
        "max_iterations": config.vmd.max_iterations,
    }

    for logical_id, record in records_by_id.items():
        if record.primary_audio_path is None or record.primary_audio_sha256 is None:
            raise AssertionError(f"CORE record lacks primary identity: {logical_id}")
        sample_count = int(record.frame_count or 0)
        identity = CacheIdentity(
            logical_id=logical_id,
            source_primary_audio_path=record.primary_audio_path,
            source_audio_sha256=record.primary_audio_sha256,
            config_hash=config.config_hash,
            source_manifest_fingerprint=source.fingerprint,
            modes_k=config.vmd.modes_k,
            sample_count=sample_count,
        )
        directory = cache.path_for(identity)
        if {path.name for path in directory.iterdir() if path.is_file()} != EXPECTED_CACHE_FILES:
            incomplete_directories.append(directory.relative_to(REPO_ROOT).as_posix())
            continue
        document = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (
            document.get("logical_id") != logical_id
            or document.get("source_primary_audio_path") != record.primary_audio_path
            or document.get("source_audio_sha256") != record.primary_audio_sha256
            or document.get("config_hash") != config.config_hash
            or document.get("source_manifest_fingerprint") != source.fingerprint
        ):
            raise AssertionError(f"cache primary/provenance binding mismatch: {logical_id}")
        nested = document.get("metadata", {})
        source_metadata = nested.get("source", {})
        if (
            source_metadata.get("speaker_id") != record.speaker_id
            or source_metadata.get("session_id") != record.session_id
            or source_metadata.get("prompt_id") != record.prompt_id
            or source_metadata.get("condition") != record.condition.value
            or source_metadata.get("primary_microphone")
            != (record.primary_microphone.value if record.primary_microphone else None)
        ):
            raise AssertionError(f"cache source metadata mismatch: {logical_id}")
        vmd_metadata = nested.get("vmd", {})
        for key, expected in expected_vmd.items():
            if vmd_metadata.get(key) != expected:
                parameter_mismatches.append(f"{logical_id}:{key}")
        if nested.get("reconstruction_policy") != "all_modes":
            parameter_mismatches.append(f"{logical_id}:reconstruction_policy")
        modes_header = np.load(directory / "modes.npy", mmap_mode="r", allow_pickle=False)
        reconstruction_header = np.load(
            directory / "reconstruction.npy", mmap_mode="r", allow_pickle=False
        )
        if modes_header.shape != (5, sample_count) or modes_header.dtype != np.float64:
            raise AssertionError(f"invalid modes header: {logical_id}")
        if (
            reconstruction_header.shape != (sample_count,)
            or reconstruction_header.dtype != np.float64
        ):
            raise AssertionError(f"invalid reconstruction header: {logical_id}")
        bound_primary_count += 1

        if result_by_id[logical_id]["converged"] is not False:
            continue
        entry = cache.load(identity)
        if not np.array_equal(entry.reconstruction, np.sum(entry.modes, axis=0, dtype=np.float64)):
            raise AssertionError(f"reconstruction is not exact all-mode sum: {logical_id}")
        mode_metadata = entry.metadata.get("mode_metadata")
        if not isinstance(mode_metadata, list) or len(mode_metadata) != 5:
            raise AssertionError(f"invalid mode metadata: {logical_id}")
        center_frequencies = [float(item["center_frequency_hz"]) for item in mode_metadata]
        solver_centers = [
            float(value)
            for value in entry.metadata["vmd"]["solver_center_frequencies_hz"]
        ]
        if not all(math.isfinite(value) and 0.0 <= value <= 8000.0 for value in center_frequencies):
            raise AssertionError(f"invalid center frequencies: {logical_id}")
        if len(solver_centers) != 5 or not all(
            math.isfinite(value) and 0.0 <= value <= 8000.0 for value in solver_centers
        ):
            raise AssertionError(f"invalid solver center frequencies: {logical_id}")

        primary_path = resolve_primary_audio(record, REPO_ROOT)
        if _sha256(primary_path) != record.primary_audio_sha256:
            raise AssertionError(f"source primary hash changed: {logical_id}")
        standardized = standardize_audio(
            primary_path,
            AudioLoadConfig(target_sample_rate=16000, target_channels=1),
            NormalizationConfig(
                strategy="rms",
                target_rms=config.audio.target_rms,
                max_gain=config.audio.max_gain,
                peak_limit=config.audio.peak_limit,
                silence_epsilon=config.audio.silence_epsilon,
            ),
        )
        recalculated = compute_reconstruction_metrics(
            standardized.waveform, entry.reconstruction
        ).to_dict()
        stored = entry.metadata["reconstruction_metrics"]
        for key in (
            "mse",
            "rmse",
            "normalized_reconstruction_error",
            "correlation_coefficient",
            "energy_ratio",
        ):
            _assert_close(float(recalculated[key]), float(stored[key]), f"{logical_id}:{key}")
        final_delta = float(entry.metadata["vmd"]["final_delta"])
        if not math.isfinite(final_delta) or final_delta < 0.0:
            raise AssertionError(f"invalid final delta: {logical_id}")
        nonconverged_rows.append(
            {
                "logical_id": logical_id,
                "speaker_id": record.speaker_id,
                "condition": record.condition.value,
                "iterations": int(entry.metadata["vmd"]["iterations"]),
                "final_delta": final_delta,
                "normalized_reconstruction_error": float(
                    recalculated["normalized_reconstruction_error"]
                ),
                "correlation_coefficient": float(recalculated["correlation_coefficient"]),
                "energy_ratio": float(recalculated["energy_ratio"]),
                "mse": float(recalculated["mse"]),
                "rmse": float(recalculated["rmse"]),
                "cache_valid": True,
                "arrays_finite": bool(
                    np.isfinite(entry.modes).all() and np.isfinite(entry.reconstruction).all()
                ),
                "dimensions_valid": True,
                "center_frequencies_valid": True,
                "reconstruction_exact_all_mode_sum": True,
                "source_primary_sha256_valid": True,
                "reconstruction_metrics_recomputed": True,
            }
        )

    if incomplete_directories or parameter_mismatches:
        raise AssertionError(
            "cache structure/parameter mismatch: "
            f"{incomplete_directories!r} {parameter_mismatches!r}"
        )
    if bound_primary_count != len(core_records):
        raise AssertionError("not every cache is bound to a CORE primary")
    nonconverged_rows.sort(key=lambda item: str(item["logical_id"]))
    if len(nonconverged_rows) != 48:
        raise AssertionError("non-converged count differs from the production summary")

    NONCONVERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NONCONVERGED_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nonconverged_rows[0]))
        writer.writeheader()
        writer.writerows(nonconverged_rows)

    production_namespaces = sorted(
        path.name for path in config.cache.root.iterdir() if path.is_dir()
    )
    experimental_root = REPO_ROOT / "data/processed/vmd_fidelity_sweep"
    experimental = []
    if experimental_root.is_dir():
        experimental = [
            _inventory_namespace(path)
            for path in sorted(experimental_root.iterdir())
            if path.is_dir()
        ]
    inventory.update(
        {
            "production_namespaces": production_namespaces,
            "experimental_root": experimental_root.relative_to(REPO_ROOT).as_posix(),
            "experimental_namespaces": experimental,
        }
    )
    _write_json(CACHE_INVENTORY_PATH, inventory)

    result_records = summary["records"]
    metric_keys = (
        "normalized_reconstruction_error",
        "correlation_coefficient",
        "energy_ratio",
        "rmse",
        "mse",
        "iterations",
        "vmd_seconds",
        "record_total_seconds",
    )
    all_distributions = {
        key: _statistics([float(record[key]) for record in result_records])
        for key in metric_keys
    }
    nonconverged_distributions = {
        key: _statistics([float(row[key]) for row in nonconverged_rows])
        for key in (
            "final_delta",
            "normalized_reconstruction_error",
            "correlation_coefficient",
            "energy_ratio",
            "rmse",
            "mse",
            "iterations",
        )
    }
    slowest = max(result_records, key=lambda item: float(item["vmd_seconds"]))
    audit = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "scope": "module1_full_core_final_signoff",
        "source": {
            "summary": SUMMARY_PATH.relative_to(REPO_ROOT).as_posix(),
            "config": CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
            "manifest_sha256": source.fingerprint,
            "raw_metadata_sha256": summary["source"]["raw_metadata_sha256"],
            "config_hash": config.config_hash,
        },
        "completion": {
            "subset": "core",
            "expected": len(core_records),
            "summary_records": len(result_records),
            "valid_structural_cache_entries": bound_primary_count,
            "computed": summary["completion"]["computed"],
            "cache_hits": summary["completion"]["cache_hits"],
            "failed": summary["completion"]["failed"],
            "converged": sum(bool(record["converged"]) for record in result_records),
            "nonconverged": len(nonconverged_rows),
        },
        "configuration": summary["configuration"],
        "full_distributions": all_distributions,
        "runtime": {
            "measured_pipeline_elapsed_seconds": summary["timing"]["run_elapsed_seconds"],
            "aggregate_record_vmd_seconds": sum(
                float(record["vmd_seconds"]) for record in result_records
            ),
            "aggregate_record_total_seconds": sum(
                float(record["record_total_seconds"]) for record in result_records
            ),
            "slowest_vmd_record": {
                "logical_id": slowest["logical_id"],
                "seconds": slowest["vmd_seconds"],
                "iterations": slowest["iterations"],
                "sample_count": slowest["original_length"],
            },
        },
        "cache": inventory,
        "primary_binding": {
            "checked": bound_primary_count,
            "mismatches": 0,
        },
        "nonconverged": {
            "count": len(nonconverged_rows),
            "logical_ids": [row["logical_id"] for row in nonconverged_rows],
            "speakers": dict(
                sorted(Counter(row["speaker_id"] for row in nonconverged_rows).items())
            ),
            "conditions": dict(
                sorted(Counter(row["condition"] for row in nonconverged_rows).items())
            ),
            "distributions": nonconverged_distributions,
            "all_cache_valid": all(bool(row["cache_valid"]) for row in nonconverged_rows),
            "all_arrays_finite": all(bool(row["arrays_finite"]) for row in nonconverged_rows),
            "all_dimensions_valid": all(bool(row["dimensions_valid"]) for row in nonconverged_rows),
            "all_center_frequencies_valid": all(
                bool(row["center_frequencies_valid"]) for row in nonconverged_rows
            ),
            "all_reconstructions_exact_mode_sums": all(
                bool(row["reconstruction_exact_all_mode_sum"])
                for row in nonconverged_rows
            ),
            "all_source_primary_hashes_valid": all(
                bool(row["source_primary_sha256_valid"]) for row in nonconverged_rows
            ),
            "all_reconstruction_metrics_recomputed": all(
                bool(row["reconstruction_metrics_recomputed"])
                for row in nonconverged_rows
            ),
        },
    }
    _write_json(AUDIT_PATH, audit)
    print(json.dumps({
        "status": audit["status"],
        "completion": audit["completion"],
        "cache": {
            "logical_cache_directories": inventory["logical_cache_directories"],
            "total_bytes": inventory["total_bytes"],
            "total_gib": inventory["total_gib"],
        },
        "nonconverged": audit["nonconverged"],
    }, indent=2))


if __name__ == "__main__":
    main()
