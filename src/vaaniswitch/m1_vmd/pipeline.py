"""Resumable Module 1 VMD build and validation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vaaniswitch.m0_data.manifest import write_csv, write_json
from vaaniswitch.m0_data.reporting import snapshot_raw_tree
from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m1_vmd.audio import (
    AudioLoadConfig,
    NormalizationConfig,
    StandardizedAudio,
    standardize_audio,
)
from vaaniswitch.m1_vmd.cache import CacheEntry, CacheIdentity, CacheValidationError, VMDCache
from vaaniswitch.m1_vmd.config import Module1Config
from vaaniswitch.m1_vmd.manifest import (
    ManifestSource,
    load_manifest_source,
    resolve_primary_audio,
    select_limited_records,
    select_subset,
)
from vaaniswitch.m1_vmd.metrics import compute_mode_metrics, compute_reconstruction_metrics
from vaaniswitch.m1_vmd.plots import write_vmd_diagnostic_svg
from vaaniswitch.m1_vmd.reconstruction import (
    reconstruct_all_modes,
    reconstruct_selected_modes,
)
from vaaniswitch.m1_vmd.vmd import VMDParameters, variational_mode_decomposition

LOGGER = logging.getLogger(__name__)
FAILURE_FIELDS = (
    "logical_id",
    "speaker_id",
    "condition",
    "primary_audio_path",
    "error_type",
    "error",
)
_MODULE1_HANDLERS: list[logging.Handler] = []


class Module1PipelineError(RuntimeError):
    """Raised when Module 1 cannot complete or validate safely."""


def run_module1(
    config: Module1Config,
    *,
    subset: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build or resume VMD cache entries for a filtered manifest selection."""

    config.validate()
    if not config.cache.enabled:
        raise Module1PipelineError("Module 1 requires the integrity-checked cache to be enabled")
    chosen_subset = subset or config.dataset.default_subset
    _configure_logging(config.outputs.pipeline_log)
    source = load_manifest_source(config.dataset.manifest_path)
    subset_records = select_subset(source.records, chosen_subset)
    selected_records = select_limited_records(subset_records, limit)
    raw_before = snapshot_raw_tree(config.dataset.raw_root)
    _validate_module0_contract(config, source, raw_before, chosen_subset, len(subset_records))

    LOGGER.info(
        "Module 1 build started: subset=%s selected=%d available=%d config_hash=%s",
        chosen_subset,
        len(selected_records),
        len(subset_records),
        config.config_hash,
    )
    cache = VMDCache(config.cache.root)
    failures: list[dict[str, str]] = []
    record_results: list[dict[str, Any]] = []
    plot_paths: list[str] = []
    run_started = time.perf_counter()

    for position, record in enumerate(selected_records, start=1):
        try:
            result, entry, waveform = _process_record(
                record,
                source,
                config,
                cache,
                force=force,
                need_waveform=position <= config.execution.plot_examples,
            )
            record_results.append(result)
            if position <= config.execution.plot_examples:
                if waveform is None:
                    waveform = _standardize_and_validate(record, config).waveform
                relative_energies = _relative_energies(entry)
                plot_path = config.outputs.plots / _plot_filename(record)
                write_vmd_diagnostic_svg(
                    plot_path,
                    logical_id=record.logical_id,
                    waveform=waveform,
                    modes=entry.modes,
                    reconstruction=entry.reconstruction,
                    relative_energies=relative_energies,
                    reconstruction_label=_scientific_label(config),
                )
                plot_paths.append(_relative_path(plot_path, config.repo_root))
        except Exception as exc:
            LOGGER.exception("Module 1 record failed: %s", record.logical_id)
            failures.append(
                {
                    "logical_id": record.logical_id,
                    "speaker_id": record.speaker_id,
                    "condition": record.condition.value,
                    "primary_audio_path": record.primary_audio_path or "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if position % config.execution.progress_every == 0 or position == len(selected_records):
            LOGGER.info(
                "Module 1 progress: %d/%d records; failures=%d",
                position,
                len(selected_records),
                len(failures),
            )

    raw_after = snapshot_raw_tree(config.dataset.raw_root)
    if raw_after != raw_before:
        raise Module1PipelineError("raw TORGO metadata changed during Module 1")
    elapsed = time.perf_counter() - run_started
    write_csv(config.outputs.failures, FAILURE_FIELDS, failures)
    summary = _build_summary(
        config=config,
        source=source,
        subset=chosen_subset,
        subset_count=len(subset_records),
        subset_duration_seconds=sum(
            float(record.duration_seconds or 0.0) for record in subset_records
        ),
        subset_sample_count=sum(
            _expected_sample_count(record, config) for record in subset_records
        ),
        selected_records=selected_records,
        results=record_results,
        failures=failures,
        plot_paths=plot_paths,
        raw_snapshot=raw_after,
        elapsed_seconds=elapsed,
        limit=limit,
    )
    write_json(config.outputs.summary, summary)
    LOGGER.info(
        "Module 1 build finished: completed=%d failed=%d elapsed=%.3fs",
        len(record_results),
        len(failures),
        elapsed,
    )
    if failures:
        raise Module1PipelineError(
            f"Module 1 completed with {len(failures)} failed records; see failure report"
        )
    return summary


def validate_module1(
    config: Module1Config,
    *,
    subset: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Validate cache integrity, all-mode semantics, reports, and source bindings."""

    config.validate()
    chosen_subset = subset or config.dataset.default_subset
    source = load_manifest_source(config.dataset.manifest_path)
    subset_records = select_subset(source.records, chosen_subset)
    selected_records = select_limited_records(subset_records, limit)
    raw_snapshot = snapshot_raw_tree(config.dataset.raw_root)
    _validate_module0_contract(config, source, raw_snapshot, chosen_subset, len(subset_records))
    summary = _read_json(config.outputs.summary, "Module 1 summary")
    _validate_summary_identity(
        summary,
        config=config,
        source=source,
        subset=chosen_subset,
        subset_count=len(subset_records),
        selected_records=selected_records,
        limit=limit,
        raw_snapshot=raw_snapshot,
    )
    _validate_failure_report(config.outputs.failures)

    cache = VMDCache(config.cache.root)
    converged_count = 0
    for record in selected_records:
        identity = _cache_identity(record, source, config)
        entry = cache.load(identity)
        _validate_entry_semantics(entry, record, config)
        expected = _reconstruct(entry.modes, config)
        if not np.array_equal(entry.reconstruction, expected):
            raise Module1PipelineError(
                f"cached reconstruction violates configured mode policy: {record.logical_id}"
            )
        standardized = _standardize_and_validate(record, config)
        reconstruction_metrics = compute_reconstruction_metrics(
            standardized.waveform, entry.reconstruction
        )
        metadata = entry.metadata
        if metadata.get("reconstruction_policy") != _reconstruction_policy(config):
            raise Module1PipelineError(
                f"cache reconstruction policy metadata mismatch: {record.logical_id}"
            )
        vmd_metadata = _require_mapping(metadata, "vmd")
        if vmd_metadata.get("converged") is True:
            converged_count += 1
        stored_metrics = _require_mapping(metadata, "reconstruction_metrics")
        _compare_numeric_mapping(
            stored_metrics,
            reconstruction_metrics.to_dict(),
            label=f"reconstruction metrics for {record.logical_id}",
        )
        stored_modes = metadata.get("mode_metadata")
        if not isinstance(stored_modes, list) or len(stored_modes) != config.vmd.modes_k:
            raise Module1PipelineError(
                f"cache mode metadata count mismatch: {record.logical_id}"
            )
        recomputed_modes = compute_mode_metrics(
            entry.modes, sample_rate=standardized.sample_rate
        )
        for stored, recomputed in zip(stored_modes, recomputed_modes, strict=True):
            if not isinstance(stored, dict):
                raise Module1PipelineError(
                    f"cache mode metadata entry is invalid: {record.logical_id}"
                )
            _compare_numeric_mapping(
                stored,
                recomputed.to_dict(),
                label=f"mode metrics for {record.logical_id}",
            )

    return {
        "status": "PASS",
        "subset": chosen_subset,
        "selected_records": len(selected_records),
        "valid_cache_entries": len(selected_records),
        "converged_records": converged_count,
        "config_hash": config.config_hash,
        "source_manifest_fingerprint": source.fingerprint,
        "raw_metadata_sha256": raw_snapshot["metadata_sha256"],
        "all_mode_default": config.reconstruction.default == "all_modes",
    }


def _process_record(
    record: ManifestRecord,
    source: ManifestSource,
    config: Module1Config,
    cache: VMDCache,
    *,
    force: bool,
    need_waveform: bool,
) -> tuple[dict[str, Any], CacheEntry, npt.NDArray[np.float64] | None]:
    identity = _cache_identity(record, source, config)
    if config.cache.resume and not force:
        try:
            entry = cache.load(identity)
            _validate_entry_semantics(entry, record, config)
        except (CacheValidationError, Module1PipelineError):
            pass
        else:
            return _record_result(record, entry, status="cache_hit"), entry, None

    started = time.perf_counter()
    standardized = _standardize_and_validate(record, config)
    parameters = VMDParameters(
        modes_k=config.vmd.modes_k,
        alpha=config.vmd.alpha,
        tau=config.vmd.tau,
        dc=config.vmd.dc,
        init=config.vmd.init,
        tolerance=config.vmd.tolerance,
        max_iterations=config.vmd.max_iterations,
    )
    vmd_started = time.perf_counter()
    decomposition = variational_mode_decomposition(
        standardized.waveform,
        sample_rate=standardized.sample_rate,
        parameters=parameters,
    )
    vmd_seconds = time.perf_counter() - vmd_started
    reconstruction = _reconstruct(decomposition.modes, config)
    mode_metrics = compute_mode_metrics(decomposition.modes, sample_rate=standardized.sample_rate)
    reconstruction_metrics = compute_reconstruction_metrics(
        standardized.waveform,
        reconstruction,
    )
    metadata: dict[str, object] = {
        "module": "1_shared_vmd",
        "implementation": config.implementation,
        "reconstruction_policy": _reconstruction_policy(config),
        "source": {
            "speaker_id": record.speaker_id,
            "condition": record.condition.value,
            "session_id": record.session_id,
            "prompt_id": record.prompt_id,
            "primary_microphone": (
                record.primary_microphone.value if record.primary_microphone else None
            ),
        },
        "audio": {
            "source_sample_rate": standardized.source_sample_rate,
            "source_channels": standardized.source_channels,
            "sample_width_bytes": standardized.sample_width_bytes,
            "source_frame_count": standardized.frame_count,
            "source_file_size": standardized.file_size,
            "working_sample_rate": standardized.sample_rate,
            "working_channels": standardized.channels,
            "working_sample_count": int(standardized.waveform.size),
            "downmixed": standardized.downmixed,
            "resampled": standardized.resampled,
            "resample_method": standardized.resample_method,
            "normalization": asdict(standardized.normalization),
        },
        "vmd": {
            **asdict(parameters),
            "iterations": decomposition.iterations,
            "converged": decomposition.converged,
            "final_delta": decomposition.final_delta,
            "solver_center_frequencies_hz": decomposition.center_frequencies_hz.tolist(),
        },
        "mode_metadata": [metric.to_dict() for metric in mode_metrics],
        "reconstruction_metrics": reconstruction_metrics.to_dict(),
        "timing": {
            "vmd_seconds": vmd_seconds,
            "record_total_seconds": time.perf_counter() - started,
        },
    }
    entry = cache.write(identity, decomposition.modes, reconstruction, metadata)
    _validate_entry_semantics(entry, record, config)
    waveform = standardized.waveform if need_waveform else None
    result = _record_result(record, entry, status="computed")
    result["record_total_seconds"] = time.perf_counter() - started
    return result, entry, waveform


def _standardize_and_validate(record: ManifestRecord, config: Module1Config) -> StandardizedAudio:
    path = resolve_primary_audio(record, config.repo_root)
    if not path.is_file():
        raise Module1PipelineError(f"primary audio is missing: {record.primary_audio_path}")
    assert record.primary_audio_sha256 is not None
    if _sha256_file(path) != record.primary_audio_sha256:
        raise Module1PipelineError(
            f"primary audio SHA-256 differs from Module 0: {record.logical_id}"
        )
    standardized = standardize_audio(
        path,
        AudioLoadConfig(
            target_sample_rate=config.audio.target_sample_rate,
            target_channels=config.audio.target_channels,
        ),
        NormalizationConfig(
            strategy="rms",
            target_rms=config.audio.target_rms,
            max_gain=config.audio.max_gain,
            peak_limit=config.audio.peak_limit,
            silence_epsilon=config.audio.silence_epsilon,
        ),
    )
    expected = {
        "sample_rate": record.sample_rate,
        "channels": record.channels,
        "sample_width_bytes": record.sample_width_bytes,
        "frame_count": record.frame_count,
        "file_size": record.file_size,
    }
    actual = {
        "sample_rate": standardized.source_sample_rate,
        "channels": standardized.source_channels,
        "sample_width_bytes": standardized.sample_width_bytes,
        "frame_count": standardized.frame_count,
        "file_size": standardized.file_size,
    }
    mismatches = [name for name in expected if expected[name] != actual[name]]
    if mismatches:
        raise Module1PipelineError(
            f"primary audio metadata differs from Module 0 for {record.logical_id}: {mismatches}"
        )
    if standardized.waveform.size != _expected_sample_count(record, config):
        raise Module1PipelineError(
            f"standardized sample count is inconsistent for {record.logical_id}"
        )
    return standardized


def _cache_identity(
    record: ManifestRecord,
    source: ManifestSource,
    config: Module1Config,
) -> CacheIdentity:
    if record.primary_audio_path is None or record.primary_audio_sha256 is None:
        raise Module1PipelineError(f"record lacks primary audio identity: {record.logical_id}")
    return CacheIdentity(
        logical_id=record.logical_id,
        source_primary_audio_path=record.primary_audio_path,
        source_audio_sha256=record.primary_audio_sha256,
        config_hash=config.config_hash,
        source_manifest_fingerprint=source.fingerprint,
        modes_k=config.vmd.modes_k,
        sample_count=_expected_sample_count(record, config),
    )


def _expected_sample_count(record: ManifestRecord, config: Module1Config) -> int:
    if record.frame_count is None or record.sample_rate is None:
        raise Module1PipelineError(f"record lacks source length metadata: {record.logical_id}")
    if record.sample_rate == config.audio.target_sample_rate:
        return record.frame_count
    return max(
        1,
        (
            record.frame_count * config.audio.target_sample_rate
            + record.sample_rate // 2
        )
        // record.sample_rate,
    )


def _reconstruct(
    modes: npt.NDArray[np.float64],
    config: Module1Config,
) -> npt.NDArray[np.float64]:
    ablation = config.reconstruction.selected_modes_ablation
    if ablation.enabled:
        return reconstruct_selected_modes(modes, list(ablation.mode_indices))
    return reconstruct_all_modes(modes)


def _reconstruction_policy(config: Module1Config) -> str:
    if config.reconstruction.selected_modes_ablation.enabled:
        selected = ",".join(
            str(index) for index in config.reconstruction.selected_modes_ablation.mode_indices
        )
        return f"selected_modes_explicit:{selected}"
    return "all_modes"


def _scientific_label(config: Module1Config) -> str:
    if config.reconstruction.selected_modes_ablation.enabled:
        return "explicit selected-mode VMD ablation reconstruction"
    return "all-mode VMD reconstruction"


def _record_result(
    record: ManifestRecord,
    entry: CacheEntry,
    *,
    status: str,
) -> dict[str, Any]:
    timing = _require_mapping(entry.metadata, "timing")
    reconstruction = _require_mapping(entry.metadata, "reconstruction_metrics")
    vmd = _require_mapping(entry.metadata, "vmd")
    audio = _require_mapping(entry.metadata, "audio")
    normalization = _require_mapping(audio, "normalization")
    return {
        "logical_id": record.logical_id,
        "speaker_id": record.speaker_id,
        "condition": record.condition.value,
        "primary_microphone": (
            record.primary_microphone.value if record.primary_microphone else None
        ),
        "status": status,
        "cache_directory": "/".join(entry.directory.parts[-4:]),
        "vmd_seconds": _finite_number(timing, "vmd_seconds"),
        "record_total_seconds": _finite_number(timing, "record_total_seconds"),
        "converged": vmd.get("converged") is True,
        "iterations": _integer(vmd, "iterations"),
        "waveform_summary": {
            "working_sample_count": _integer(audio, "working_sample_count"),
            "working_sample_rate": _integer(audio, "working_sample_rate"),
            "rms_before_normalization": _finite_number(normalization, "rms_before"),
            "rms_after_normalization": _finite_number(normalization, "rms_after"),
            "peak_before_normalization": _finite_number(normalization, "peak_before"),
            "peak_after_normalization": _finite_number(normalization, "peak_after"),
        },
        "original_length": _integer(reconstruction, "original_length"),
        "reconstruction_length": _integer(reconstruction, "reconstruction_length"),
        "mse": _finite_number(reconstruction, "mse"),
        "rmse": _finite_number(reconstruction, "rmse"),
        "normalized_reconstruction_error": _finite_number(
            reconstruction, "normalized_reconstruction_error"
        ),
        "correlation_coefficient": _finite_number(
            reconstruction, "correlation_coefficient"
        ),
        "energy_ratio": _finite_number(reconstruction, "energy_ratio"),
        "relative_mode_energies": _relative_energies(entry),
    }


def _validate_entry_semantics(
    entry: CacheEntry,
    record: ManifestRecord,
    config: Module1Config,
) -> None:
    metadata = entry.metadata
    if metadata.get("module") != "1_shared_vmd":
        raise Module1PipelineError("cache module metadata is invalid")
    if metadata.get("implementation") != config.implementation:
        raise Module1PipelineError("cache implementation metadata is invalid")
    if metadata.get("reconstruction_policy") != _reconstruction_policy(config):
        raise Module1PipelineError("cache reconstruction policy metadata is invalid")

    source = _require_mapping(metadata, "source")
    expected_source = {
        "speaker_id": record.speaker_id,
        "condition": record.condition.value,
        "session_id": record.session_id,
        "prompt_id": record.prompt_id,
        "primary_microphone": (
            record.primary_microphone.value if record.primary_microphone else None
        ),
    }
    if source != expected_source:
        raise Module1PipelineError("cache source provenance metadata is invalid")

    audio = _require_mapping(metadata, "audio")
    expected_audio = {
        "source_sample_rate": record.sample_rate,
        "source_channels": record.channels,
        "sample_width_bytes": record.sample_width_bytes,
        "source_frame_count": record.frame_count,
        "source_file_size": record.file_size,
        "working_sample_rate": config.audio.target_sample_rate,
        "working_channels": config.audio.target_channels,
        "working_sample_count": _expected_sample_count(record, config),
        "downmixed": record.channels != config.audio.target_channels,
        "resampled": record.sample_rate != config.audio.target_sample_rate,
        "resample_method": (
            "linear_interpolation"
            if record.sample_rate != config.audio.target_sample_rate
            else None
        ),
    }
    for key, expected_audio_value in expected_audio.items():
        if audio.get(key) != expected_audio_value:
            raise Module1PipelineError(f"cache audio metadata mismatch for {key}")
    normalization = _require_mapping(audio, "normalization")
    expected_normalization_keys = {
        "strategy",
        "rms_before",
        "rms_after",
        "peak_before",
        "peak_after",
        "applied_gain",
        "is_silent",
        "gain_limited",
        "peak_limited",
    }
    if set(normalization) != expected_normalization_keys:
        raise Module1PipelineError("cache normalization metadata fields are invalid")
    if normalization.get("strategy") != config.audio.normalization:
        raise Module1PipelineError("cache normalization strategy is invalid")
    for key in ("is_silent", "gain_limited", "peak_limited"):
        if not isinstance(normalization.get(key), bool):
            raise Module1PipelineError(f"cache normalization {key} must be boolean")
    for key in (
        "rms_before",
        "rms_after",
        "peak_before",
        "peak_after",
        "applied_gain",
    ):
        if _finite_number(normalization, key) < 0.0:
            raise Module1PipelineError(f"cache normalization {key} must be non-negative")

    vmd = _require_mapping(metadata, "vmd")
    expected_vmd = {
        "modes_k": config.vmd.modes_k,
        "alpha": config.vmd.alpha,
        "tau": config.vmd.tau,
        "dc": config.vmd.dc,
        "init": config.vmd.init,
        "tolerance": config.vmd.tolerance,
        "max_iterations": config.vmd.max_iterations,
        "random_seed": 0,
    }
    for key, expected_vmd_value in expected_vmd.items():
        if vmd.get(key) != expected_vmd_value:
            raise Module1PipelineError(f"cache VMD parameter mismatch for {key}")
    iterations = _integer(vmd, "iterations")
    if iterations < 0 or iterations > config.vmd.max_iterations:
        raise Module1PipelineError("cache VMD iteration count is invalid")
    if not isinstance(vmd.get("converged"), bool):
        raise Module1PipelineError("cache VMD convergence flag must be boolean")
    if _finite_number(vmd, "final_delta") < 0.0:
        raise Module1PipelineError("cache VMD final delta must be non-negative")
    centers = vmd.get("solver_center_frequencies_hz")
    if not isinstance(centers, list) or len(centers) != config.vmd.modes_k:
        raise Module1PipelineError("cache VMD center-frequency list is invalid")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0.0
        or value > config.audio.target_sample_rate / 2.0
        for value in centers
    ):
        raise Module1PipelineError("cache VMD center frequency is out of bounds")

    timing = _require_mapping(metadata, "timing")
    if set(timing) != {"vmd_seconds", "record_total_seconds"}:
        raise Module1PipelineError("cache timing metadata fields are invalid")
    if any(_finite_number(timing, key) < 0.0 for key in timing):
        raise Module1PipelineError("cache timing metadata must be non-negative")

    mode_metadata = metadata.get("mode_metadata")
    if not isinstance(mode_metadata, list) or len(mode_metadata) != config.vmd.modes_k:
        raise Module1PipelineError("cache mode metadata count is invalid")
    expected_mode_keys = {
        "mode_index",
        "sample_count",
        "center_frequency_hz",
        "absolute_energy",
        "relative_energy",
        "rms",
        "spectral_centroid_hz",
    }
    relative_energies: list[float] = []
    absolute_energies: list[float] = []
    for expected_index, item in enumerate(mode_metadata, start=1):
        if not isinstance(item, dict) or set(item) != expected_mode_keys:
            raise Module1PipelineError("cache mode metadata fields are invalid")
        if _integer(item, "mode_index") != expected_index:
            raise Module1PipelineError("cache mode index is invalid")
        if _integer(item, "sample_count") != entry.identity.sample_count:
            raise Module1PipelineError("cache mode sample count is invalid")
        center = _finite_number(item, "center_frequency_hz")
        centroid = _finite_number(item, "spectral_centroid_hz")
        if not 0.0 <= center <= config.audio.target_sample_rate / 2.0:
            raise Module1PipelineError("cache mode center frequency is out of bounds")
        if not 0.0 <= centroid <= config.audio.target_sample_rate / 2.0:
            raise Module1PipelineError("cache mode spectral centroid is out of bounds")
        for key in ("absolute_energy", "relative_energy", "rms"):
            if _finite_number(item, key) < 0.0:
                raise Module1PipelineError(f"cache mode {key} must be non-negative")
        absolute_energies.append(_finite_number(item, "absolute_energy"))
        relative_energies.append(_finite_number(item, "relative_energy"))
    expected_relative_sum = 0.0 if sum(absolute_energies) == 0.0 else 1.0
    if not math.isclose(sum(relative_energies), expected_relative_sum, abs_tol=1e-12):
        raise Module1PipelineError("cache relative mode energies are not normalized")

    reconstruction = _require_mapping(metadata, "reconstruction_metrics")
    expected_reconstruction_keys = {
        "original_length",
        "reconstruction_length",
        "mse",
        "rmse",
        "normalized_reconstruction_error",
        "correlation_coefficient",
        "original_energy",
        "reconstruction_energy",
        "energy_difference",
        "energy_ratio",
    }
    if set(reconstruction) != expected_reconstruction_keys:
        raise Module1PipelineError("cache reconstruction metric fields are invalid")
    if _integer(reconstruction, "original_length") != entry.identity.sample_count:
        raise Module1PipelineError("cache original length metadata is invalid")
    if _integer(reconstruction, "reconstruction_length") != entry.identity.sample_count:
        raise Module1PipelineError("cache reconstruction length metadata is invalid")
    for key in expected_reconstruction_keys - {"original_length", "reconstruction_length"}:
        _finite_number(reconstruction, key)


def _build_summary(
    *,
    config: Module1Config,
    source: ManifestSource,
    subset: str,
    subset_count: int,
    subset_duration_seconds: float,
    subset_sample_count: int,
    selected_records: Sequence[ManifestRecord],
    results: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, str]],
    plot_paths: Sequence[str],
    raw_snapshot: Mapping[str, Any],
    elapsed_seconds: float,
    limit: int | None,
) -> dict[str, Any]:
    vmd_times = [float(result["vmd_seconds"]) for result in results]
    total_times = [float(result["record_total_seconds"]) for result in results]
    normalized_errors = [
        float(result["normalized_reconstruction_error"]) for result in results
    ]
    mean_squared_errors = [float(result["mse"]) for result in results]
    root_mean_squared_errors = [float(result["rmse"]) for result in results]
    correlations = [float(result["correlation_coefficient"]) for result in results]
    energy_ratios = [float(result["energy_ratio"]) for result in results]
    relative_mode_energies = [
        [float(value) for value in result["relative_mode_energies"]] for result in results
    ]
    mean_vmd = statistics.fmean(vmd_times) if vmd_times else 0.0
    mean_total = statistics.fmean(total_times) if total_times else 0.0
    selected_audio_seconds = sum(
        float(result["waveform_summary"]["working_sample_count"])
        / float(result["waveform_summary"]["working_sample_rate"])
        for result in results
    )
    estimate_vmd_by_records = mean_vmd * subset_count
    vmd_per_audio_second = (
        sum(vmd_times) / selected_audio_seconds if selected_audio_seconds > 0.0 else 0.0
    )
    total_per_audio_second = (
        sum(total_times) / selected_audio_seconds if selected_audio_seconds > 0.0 else 0.0
    )
    fixed_run_overhead = max(0.0, elapsed_seconds - sum(total_times))
    estimate_pipeline_by_records = mean_total * subset_count + fixed_run_overhead
    estimate_vmd_seconds = vmd_per_audio_second * subset_duration_seconds
    estimate_pipeline_seconds = (
        total_per_audio_second * subset_duration_seconds + fixed_run_overhead
    )
    return {
        "schema_version": "1.0",
        "module": "1_shared_vmd",
        "status": (
            "smoke_failed"
            if failures and limit is not None
            else "full_build_failed"
            if failures
            else "smoke_validated"
            if limit is not None
            else "full_build_completed"
        ),
        "scientific_label": _scientific_label(config),
        "implementation": {
            "name": config.implementation,
            "library": "project-owned pure NumPy",
            "numpy_version": np.__version__,
        },
        "source": {
            "manifest_path": _relative_path(source.path, config.repo_root),
            "manifest_fingerprint": source.fingerprint,
            "raw_metadata_sha256": raw_snapshot["metadata_sha256"],
        },
        "selection": {
            "subset": subset,
            "default_subset": config.dataset.default_subset,
            "subset_record_count": subset_count,
            "limit": limit,
            "selected_record_count": len(selected_records),
            "selection_fingerprint": _selection_fingerprint(selected_records),
            "conditions": sorted({record.condition.value for record in selected_records}),
            "speakers": sorted({record.speaker_id for record in selected_records}),
        },
        "configuration": {
            "config_hash": config.config_hash,
            "processing_identity": config.processing_identity,
            "all_mode_default": config.reconstruction.default == "all_modes",
            "selected_mode_ablation_enabled": (
                config.reconstruction.selected_modes_ablation.enabled
            ),
        },
        "completion": {
            "computed": sum(result["status"] == "computed" for result in results),
            "cache_hits": sum(result["status"] == "cache_hit" for result in results),
            "failed": len(failures),
            "successful": len(results),
        },
        "timing": {
            "run_elapsed_seconds": elapsed_seconds,
            "smoke_audio_seconds": selected_audio_seconds,
            "full_subset_audio_seconds": subset_duration_seconds,
            "vmd_seconds_mean": mean_vmd,
            "vmd_seconds_median": statistics.median(vmd_times) if vmd_times else 0.0,
            "vmd_seconds_per_audio_second": vmd_per_audio_second,
            "record_total_seconds_mean": (
                mean_total
            ),
            "estimated_full_subset_vmd_seconds": estimate_vmd_seconds,
            "estimated_full_subset_vmd_hours": estimate_vmd_seconds / 3600.0,
            "estimated_full_subset_pipeline_seconds": estimate_pipeline_seconds,
            "estimated_full_subset_pipeline_hours": estimate_pipeline_seconds / 3600.0,
            "record_count_scaled_vmd_seconds": estimate_vmd_by_records,
            "record_count_scaled_pipeline_seconds": estimate_pipeline_by_records,
            "estimate_note": (
                "Primary estimate scales smoke timing by manifest audio duration and adds "
                "observed fixed run overhead; it is not a measured full-corpus runtime."
            ),
        },
        "cache_storage_estimate": {
            "array_dtype": "float64",
            "arrays_per_record": config.vmd.modes_k + 1,
            "full_subset_sample_count": subset_sample_count,
            "nominal_array_bytes": subset_sample_count * (config.vmd.modes_k + 1) * 8,
            "nominal_array_gib": (
                subset_sample_count * (config.vmd.modes_k + 1) * 8 / (1024**3)
            ),
            "note": "Excludes JSON metadata, NumPy headers, and filesystem overhead.",
        },
        "reconstruction": {
            "policy": _reconstruction_policy(config),
            "converged_count": sum(bool(result["converged"]) for result in results),
            "mse": _distribution(mean_squared_errors),
            "rmse": _distribution(root_mean_squared_errors),
            "normalized_error": _distribution(normalized_errors),
            "correlation": _distribution(correlations),
            "energy_ratio": _distribution(energy_ratios),
            "relative_mode_energy_by_index": {
                f"mode_{index + 1}": _distribution(
                    [energies[index] for energies in relative_mode_energies]
                )
                for index in range(config.vmd.modes_k)
            },
            "threshold_note": (
                "Metrics are reported descriptively; Module 1 imposes no arbitrary "
                "perfect-reconstruction threshold."
            ),
        },
        "records": list(results),
        "outputs": {
            "cache_root": _relative_path(config.cache.root, config.repo_root),
            "failures": _relative_path(config.outputs.failures, config.repo_root),
            "pipeline_log": _relative_path(config.outputs.pipeline_log, config.repo_root),
            "plots": list(plot_paths),
        },
        "raw_data_immutability": {
            "verified_unchanged_during_run": True,
            "metadata_sha256": raw_snapshot["metadata_sha256"],
        },
    }


def _validate_module0_contract(
    config: Module1Config,
    source: ManifestSource,
    raw_snapshot: Mapping[str, Any],
    subset: str,
    subset_count: int,
) -> None:
    summary = _read_json(config.dataset.module0_summary_path, "Module 0 dataset summary")
    source_section = _require_mapping(summary, "source")
    if source_section.get("manifest_sha256") != source.fingerprint:
        raise Module1PipelineError("Module 0 summary has a stale manifest fingerprint")
    inventory = _require_mapping(summary, "raw_inventory")
    if inventory.get("metadata_sha256") != raw_snapshot.get("metadata_sha256"):
        raise Module1PipelineError("raw TORGO metadata differs from signed-off Module 0")
    policies = _require_mapping(summary, "downstream_subset_policy")
    policy = _require_mapping(policies, subset)
    if policy.get("record_count") != subset_count:
        raise Module1PipelineError(f"Module 0 {subset} subset count does not match manifest")

    leakage = _read_json(
        config.dataset.module0_leakage_report_path,
        "Module 0 leakage report",
    )
    if leakage.get("status") != "PASS" or leakage.get("violation_count") != 0:
        raise Module1PipelineError("Module 0 leakage hard gate is not PASS")
    leakage_source = _require_mapping(leakage, "source")
    if leakage_source.get("manifest_sha256") != source.fingerprint:
        raise Module1PipelineError("Module 0 leakage report has a stale manifest fingerprint")


def _validate_summary_identity(
    summary: Mapping[str, Any],
    *,
    config: Module1Config,
    source: ManifestSource,
    subset: str,
    subset_count: int,
    selected_records: Sequence[ManifestRecord],
    limit: int | None,
    raw_snapshot: Mapping[str, Any],
) -> None:
    source_data = _require_mapping(summary, "source")
    selection = _require_mapping(summary, "selection")
    configuration = _require_mapping(summary, "configuration")
    completion = _require_mapping(summary, "completion")
    reconstruction = _require_mapping(summary, "reconstruction")
    outputs = _require_mapping(summary, "outputs")
    expected_status = "smoke_validated" if limit is not None else "full_build_completed"
    if (
        summary.get("schema_version") != "1.0"
        or summary.get("module") != "1_shared_vmd"
        or summary.get("status") != expected_status
        or summary.get("scientific_label") != _scientific_label(config)
    ):
        raise Module1PipelineError("Module 1 summary status or schema is invalid")
    if source_data.get("manifest_fingerprint") != source.fingerprint:
        raise Module1PipelineError("Module 1 summary manifest fingerprint is stale")
    if source_data.get("manifest_path") != _relative_path(source.path, config.repo_root):
        raise Module1PipelineError("Module 1 summary manifest path is invalid")
    if source_data.get("raw_metadata_sha256") != raw_snapshot.get("metadata_sha256"):
        raise Module1PipelineError("Module 1 summary raw fingerprint is stale")
    if configuration.get("config_hash") != config.config_hash:
        raise Module1PipelineError("Module 1 summary config hash is stale")
    if configuration.get("processing_identity") != config.processing_identity:
        raise Module1PipelineError("Module 1 summary processing identity is stale")
    if configuration.get("all_mode_default") is not True:
        raise Module1PipelineError("Module 1 summary must retain all-mode default")
    if configuration.get("selected_mode_ablation_enabled") is not (
        config.reconstruction.selected_modes_ablation.enabled
    ):
        raise Module1PipelineError("Module 1 summary ablation state is inconsistent")
    expected_selection = {
        "subset": subset,
        "default_subset": config.dataset.default_subset,
        "subset_record_count": subset_count,
        "limit": limit,
        "selected_record_count": len(selected_records),
        "selection_fingerprint": _selection_fingerprint(selected_records),
    }
    if any(selection.get(key) != value for key, value in expected_selection.items()):
        raise Module1PipelineError("Module 1 summary selection does not match validation request")
    if completion.get("failed") != 0 or completion.get("successful") != len(selected_records):
        raise Module1PipelineError("Module 1 summary does not report complete success")
    if completion.get("computed", 0) + completion.get("cache_hits", 0) != len(
        selected_records
    ):
        raise Module1PipelineError("Module 1 summary completion counts are inconsistent")
    if reconstruction.get("policy") != _reconstruction_policy(config):
        raise Module1PipelineError("Module 1 summary reconstruction policy is inconsistent")
    raw_records = summary.get("records")
    if not isinstance(raw_records, list) or [
        item.get("logical_id") if isinstance(item, dict) else None for item in raw_records
    ] != [record.logical_id for record in selected_records]:
        raise Module1PipelineError("Module 1 summary record list is inconsistent")

    expected_outputs = {
        "cache_root": _relative_path(config.cache.root, config.repo_root),
        "failures": _relative_path(config.outputs.failures, config.repo_root),
        "pipeline_log": _relative_path(config.outputs.pipeline_log, config.repo_root),
    }
    if any(outputs.get(key) != value for key, value in expected_outputs.items()):
        raise Module1PipelineError("Module 1 summary output paths are inconsistent")
    if not config.outputs.pipeline_log.is_file():
        raise Module1PipelineError("Module 1 pipeline log is missing")
    if str(config.repo_root) in config.outputs.pipeline_log.read_text(encoding="utf-8"):
        raise Module1PipelineError("Module 1 pipeline log exposes an absolute repository path")
    plot_paths = outputs.get("plots")
    expected_plot_count = min(config.execution.plot_examples, len(selected_records))
    if not isinstance(plot_paths, list) or len(plot_paths) != expected_plot_count:
        raise Module1PipelineError("Module 1 summary plot list is inconsistent")
    for raw_path in plot_paths:
        if not isinstance(raw_path, str):
            raise Module1PipelineError("Module 1 plot path must be a string")
        plot_path = (config.repo_root / raw_path).resolve()
        try:
            plot_path.relative_to(config.outputs.plots.resolve())
        except ValueError as exc:
            raise Module1PipelineError("Module 1 plot path escapes plot output root") from exc
        if not plot_path.is_file():
            raise Module1PipelineError(f"Module 1 plot is missing: {raw_path}")


def _validate_failure_report(path: Path) -> None:
    if not path.is_file():
        raise Module1PipelineError("Module 1 failure report is missing")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FAILURE_FIELDS):
            raise Module1PipelineError("Module 1 failure report schema is invalid")
        if any(True for _ in reader):
            raise Module1PipelineError("Module 1 failure report is not empty")


def _compare_numeric_mapping(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if set(stored) != set(expected):
        raise Module1PipelineError(f"{label} fields do not match schema")
    for key, expected_value in expected.items():
        stored_value = stored.get(key)
        if isinstance(expected_value, int):
            if stored_value != expected_value:
                raise Module1PipelineError(f"{label} mismatch for {key}")
        elif not isinstance(stored_value, (int, float)) or isinstance(stored_value, bool):
            raise Module1PipelineError(f"{label} must contain numeric {key}")
        elif not math.isclose(
            float(stored_value),
            float(expected_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise Module1PipelineError(f"{label} mismatch for {key}")


def _relative_energies(entry: CacheEntry) -> list[float]:
    raw = entry.metadata.get("mode_metadata")
    if not isinstance(raw, list):
        raise Module1PipelineError("cache mode_metadata must be a list")
    energies: list[float] = []
    for item in raw:
        if not isinstance(item, dict):
            raise Module1PipelineError("cache mode metadata entry must be an object")
        energies.append(_finite_number(item, "relative_energy"))
    return energies


def _selection_fingerprint(records: Sequence[ManifestRecord]) -> str:
    content = "".join(f"{record.logical_id}\n" for record in records).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _plot_filename(record: ManifestRecord) -> str:
    return f"{record.speaker_id}__{record.session_id}__{record.prompt_id}.svg"


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Module1PipelineError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Module1PipelineError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise Module1PipelineError(f"{label} must contain a JSON object")
    return value


def _require_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise Module1PipelineError(f"{key} must be a JSON object")
    return item


def _finite_number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
        raise Module1PipelineError(f"{key} must be a finite number")
    return float(item)


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise Module1PipelineError(f"{key} must be an integer")
    return item


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise Module1PipelineError(f"output path escapes repository: {path}") from exc


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in _MODULE1_HANDLERS:
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        handler.close()
    _MODULE1_HANDLERS.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    _MODULE1_HANDLERS.extend((file_handler, console_handler))
