"""Module 2 orchestration: signed source -> three representations -> validated cache."""

from __future__ import annotations

import csv
import json
import logging
import os
import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m1_vmd.manifest import select_limited_records
from vaaniswitch.m2_representations.cache import CacheEntry, CacheIdentity, RepresentationCache
from vaaniswitch.m2_representations.config import Module2Config
from vaaniswitch.m2_representations.mel import generate_log_mel
from vaaniswitch.m2_representations.plots import write_representation_diagnostic_svg
from vaaniswitch.m2_representations.recurrence import generate_recurrence
from vaaniswitch.m2_representations.source import (
    SourceContract,
    load_reconstruction,
    load_source_contract,
)
from vaaniswitch.m2_representations.visibility import generate_visibility_graph

LOGGER = logging.getLogger(__name__)
FAILURE_FIELDS = ("logical_id", "error_type", "message")
_HANDLERS: list[logging.Handler] = []


class Module2PipelineError(RuntimeError):
    """Raised when Module 2 cannot complete or validate the selected records."""


def run_module2(
    config: Module2Config,
    *,
    subset: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a resumable Module 2 cache and publish smoke/full summary evidence."""

    started = time.perf_counter()
    _configure_logging(config.outputs.pipeline_log)
    chosen_subset = subset or config.source.default_subset
    contract = load_source_contract(config, chosen_subset)
    selected = _select_records(contract.records, config, limit)
    duration_audit = _duration_audit(contract.records, config.mel.fixed_duration_seconds)
    cache = RepresentationCache(config.cache.root)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    computed = 0
    cache_hits = 0
    plot_candidates: list[tuple[ManifestRecord, CacheEntry]] = []
    for index, record in enumerate(selected, start=1):
        try:
            source_started = time.perf_counter()
            source = load_reconstruction(config, contract, record)
            source_seconds = time.perf_counter() - source_started
            identity = _cache_identity(
                config, contract, source.logical_id, source.relative_path, source.artifact_sha256
            )
            entry: CacheEntry
            status: str
            if config.cache.resume and not force and cache.is_valid(identity):
                entry = cache.load(identity)
                cache_hits += 1
                status = "cache_hit"
            else:
                mel_started = time.perf_counter()
                mel = generate_log_mel(source.waveform, config.mel)
                mel_seconds = time.perf_counter() - mel_started
                recurrence_started = time.perf_counter()
                recurrence = generate_recurrence(source.waveform, config.recurrence)
                recurrence_seconds = time.perf_counter() - recurrence_started
                graph_started = time.perf_counter()
                graph = generate_visibility_graph(source.waveform, config.visibility_graph)
                graph_seconds = time.perf_counter() - graph_started
                transform_seconds = mel_seconds + recurrence_seconds + graph_seconds
                representation_metadata = {
                    "source": {
                        "module1_converged": source.converged,
                        "sample_count": int(source.waveform.size),
                        "sample_rate": 16000,
                        "reconstruction_policy": "all_modes",
                    },
                    "mel": {
                        "shape": list(mel.image.shape),
                        "dtype": str(mel.image.dtype),
                        "native_frame_count": mel.native_frame_count,
                        "fixed_frame_count": mel.fixed_frame_count,
                        "time_handling": mel.time_handling,
                        "padded_seconds": mel.padded_seconds,
                        "cropped_seconds": mel.cropped_seconds,
                        "minimum": float(np.min(mel.image)),
                        "maximum": float(np.max(mel.image)),
                    },
                    "recurrence": {
                        "shape": list(recurrence.image.shape),
                        "dtype": str(recurrence.image.dtype),
                        "scalar_state_count": recurrence.scalar_state_count,
                        "embedded_state_count": recurrence.embedded_state_count,
                        "threshold": recurrence.threshold,
                        "density": recurrence.recurrence_density,
                        "symmetric": bool(np.array_equal(recurrence.image, recurrence.image.T)),
                    },
                    "visibility_graph": {
                        "node_shape": list(graph.node_features.shape),
                        "node_dtype": str(graph.node_features.dtype),
                        "edge_shape": list(graph.edge_index.shape),
                        "edge_dtype": str(graph.edge_index.dtype),
                        "node_count": graph.node_features.shape[0],
                        "directed_edge_count": graph.edge_index.shape[1],
                        "undirected_edge_count": graph.edge_index.shape[1] // 2,
                        "density": graph.graph_density,
                        "isolated_node_count": graph.isolated_node_count,
                        "connected_component_count": graph.connected_component_count,
                        "adjacency_contract": config.visibility_graph.adjacency_contract,
                    },
                    "timing_seconds": {
                        "source_validation": source_seconds,
                        "mel": mel_seconds,
                        "recurrence": recurrence_seconds,
                        "visibility_graph": graph_seconds,
                        "representations_total": transform_seconds,
                    },
                }
                entry = cache.write(
                    identity,
                    mel.image,
                    recurrence.image,
                    graph.node_features,
                    graph.edge_index,
                    processing_identity=config.processing_identity,
                    representation_metadata=representation_metadata,
                )
                computed += 1
                status = "computed"
            records.append(_record_summary(record, entry, status, config))
            plot_candidates.append((record, entry))
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.exception("failed %s", record.logical_id)
            failures.append(
                {
                    "logical_id": record.logical_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        if index % config.execution.progress_every == 0:
            LOGGER.info("processed %d/%d records", index, len(selected))
    _write_failures(config.outputs.failures, failures)
    plots = _write_diagnostics(config, plot_candidates)
    status = "smoke_validated" if limit is not None and not failures else "full_build_completed"
    if failures:
        status = "smoke_failed" if limit is not None else "full_build_failed"
    summary = _build_summary(
        config=config,
        contract=contract,
        selected=selected,
        records=records,
        failures=failures,
        computed=computed,
        cache_hits=cache_hits,
        duration_audit=duration_audit,
        plots=plots,
        status=status,
        limit=limit,
        elapsed=time.perf_counter() - started,
    )
    _write_json_atomic(config.outputs.summary, summary)
    if failures:
        raise Module2PipelineError(f"Module 2 failed for {len(failures)} selected records")
    return summary


def validate_module2(
    config: Module2Config, *, subset: str | None = None, limit: int | None = None
) -> dict[str, Any]:
    """Validate selected Module 2 caches against current signed Module 1 artifacts."""

    chosen_subset = subset or config.source.default_subset
    contract = load_source_contract(config, chosen_subset)
    selected = _select_records(contract.records, config, limit)
    cache = RepresentationCache(config.cache.root)
    valid = 0
    for record in selected:
        source = load_reconstruction(config, contract, record)
        identity = _cache_identity(
            config, contract, record.logical_id, source.relative_path, source.artifact_sha256
        )
        entry = cache.load(identity)
        stored_source = entry.metadata.get("source")
        if (
            not isinstance(stored_source, dict)
            or stored_source.get("reconstruction_policy") != "all_modes"
        ):
            raise Module2PipelineError(f"cache source metadata is stale: {record.logical_id}")
        valid += 1
    return {
        "schema_version": "1.0",
        "module": "2_multirepresentation_generation",
        "status": "PASS",
        "subset": chosen_subset,
        "selected_record_count": len(selected),
        "valid_cache_entries": valid,
        "representation_config_hash": config.config_hash,
        "module1_config_hash": config.source.module1_config_hash,
        "manifest_fingerprint": contract.manifest_fingerprint,
    }


def _cache_identity(
    config: Module2Config,
    contract: SourceContract,
    logical_id: str,
    source_path: str,
    source_hash: str,
) -> CacheIdentity:
    return CacheIdentity(
        logical_id=logical_id,
        manifest_fingerprint=contract.manifest_fingerprint,
        module1_config_hash=config.source.module1_config_hash,
        source_reconstruction_path=source_path,
        source_reconstruction_sha256=source_hash,
        representation_config_hash=config.config_hash,
    )


def _select_records(
    records: tuple[ManifestRecord, ...], config: Module2Config, limit: int | None
) -> tuple[ManifestRecord, ...]:
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        raise Module2PipelineError("limit must be positive")
    by_id = {record.logical_id: record for record in records}
    configured = [by_id[item] for item in config.execution.smoke_logical_ids if item in by_id]
    if len(configured) >= limit:
        return tuple(configured[:limit])
    return select_limited_records(records, limit)


def _duration_audit(records: Sequence[ManifestRecord], fixed_seconds: float) -> dict[str, Any]:
    values = np.asarray([record.duration_seconds for record in records], dtype=np.float64)
    cropped = values > fixed_seconds
    padded = values < fixed_seconds
    return {
        "count": int(values.size),
        "seconds": {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        },
        "fixed_duration_seconds": fixed_seconds,
        "padding": {
            "count": int(np.count_nonzero(padded)),
            "percentage": float(100 * np.mean(padded)),
        },
        "cropping": {
            "count": int(np.count_nonzero(cropped)),
            "percentage": float(100 * np.mean(cropped)),
            "maximum_discarded_seconds": float(np.max(np.maximum(values - fixed_seconds, 0))),
        },
    }


def _record_summary(
    record: ManifestRecord, entry: CacheEntry, status: str, config: Module2Config
) -> dict[str, Any]:
    metadata = entry.metadata
    return {
        "logical_id": record.logical_id,
        "condition": record.condition.value,
        "gender": record.gender.value,
        "speaker_id": record.speaker_id,
        "primary_microphone": record.primary_microphone.value
        if record.primary_microphone
        else None,
        "duration_seconds": record.duration_seconds,
        "status": status,
        "cache_directory": entry.directory.relative_to(config.cache.root).as_posix(),
        "source": metadata["source"],
        "mel": metadata["mel"],
        "recurrence": metadata["recurrence"],
        "visibility_graph": metadata["visibility_graph"],
        "timing_seconds": metadata["timing_seconds"],
    }


def _write_diagnostics(
    config: Module2Config, candidates: Sequence[tuple[ManifestRecord, CacheEntry]]
) -> list[str]:
    selected: list[tuple[ManifestRecord, CacheEntry]] = []
    seen_conditions: set[str] = set()
    for record, entry in candidates:
        condition = record.condition.value
        if condition not in seen_conditions:
            selected.append((record, entry))
            seen_conditions.add(condition)
        if len(selected) >= config.execution.diagnostic_examples:
            break
    paths: list[str] = []
    for record, entry in selected:
        filename = record.logical_id.replace("/", "__") + ".svg"
        path = config.outputs.plots / filename
        write_representation_diagnostic_svg(
            path,
            logical_id=record.logical_id,
            condition=record.condition.value,
            mel=entry.mel,
            recurrence=entry.recurrence,
            graph_nodes=entry.graph_nodes,
            graph_edges=entry.graph_edges,
        )
        paths.append(path.relative_to(config.repo_root).as_posix())
    return paths


def _build_summary(
    *,
    config: Module2Config,
    contract: SourceContract,
    selected: Sequence[ManifestRecord],
    records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    computed: int,
    cache_hits: int,
    duration_audit: dict[str, Any],
    plots: list[str],
    status: str,
    limit: int | None,
    elapsed: float,
) -> dict[str, Any]:
    timing_names = ("mel", "recurrence", "visibility_graph", "representations_total")
    performance: dict[str, Any] = {}
    for name in timing_names:
        values = [float(record["timing_seconds"][name]) for record in records]
        performance[name] = _distribution(values)
    bytes_per_record = [
        sum(
            path.stat().st_size
            for path in (config.cache.root / record["cache_directory"]).iterdir()
            if path.is_file()
        )
        for record in records
    ]
    mean_bytes = statistics.fmean(bytes_per_record) if bytes_per_record else 0.0
    full_count = contract.signed_record_count
    smoke_statistics = _smoke_statistics(records)
    return {
        "schema_version": "1.0",
        "module": "2_multirepresentation_generation",
        "status": status,
        "source": {
            "subset": contract.subset,
            "manifest_fingerprint": contract.manifest_fingerprint,
            "module1_config_hash": config.source.module1_config_hash,
            "module1_cache_root": config.source.module1_cache_root.relative_to(
                config.repo_root
            ).as_posix(),
            "reconstruction_filename": "reconstruction.npy",
            "reconstruction_policy": "all_modes",
            "raw_metadata_fingerprint": contract.raw_metadata_fingerprint,
        },
        "selection": {
            "full_core_record_count": full_count,
            "limit": limit,
            "selected_record_count": len(selected),
            "logical_ids": [record.logical_id for record in selected],
        },
        "configuration": {
            "representation_config_hash": config.config_hash,
            "processing_identity": config.processing_identity,
        },
        "core_duration_audit": duration_audit,
        "smoke_statistics": smoke_statistics,
        "completion": {
            "computed": computed,
            "cache_hits": cache_hits,
            "failed": len(failures),
            "successful": len(records),
        },
        "performance_seconds": {
            **performance,
            "pipeline_elapsed": elapsed,
            "estimated_full_core_seconds": performance["representations_total"]["mean"]
            * full_count,
            "estimated_full_core_pipeline_seconds": (
                elapsed / len(selected) * full_count if selected else 0.0
            ),
            "estimate_note": (
                "Both estimates are smoke-scaled, not measured full-corpus runtimes; the "
                "pipeline estimate includes source validation, writes, reporting, and plots."
            ),
        },
        "storage": {
            "smoke_total_bytes": sum(bytes_per_record),
            "mean_bytes_per_record": mean_bytes,
            "estimated_full_core_bytes": mean_bytes * full_count,
            "estimated_full_core_gib": mean_bytes * full_count / (1024**3),
            "estimate_note": (
                "Scaled from complete smoke cache directories, including metadata and "
                "NumPy headers."
            ),
        },
        "outputs": {
            "cache_root": config.cache.root.relative_to(config.repo_root).as_posix(),
            "failures": config.outputs.failures.relative_to(config.repo_root).as_posix(),
            "pipeline_log": config.outputs.pipeline_log.relative_to(config.repo_root).as_posix(),
            "plots": plots,
        },
        "records": records,
        "failures": failures,
    }


def _smoke_statistics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    handling: dict[str, int] = {}
    for record in records:
        name = str(record["mel"]["time_handling"])
        handling[name] = handling.get(name, 0) + 1
    return {
        "mel": {
            "shape": [224, 224],
            "dtype": "float32",
            "finite_minimum": min(float(record["mel"]["minimum"]) for record in records),
            "finite_maximum": max(float(record["mel"]["maximum"]) for record in records),
            "native_frame_count": _distribution(
                [float(record["mel"]["native_frame_count"]) for record in records]
            ),
            "time_handling_counts": handling,
        },
        "recurrence": {
            "shape": [224, 224],
            "dtype": "uint8",
            "density": _distribution(
                [float(record["recurrence"]["density"]) for record in records]
            ),
            "all_symmetric": all(bool(record["recurrence"]["symmetric"]) for record in records),
        },
        "visibility_graph": {
            "node_count": _distribution(
                [float(record["visibility_graph"]["node_count"]) for record in records]
            ),
            "undirected_edge_count": _distribution(
                [float(record["visibility_graph"]["undirected_edge_count"]) for record in records]
            ),
            "density": _distribution(
                [float(record["visibility_graph"]["density"]) for record in records]
            ),
            "isolated_node_count": _distribution(
                [float(record["visibility_graph"]["isolated_node_count"]) for record in records]
            ),
            "connected_component_count": _distribution(
                [
                    float(record["visibility_graph"]["connected_component_count"])
                    for record in records
                ]
            ),
        },
    }


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


def _write_failures(path: Path, failures: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FAILURE_FIELDS))
        writer.writeheader()
        writer.writerows(failures)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in _HANDLERS:
        if handler in root.handlers:
            root.removeHandler(handler)
        handler.close()
    _HANDLERS.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.INFO)
    _HANDLERS.extend((file_handler, console_handler))
