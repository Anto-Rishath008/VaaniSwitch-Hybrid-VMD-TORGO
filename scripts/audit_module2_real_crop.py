"""Audit real >8-second CORE reconstructions without writing Module 2 caches."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vaaniswitch.m2_representations.config import Module2Config
from vaaniswitch.m2_representations.mel import generate_log_mel
from vaaniswitch.m2_representations.plots import write_crop_audit_diagnostic_svg
from vaaniswitch.m2_representations.recurrence import generate_recurrence
from vaaniswitch.m2_representations.source import load_reconstruction, load_source_contract
from vaaniswitch.m2_representations.visibility import generate_visibility_graph

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/representations/default.yaml"
SUMMARY_PATH = REPO_ROOT / "artifacts/metrics/module2_real_crop_audit.json"
PLOT_PATH = (
    REPO_ROOT
    / "artifacts/plots/module2_crop_audit/F03__Session1__0118_real_crop_diagnostic.svg"
)

# Chosen before transforms by descriptive strata only: three 8-12 s records
# spanning both conditions/genders, one >12 s, one >30 s, and the maximum.
CROP_PANEL_IDS = (
    "F03/Session1/0024",
    "FC02/Session3/0317",
    "MC04/Session1/0210",
    "M05/Session2/0242",
    "M03/Session2/0027",
    "F03/Session1/0118",
)


def main() -> None:
    config = Module2Config.from_yaml(CONFIG_PATH, repo_root=REPO_ROOT)
    contract = load_source_contract(config, "core")
    records_by_id = {record.logical_id: record for record in contract.records}
    long_records = [
        record
        for record in contract.records
        if record.duration_seconds is not None
        and record.duration_seconds > config.mel.fixed_duration_seconds
    ]
    durations = np.asarray([_duration(record) for record in long_records], dtype=np.float64)
    panel = []
    longest_id = max(long_records, key=_duration).logical_id

    for logical_id in CROP_PANEL_IDS:
        record = records_by_id[logical_id]
        source = load_reconstruction(config, contract, record)
        target = config.mel.fixed_sample_count
        start = (source.waveform.size - target) // 2
        stop = start + target
        centered = np.array(source.waveform[start:stop], dtype=np.float64, copy=True)

        mel_started = time.perf_counter()
        mel = generate_log_mel(source.waveform, config.mel)
        mel_seconds = time.perf_counter() - mel_started
        repeated_mel = generate_log_mel(source.waveform, config.mel)
        centered_mel = generate_log_mel(centered, config.mel)

        recurrence_started = time.perf_counter()
        recurrence = generate_recurrence(source.waveform, config.recurrence)
        recurrence_seconds = time.perf_counter() - recurrence_started
        cropped_recurrence = generate_recurrence(centered, config.recurrence)

        graph_started = time.perf_counter()
        graph = generate_visibility_graph(source.waveform, config.visibility_graph)
        graph_seconds = time.perf_counter() - graph_started
        cropped_graph = generate_visibility_graph(centered, config.visibility_graph)

        sample_duration = source.waveform.size / config.mel.sample_rate
        manifest_duration = _duration(record)
        expected_cropped = sample_duration - config.mel.fixed_duration_seconds
        crop_start = start / config.mel.sample_rate
        crop_end = stop / config.mel.sample_rate
        edge_pairs = {tuple(int(value) for value in edge) for edge in graph.edge_index.T}
        edge_indices_valid = bool(
            graph.edge_index.shape[0] == 2
            and np.all(graph.edge_index >= 0)
            and np.all(graph.edge_index < graph.node_features.shape[0])
        )
        bidirectional = all(
            (target_node, source_node) in edge_pairs
            for source_node, target_node in edge_pairs
        )

        assert mel.time_handling == "cropped"
        assert abs(mel.cropped_seconds - expected_cropped) <= 1e-12
        assert abs(expected_cropped - (manifest_duration - 8.0)) <= 1 / 16_000
        assert centered.size == 128_000
        assert mel.fixed_frame_count == 798
        assert np.array_equal(mel.image, repeated_mel.image)
        assert np.array_equal(mel.image, centered_mel.image)
        assert mel.image.shape == (224, 224) and mel.image.dtype == np.float32
        assert np.all(np.isfinite(mel.image))
        assert recurrence.image.shape == (224, 224)
        assert recurrence.image.dtype == np.uint8
        assert np.array_equal(recurrence.image, recurrence.image.T)
        assert set(np.unique(recurrence.image)).issubset({0, 1})
        assert np.all(np.isfinite(recurrence.image))
        assert not np.array_equal(recurrence.image, cropped_recurrence.image)
        assert graph.node_features.shape[0] <= 256
        assert graph.node_features.dtype == np.float32
        assert graph.edge_index.dtype == np.int16
        assert edge_indices_valid and len(edge_pairs) == graph.edge_index.shape[1]
        assert all(source_node != target_node for source_node, target_node in edge_pairs)
        assert bidirectional
        assert not (
            np.array_equal(graph.node_features, cropped_graph.node_features)
            and np.array_equal(graph.edge_index, cropped_graph.edge_index)
        )

        item: dict[str, Any] = {
            "logical_id": logical_id,
            "condition": record.condition.value,
            "gender": record.gender.value,
            "speaker": record.speaker_id,
            "duration_seconds": manifest_duration,
            "source": {
                "reconstruction_path": source.relative_path,
                "reconstruction_sha256": source.artifact_sha256,
                "sample_count": int(source.waveform.size),
                "sample_duration_seconds": sample_duration,
                "complete_reconstruction_passed_to_recurrence": True,
                "complete_reconstruction_passed_to_visibility_graph": True,
            },
            "mel": {
                "shape": list(mel.image.shape),
                "dtype": str(mel.image.dtype),
                "finite": bool(np.all(np.isfinite(mel.image))),
                "time_handling": mel.time_handling,
                "cropped_seconds": mel.cropped_seconds,
                "fixed_sample_count": centered.size,
                "fixed_frame_count": mel.fixed_frame_count,
                "crop_start_sample": start,
                "crop_stop_sample_exclusive": stop,
                "crop_start_seconds": crop_start,
                "crop_end_seconds": crop_end,
                "deterministic": bool(np.array_equal(mel.image, repeated_mel.image)),
                "equals_explicit_center_slice_transform": bool(
                    np.array_equal(mel.image, centered_mel.image)
                ),
                "minimum": float(np.min(mel.image)),
                "maximum": float(np.max(mel.image)),
            },
            "recurrence": {
                "shape": list(recurrence.image.shape),
                "dtype": str(recurrence.image.dtype),
                "symmetric": bool(np.array_equal(recurrence.image, recurrence.image.T)),
                "binary": bool(set(np.unique(recurrence.image)).issubset({0, 1})),
                "finite": bool(np.all(np.isfinite(recurrence.image))),
                "density": recurrence.recurrence_density,
                "input_sample_count": int(source.waveform.size),
                "differs_from_center_crop_reference": bool(
                    not np.array_equal(recurrence.image, cropped_recurrence.image)
                ),
            },
            "visibility_graph": {
                "node_count": int(graph.node_features.shape[0]),
                "directed_edge_count": int(graph.edge_index.shape[1]),
                "undirected_edge_count": int(graph.edge_index.shape[1] // 2),
                "density": graph.graph_density,
                "isolated_node_count": graph.isolated_node_count,
                "connected_component_count": graph.connected_component_count,
                "edge_indices_valid": edge_indices_valid,
                "edges_unique": len(edge_pairs) == graph.edge_index.shape[1],
                "bidirectional": bidirectional,
                "self_loops": sum(
                    source_node == target_node for source_node, target_node in edge_pairs
                ),
                "input_sample_count": int(source.waveform.size),
                "differs_from_center_crop_reference": bool(
                    not (
                        np.array_equal(graph.node_features, cropped_graph.node_features)
                        and np.array_equal(graph.edge_index, cropped_graph.edge_index)
                    )
                ),
            },
            "timing_seconds": {
                "mel": mel_seconds,
                "recurrence": recurrence_seconds,
                "visibility_graph": graph_seconds,
                "representations_total": mel_seconds + recurrence_seconds + graph_seconds,
            },
        }
        panel.append(item)

        if logical_id == longest_id:
            write_crop_audit_diagnostic_svg(
                PLOT_PATH,
                logical_id=logical_id,
                condition=record.condition.value,
                duration_seconds=sample_duration,
                crop_start_seconds=crop_start,
                crop_end_seconds=crop_end,
                mel=mel.image,
                recurrence=recurrence.image,
                graph_nodes=graph.node_features,
                graph_edges=graph.edge_index,
            )

    longest = records_by_id[longest_id]
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "module": "2_multirepresentation_generation",
        "status": "PASS",
        "purpose": "real_production_center_crop_pre_full_build_gate",
        "source": {
            "subset": "core",
            "core_record_count": contract.signed_record_count,
            "manifest_fingerprint": contract.manifest_fingerprint,
            "module1_config_hash": config.source.module1_config_hash,
            "representation_config_hash": config.config_hash,
        },
        "long_record_audit": {
            "threshold_seconds": 8.0,
            "count": len(long_records),
            "percentage_of_core": 100.0 * len(long_records) / contract.signed_record_count,
            "condition_counts": dict(
                sorted(Counter(record.condition.value for record in long_records).items())
            ),
            "gender_counts": dict(
                sorted(Counter(record.gender.value for record in long_records).items())
            ),
            "speaker_counts": dict(
                sorted(Counter(record.speaker_id for record in long_records).items())
            ),
            "duration_seconds": {
                "min": float(np.min(durations)),
                "median": float(np.median(durations)),
                "mean": float(np.mean(durations)),
                "p90": float(np.quantile(durations, 0.90)),
                "p95": float(np.quantile(durations, 0.95)),
                "p99": float(np.quantile(durations, 0.99)),
                "max": float(np.max(durations)),
            },
            "threshold_counts": {
                ">8": int(np.count_nonzero(durations > 8.0)),
                ">10": int(np.count_nonzero(durations > 10.0)),
                ">12": int(np.count_nonzero(durations > 12.0)),
                ">20": int(np.count_nonzero(durations > 20.0)),
                ">30": int(np.count_nonzero(durations > 30.0)),
            },
        },
        "panel_selection": {
            "method": (
                "fixed descriptive duration/condition/gender strata selected without "
                "representation or model performance"
            ),
            "logical_ids": list(CROP_PANEL_IDS),
        },
        "records": panel,
        "longest_record": {
            "logical_id": longest.logical_id,
            "condition": longest.condition.value,
            "gender": longest.gender.value,
            "speaker": longest.speaker_id,
            "prompt_type": longest.prompt_type.value,
            "transcript": longest.transcript,
            "prompt_path": longest.prompt_path,
            "primary_microphone": (
                longest.primary_microphone.value if longest.primary_microphone else None
            ),
            "usability_status": longest.usability_status.value,
            "duration_seconds": _duration(longest),
            "diagnostic_path": PLOT_PATH.relative_to(REPO_ROOT).as_posix(),
        },
        "policy_decision": {
            "mel_fixed_duration_seconds": 8.0,
            "decision": "RETAINED",
            "rationale": (
                "The real crop path passed; 97.2904% of CORE avoids cropping, while 8 s "
                "preserves about 35.7 ms per final time pixel. The 235 cropped records "
                "remain a documented Module 5 robustness limitation."
            ),
        },
    }
    _write_json_atomic(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _duration(record: Any) -> float:
    duration = record.duration_seconds
    if not isinstance(duration, float):
        raise ValueError(f"missing duration for {record.logical_id}")
    return duration


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
