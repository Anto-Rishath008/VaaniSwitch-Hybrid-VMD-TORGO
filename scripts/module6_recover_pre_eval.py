"""Validate the retained partial M6 run without recognition or synthesis execution.

Writes new recovery evidence only. Never creates a PASS freeze or starts final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vaaniswitch.m6_reconstruction.artifacts import digest, sha256, verify_checksum, write_json
from vaaniswitch.m6_reconstruction.audio import decode_pcm, validate_generated_wav
from vaaniswitch.m6_reconstruction.config import MODEL_SHA, VOICE_ID, load_config
from vaaniswitch.m6_reconstruction.metrics import METRIC_SHA, aggregate, score
from vaaniswitch.m6_reconstruction.normalization import NORMALIZATION, NORMALIZATION_SHA
from vaaniswitch.m6_reconstruction.phase234_compat import (
    audio_job,
    build_reference_inventory,
    inherited_group,
    load_manifest,
    signed_core_ids,
)
from vaaniswitch.m6_reconstruction.phase234_compat import score as compatibility_score
from vaaniswitch.m6_reconstruction.references import REFERENCE_LIMITATION
from vaaniswitch.m6_reconstruction.schemas import DEMO_IDS, DEVELOPMENT_IDS, PANEL_SHA

PARENT = "artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z"
BASE = "artifacts/module6/recovery_20261008"


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def recover(root: Path) -> Path:
    started = datetime.now(UTC).isoformat()
    parent = root / PARENT
    baseline = read(root / BASE / "recovery_baseline.json")
    preserved = {p: h for p, h in baseline["files"].items() if p.startswith(PARENT + "/")}
    for p, value in preserved.items():
        verify_checksum(root / p, value["sha256"])
    historical = read(root / "artifacts/module6/pre_evaluation_work/baseline.json")
    for p, expected in historical["retained_files"].items():
        verify_checksum(root / p, expected)
    config = load_config(root / "configs/module6/pipeline.yaml", root)
    verify_checksum(config.checkpoint, MODEL_SHA)
    delivery = read(root / "artifacts/module6/setup/delivery_manifest.json")["files_sha256"]
    for p in (
        "configs/module6/pipeline.yaml",
        "src/vaaniswitch/m6_reconstruction/asr.py",
        "configs/module6/environment/requirements-win-py312.lock",
    ):
        verify_checksum(root / p, delivery[p])
    assert config.identity == "60db6bb2d5d8ae87716e1bbd3b990a5935677eac294acfc616d2458410156209"
    manifest = load_manifest(root)
    assert len({row["speaker_id"] for row in manifest.values()}) == 15
    inventory = build_reference_inventory(root, manifest)
    retained_inventory = [
        json.loads(line)
        for line in (parent / "reference_eligibility.jsonl").read_bytes().splitlines()
    ]
    assert retained_inventory == [inventory[k].to_dict() for k in sorted(inventory)]
    core = signed_core_ids(root, manifest)
    types = Counter(manifest[k]["prompt_type"] for k, ref in inventory.items() if ref.eligible)
    assert types == {"word": 6458, "sentence": 2104}
    assert read(parent / "normalizer.json") == {"spec": NORMALIZATION, "sha256": NORMALIZATION_SHA}
    jobs = [
        json.loads(line) for line in (parent / "recognition_jobs.jsonl").read_bytes().splitlines()
    ]
    assert jobs == [audio_job(manifest[k]) for k in DEVELOPMENT_IDS]
    probe = read(parent / "model_probe.json")
    assert probe["device"] == "cuda" and probe["dtype"] == "float16"
    assert probe["cuda_device_name"] == "NVIDIA GeForce RTX 4070 Laptop GPU"
    assert probe["all_parameters_frozen"] is True and probe["cuda_allocation"] is True
    phase0 = read(root / "docs/modules/6_english_asr_tts/PHASE0_EVIDENCE.json")
    expected_audio = phase0["panel_audio_verification"]["primary_audio_hashes"]
    stored_summary = read(parent / "metrics_raw.json")
    stored_rows = {r["metadata"]["logical_id"]: r for r in stored_summary["records"]}
    rows: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for logical_id in DEVELOPMENT_IDS:
        row = manifest[logical_id]
        directory = parent / "records" / logical_id
        record = read(directory / "raw_asr.json")
        raw = (directory / "raw_asr.txt").read_bytes()
        backend = read(directory / "backend_result.json")
        assert record["logical_id"] == logical_id
        assert raw == record["raw_text"].encode("utf-8") == backend["text"].encode("utf-8")
        verify_checksum(directory / "raw_asr.txt", record["raw_text_sha256"])
        verify_checksum(directory / "backend_result.json", record["backend_result_sha256"])
        source = root / row["primary_audio_path"]
        verify_checksum(source, expected_audio[logical_id])
        assert record["source_audio_sha256"] == row["primary_audio_sha256"] == sha256(source)
        values, _ = decode_pcm(source.read_bytes(), source=True)
        assert len(values) == record["input_frames"] == row["frame_count"]
        assert hashlib.sha256(values.tobytes()).hexdigest() == record["input_float32_sha256"]
        assert record["checkpoint_sha256"] == MODEL_SHA
        assert record["device"] == "cuda" and record["dtype"] == "float16"
        assert record["reference_context_used"] is False
        assert record["elapsed_seconds"] > 0
        reference = inventory[logical_id]
        assert read(directory / "evaluation/reference.json") == reference.to_dict()
        assert (directory / "evaluation/reference_source.txt").read_bytes() == (
            reference.source_text or ""
        ).encode("utf-8")
        assert (directory / "evaluation/reference.txt").read_bytes() == (
            reference.scoring_text or ""
        ).encode("utf-8")
        old_metric = compatibility_score(
            logical_id, reference.scoring_text, raw.decode("utf-8")
        ).to_dict()
        assert old_metric == read(directory / "evaluation/metrics_raw.json")
        assert old_metric == stored_rows[logical_id]["metric"]
        assert old_metric["metric_config_sha256"] == METRIC_SHA
        provenance = record["provenance"]
        if provenance["type"] == "PHASE1_REUSE":
            source_run = root / provenance["source_run"]
            completion = read(source_run / "RUNTIME_COMPLETE.json")
            assert completion["status"] == "PASS"
            assert completion["identity"]["config"] == config.identity
            verify_checksum(source_run / "asr_smoke.json", provenance["source_asr_sha256"])
            assert raw == (source_run / "raw_asr.txt").read_bytes()
            assert (directory / "backend_result.json").read_bytes() == (
                source_run / "backend_result.json"
            ).read_bytes()
        rows.append(
            {
                "logical_id": logical_id,
                "speaker_id": row["speaker_id"],
                "condition": row["condition"],
                "prompt_type": row["prompt_type"],
                "inherited_group": inherited_group(row["speaker_id"]),
                "metrics": score(reference.to_dict(), raw.decode("utf-8")),
            }
        )
        retained.append(
            {
                **record,
                "validation": "PASS",
                "config_identity": config.identity,
                "recovery_new_forward": False,
                "raw_metadata_sha256": sha256(directory / "raw_asr.json"),
            }
        )
    with (parent / "metrics_raw.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [r["logical_id"] for r in csv_rows] == list(DEVELOPMENT_IDS)
    for r in csv_rows:
        metric = stored_rows[r["logical_id"]]["metric"]
        assert float(r["wer"]) == metric["wer"] and float(r["cer"]) == metric["cer"]
    metrics = aggregate(rows, scope="DEVELOPMENT / SMOKE; NOT FINAL BENCHMARK")
    assert metrics["corpus"]["wer"]["ratio"] == stored_summary["summary"]["micro"]["wer"]
    assert metrics["corpus"]["cer"]["ratio"] == stored_summary["summary"]["micro"]["cer"]
    demos: dict[str, Any] = {}
    for logical_id in DEMO_IDS:
        directory = parent / "records" / logical_id
        synthesis = read(directory / "tts_raw/synthesis.json")
        wav = directory / "tts_raw/clear.wav"
        assert synthesis["validation"] == validate_generated_wav(wav)
        verify_checksum(wav, synthesis["wav_sha256"])
        assert synthesis["actual_settings"] == {"voice_id": VOICE_ID, "rate": 175, "volume": 1.0}
        assert synthesis["selected_voice"]["id"] == VOICE_ID
        assert synthesis["backend"] == "pyttsx3" and synthesis["package_version"] == "2.99"
        assert synthesis["driver"] == "sapi5"
        assert synthesis["reference_used_for_synthesis"] is False
        assert synthesis["source"] == "RAW_ASR_TRANSCRIPT"
        assert (directory / "tts_raw/synthesis_text.txt").read_bytes() == (
            directory / "raw_asr.txt"
        ).read_bytes()
        assert synthesis["synthesis_text_sha256"] == sha256(directory / "raw_asr.txt")
        demos[logical_id] = {**synthesis, "recovery_validation": "PASS"}
    assert not list(parent.rglob("correction.json"))
    ids = phase0["proposed_panel"]["evaluation_ids"]
    assert len(ids) == len(set(ids)) == 130
    assert digest(ids) == PANEL_SHA == phase0["proposed_panel"]["ids_sha256"]
    speakers = {manifest[k]["speaker_id"] for k in ids}
    assert len(speakers) == 13 and not speakers & {"M03", "MC02"}
    conditions = Counter(manifest[k]["condition"] for k in ids)
    prompt_types = Counter(manifest[k]["prompt_type"] for k in ids)
    assert conditions == {"dysarthric": 70, "control": 60}
    assert prompt_types == {"word": 65, "sentence": 65}
    duration = sum(manifest[k]["frame_count"] for k in ids) / 16000
    assert duration == 545.6203125
    failures = []
    for logical_id in ids:
        row = manifest[logical_id]
        verify_checksum(root / row["primary_audio_path"], expected_audio[logical_id])
        assert row["primary_audio_sha256"] == expected_audio[logical_id]
        assert inventory[logical_id].eligible
        audio_job(row)
        try:
            values, _ = decode_pcm((root / row["primary_audio_path"]).read_bytes(), source=True)
            assert len(values) == row["frame_count"]
        except ValueError as exc:
            failures.append(
                {"logical_id": logical_id, "path": row["primary_audio_path"], "error": str(exc)}
            )
    assert not list((root / "artifacts/module6/final_evaluation").glob("*"))
    preservation = read(root / BASE / "preservation_after.json")
    assert preservation["status"] == "PASS"
    code = {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted((root / "src/vaaniswitch/m6_reconstruction").glob("*.py"))
    }
    run = root / BASE / ("recovered_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    phase2 = {
        "status": "PASS",
        "manifest_records": len(manifest),
        "speakers": 15,
        "signed_core_records": len(core),
        "core_image_exclusions": 111,
        "eligible_references": sum(r.eligible for r in inventory.values()),
        "eligible_prompt_types": dict(types),
        "normalizer_sha256": NORMALIZATION_SHA,
        "metric_config_sha256": METRIC_SHA,
        "reference_limitation": REFERENCE_LIMITATION,
        "reference_inventory_sha256": sha256(parent / "reference_eligibility.jsonl"),
    }
    phase3 = {
        "status": "PASS",
        "scope": "DEVELOPMENT / SMOKE; NOT FINAL BENCHMARK",
        "retained_records": retained,
        "parent": PARENT,
        "new_development_forwards": 0,
        "partial_parent_new_development_forwards": 3,
        "phase1_reused_records": 1,
        "config_binding_method": "Recovery verification of unchanged frozen YAML and ASR "
        "implementation against Phase-1 hashes; raw metadata lacked a config field. "
        "This is a recovery binding, not a contemporaneous per-record config seal.",
        "references_entered_recognition": False,
        "raw_metrics": metrics,
    }
    phase4 = {
        "status": "FAIL" if failures else "PENDING_FINAL_GATE",
        "tts_mechanical_validation": "PASS",
        "demos": demos,
        "correction": "DISABLED / NOT EVALUATED",
        "corrections_created": 0,
        "human_listening": "NOT PERFORMED",
        "preflight": "FAIL" if failures else "PENDING",
    }
    preflight = {
        "status": "FAIL" if failures else "PENDING",
        "panel_identity": PANEL_SHA,
        "ordered_panel_ids": ids,
        "panel_membership_validation": "PASS",
        "panel_records": 130,
        "speakers": sorted(speakers),
        "conditions": dict(conditions),
        "prompt_types": dict(prompt_types),
        "source_duration_seconds": duration,
        "source_hashes": "PASS",
        "reference_eligibility": "PASS",
        "audio_only_jobs": "PASS",
        "audio_format_validation": "FAIL" if failures else "PASS",
        "audio_failures": failures,
        "model_sha256": MODEL_SHA,
        "config_identity": config.identity,
        "normalizer_sha256": NORMALIZATION_SHA,
        "metric_config_sha256": METRIC_SHA,
        "environment_lock_sha256": sha256(
            root / "configs/module6/environment/requirements-win-py312.lock"
        ),
        "code_identity": {"files": code, "sha256": digest(code)},
        "preservation": preservation,
        "evaluation_panel_forwards": 0,
        "cuda_allocation": "NOT RUN: stopped at source-audio gate",
        "existing_final_evaluation_outputs": 0,
    }
    freeze = {
        "status": "BLOCKED",
        "valid_for_execution": False,
        "reason": "Source audio violates the explicit declared-RIFF <= physical-byte gate",
        "parent_run": PARENT,
        "preflight": preflight,
        "phase2": phase2,
        "phase3": {"status": "PASS", "new_forwards": 0},
        "phase4": phase4,
        "panel_identity": PANEL_SHA,
        "final_evaluation_status": "NOT STARTED",
        "evaluation_panel_whisper_forwards": 0,
    }
    runtime = [r["elapsed_seconds"] for r in retained]
    recovery = {
        "status": "RECOVERED; FINAL GATE BLOCKED",
        "started_utc": started,
        "parent_runs_found": baseline["partial_runs"],
        "parent_run": PARENT,
        "preserved_partial_files": len(preserved),
        "changed_partial_files": [],
        "preserved_phase1_files": len(historical["retained_files"]),
        "new_development_forwards": 0,
        "evaluation_panel_forwards": 0,
        "new_tts_synthesis": 0,
        "runtime_retained_asr_seconds": sum(runtime),
        "runtime_retained_asr_min": min(runtime),
        "runtime_retained_asr_median": statistics.median(runtime),
        "runtime_retained_asr_max": max(runtime),
        "runtime_model_load_seconds": probe["model_load_seconds"],
        "runtime_tts_synthesis_seconds": None,
        "runtime_tts_note": "Not recorded in partial-run synthesis metadata",
        "module5_executed": False,
        "module7_started": False,
        "original_files": preserved,
        "raw_records": retained,
    }
    for filename, payload in [
        ("RECOVERY_VALIDATION.json", recovery),
        ("PHASE2_EVIDENCE.json", phase2),
        ("PHASE3_EVIDENCE.json", phase3),
        ("PHASE4_EVIDENCE.json", phase4),
        ("metrics_development_verified.json", metrics),
        ("FINAL_EVALUATION_PREFLIGHT.json", preflight),
        ("PRE_EVALUATION_FREEZE.json", freeze),
        (
            "FINAL_EVALUATION_STATUS.json",
            {
                "status": "NOT STARTED / BLOCKED",
                "requested_records": 130,
                "attempted_records": 0,
                "successful_records": 0,
                "asr_failures": 0,
                "scored_records": 0,
                "evaluation_exclusions": 0,
                "preflight_audio_failures": 5,
                "unattempted_records": 130,
                "raw_micro_wer": None,
                "raw_micro_cer": None,
                "runtime_seconds": None,
                "human_listening": "PENDING",
                "module6_signoff": "PENDING",
            },
        ),
    ]:
        write_json(root, run / filename, payload)
    freeze_path = run / "PRE_EVALUATION_FREEZE.json"
    write_json(
        root,
        run / "PRE_EVALUATION_FREEZE_SHA.json",
        {
            "path": freeze_path.relative_to(root).as_posix(),
            "sha256": sha256(freeze_path),
            "status": "BLOCKED; NOT AN EXECUTION AUTHORIZATION",
        },
    )
    for p, value in preserved.items():
        verify_checksum(root / p, value["sha256"])
    print(
        json.dumps(
            {
                "run": run.relative_to(root).as_posix(),
                "status": freeze["status"],
                "freeze_sha256": sha256(freeze_path),
                "audio_failures": failures,
            },
            indent=2,
        )
    )
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    recover(args.root.resolve())
