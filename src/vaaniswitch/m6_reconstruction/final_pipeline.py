"""Gated recovery/preflight and final evaluation; raw recognition precedes all final scoring."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import digest, sha256, verify_checksum, write_json
from .audio import validate_generated_wav
from .config import DECODING, MODEL_SHA, VOICE_ID, load_config
from .final_state import (
    claim_run,
    exclusive_run,
    persist,
    read_json,
    recognize,
    require_terminal,
)
from .metrics import METRIC_SHA
from .normalization import NORMALIZATION_SHA
from .schemas import DEMO_IDS, DEVELOPMENT_IDS, PANEL_SHA, AudioJob
from .source_audio import EXCEPTION_ID, EXCEPTION_IDS, SourceReader

BASE = "artifacts/module6/exception_evaluation_20261008"
PARENT = "artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z"
OLD = "artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z"
OLD_FREEZE_SHA = "385c3be9ec4cdee0b5397d9cac9dc4d713952f945125592a7145940f5ee5d169"
OLD_DELIVERY_SHA = "1aee03f9f481bf5fce66cab548e30984fbc9d4c72007b54be0a5f396c9c07034"
CONFIG_SHA = "60db6bb2d5d8ae87716e1bbd3b990a5935677eac294acfc616d2458410156209"
LOCK_SHA = "49cc7da351e802a277bd5435dc520c35e1ad15b36fc8d52b4c6f9cc0bc5d6af1"
GPU = "NVIDIA GeForce RTX 4070 Laptop GPU"


def code_identity(root: Path) -> dict[str, Any]:
    files = list((root / "src/vaaniswitch/m6_reconstruction").glob("*.py"))
    files += list((root / "tests/m6_reconstruction").glob("*.py"))
    files += list((root / "scripts").glob("module6*.py"))
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}
    return {"files": hashes, "sha256": digest(hashes)}


def _quality(root: Path) -> dict[str, Any]:
    path = root / BASE / "QUALITY_GATES.json"
    value = read_json(path)
    required = {
        "pytest",
        "retained_runtime",
        "ruff",
        "mypy",
        "compileall",
        "whitespace",
        "dependencies",
        "runner_resume_tests",
        "source_exception_tests",
    }
    if set(value["checks"]) != required or any(v != "PASS" for v in value["checks"].values()):
        raise ValueError("all explicit quality gates must pass")
    if value["code_identity"] != code_identity(root):
        raise ValueError("code changed after quality checks")
    for name, expected in value["log_sha256"].items():
        verify_checksum(root / BASE / name, expected)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), **value}


def _environment(root: Path) -> dict[str, Any]:
    if Path(sys.prefix).resolve() != (root / ".venv-m6").resolve():
        raise RuntimeError("isolated .venv-m6 interpreter required")
    packages = dict(
        sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    )
    phase1 = root / (
        "artifacts/module6/60db6bb2d5d8ae87716e1bbd3b990a5935677eac294acfc616d2458410156209/"
        "phase1_20261008T035618759610Z/environment.json"
    )
    previous = read_json(phase1)
    if packages != previous["packages"] or platform.python_version() != "3.12.2":
        raise ValueError("isolated Python/package identity drift")
    verify_checksum(root / "configs/module6/environment/requirements-win-py312.lock", LOCK_SHA)
    return {
        "packages": packages,
        "python": platform.python_version(),
        "lock_sha256": LOCK_SHA,
        "python_executable_sha256": sha256(Path(sys.executable)),
        "packages_sha256": digest(packages),
    }


def _cuda() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.version.cuda != "12.6":
        raise RuntimeError("frozen CUDA runtime unavailable")
    if torch.cuda.get_device_name(0) != GPU:
        raise RuntimeError("wrong GPU")
    tensor = torch.ones(8, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    if tensor.sum().item() != 8:
        raise RuntimeError("CUDA allocation failed")
    del tensor
    return {
        "status": "PASS",
        "gpu": GPU,
        "device": "cuda",
        "dtype": "float16",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }


def _recovery(root: Path, reader: SourceReader) -> dict[str, Any]:
    verify_checksum(root / OLD / "DELIVERY_MANIFEST.json", OLD_DELIVERY_SHA)
    old_files = read_json(root / OLD / "DELIVERY_MANIFEST.json")["files_sha256"]
    for relative, expected in old_files.items():
        if relative.startswith(OLD + "/"):
            verify_checksum(root / relative, expected)
    verify_checksum(root / OLD / "PRE_EVALUATION_FREEZE.json", OLD_FREEZE_SHA)
    retained = read_json(root / OLD / "RECOVERY_VALIDATION.json")
    for relative, metadata in retained["original_files"].items():
        verify_checksum(root / relative, metadata["sha256"])
    for record in retained["raw_records"]:
        job = reader.jobs[record["logical_id"]]
        audio, _ = reader.load(job)
        if (
            hashlib.sha256(audio.tobytes()).hexdigest() != record["input_float32_sha256"]
            or record["reference_context_used"] is not False
            or record["checkpoint_sha256"] != MODEL_SHA
        ):
            raise ValueError("retained development source/model binding mismatch")
    if [r["logical_id"] for r in retained["raw_records"]] != list(DEVELOPMENT_IDS):
        raise ValueError("development record set/order drift")
    for key in DEMO_IDS:
        directory = root / PARENT / "records" / key
        metadata = read_json(directory / "tts_raw/synthesis.json")
        if metadata["validation"] != validate_generated_wav(directory / "tts_raw/clear.wav"):
            raise ValueError("retained TTS mechanical validation drift")
        if metadata["actual_settings"] != {"voice_id": VOICE_ID, "rate": 175, "volume": 1.0}:
            raise ValueError("retained TTS settings drift")
    return {
        "status": "PASS",
        "parent": PARENT,
        "old_recovery": OLD,
        "old_blocked_freeze_sha256": OLD_FREEZE_SHA,
        "reused_development_ids": list(DEVELOPMENT_IDS),
        "reused_tts_ids": list(DEMO_IDS),
        "new_development_forwards": 0,
        "new_tts_synthesis": 0,
        "phase2": "PASS",
        "phase3": "PASS",
        "tts_mechanical": "PASS",
    }


def prepare(root: Path) -> Path:
    # Imports are evaluator/coordinator-only; raw state machine consumes AudioJob exclusively.
    from .phase234 import _panel, _preservation
    from .phase234_compat import build_reference_inventory, load_manifest

    quality = _quality(root)
    reader = SourceReader(root)
    recovery = _recovery(root, reader)
    config = load_config(root / "configs/module6/pipeline.yaml", root)
    if config.identity != CONFIG_SHA or config.device != "cuda":
        raise ValueError("frozen execution config drift")
    verify_checksum(config.checkpoint, MODEL_SHA)
    environment = _environment(root)
    manifest = load_manifest(root)
    ids = _panel(root, manifest)
    duration = sum(manifest[k]["frame_count"] for k in ids) / 16000
    if duration != 545.6203125:
        raise ValueError("panel duration drift")
    inventory = build_reference_inventory(root, manifest)
    source_validation = []
    jobs = []
    for key in ids:
        if not inventory[key].eligible:
            raise ValueError(f"ineligible final reference: {key}")
        job = reader.jobs[key]
        _, metadata = reader.load(job)
        source_validation.append(metadata)
        jobs.append(job.to_dict())
    exceptions = [m for m in source_validation if m["exception_id"]]
    counts = Counter(m["format_status"] for m in source_validation)
    if (
        counts != {"STRICT_PASS": 125, EXCEPTION_ID: 5}
        or {m["logical_id"] for m in exceptions} != EXCEPTION_IDS
    ):
        raise ValueError("unexpected source-container exceptions")
    if (root / "artifacts/module6/final_evaluation/CLAIM.json").exists() or list(
        (root / "artifacts/module6/final_evaluation").glob("*/FINAL_EVALUATION_COMPLETE.json")
    ):
        raise ValueError("final evaluation already claimed/completed; new freeze prohibited")
    cuda = _cuda()
    preservation = _preservation(root)
    run = root / BASE / ("preflight_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    write_json(root, run / "RECOVERY_REUSE.json", recovery)
    write_json(root, run / "source_validation.json", source_validation)
    write_json(root, run / "source_exceptions.json", exceptions)
    write_json(root, run / "recognition_jobs.json", jobs)
    reference_identity = digest([inventory[k].to_dict() for k in ids])
    preflight = {
        "status": "PASS",
        "panel_ids": ids,
        "panel_sha256": PANEL_SHA,
        "records": 130,
        "speakers": 13,
        "conditions": {"dysarthric": 70, "control": 60},
        "prompt_types": {"word": 65, "sentence": 65},
        "duration_seconds": duration,
        "excluded_development_speakers": ["M03", "MC02"],
        "source_formats": dict(counts),
        "source_hashes": "PASS",
        "eligible_references": 130,
        "reference_inventory_sha256": reference_identity,
        "audio_only_jobs": True,
        "model_sha256": MODEL_SHA,
        "config_identity": config.identity,
        "environment": environment,
        "normalizer_sha256": NORMALIZATION_SHA,
        "metric_sha256": METRIC_SHA,
        "code_identity": code_identity(root),
        "cuda": cuda,
        "preservation": preservation,
        "quality": quality,
        "accepted_final_completions": 0,
        "evaluation_panel_forwards": 0,
    }
    write_json(root, run / "FINAL_EVALUATION_PREFLIGHT.json", preflight)
    freeze = {
        "schema": "m6_final_freeze_v2",
        "status": "PASS",
        "phase2": "PASS",
        "phase3": "PASS",
        "phase4": "PASS",
        "preflight": preflight,
        "old_blocked_freeze": {
            "path": OLD + "/PRE_EVALUATION_FREEZE.json",
            "sha256": OLD_FREEZE_SHA,
        },
        "source_validation_sha256": sha256(run / "source_validation.json"),
        "recognition_jobs_sha256": sha256(run / "recognition_jobs.json"),
        "recovery_reuse_sha256": sha256(run / "RECOVERY_REUSE.json"),
        "created_utc": datetime.now(UTC).isoformat(),
        "final_evaluation_status": "NOT STARTED",
        "correction": "DISABLED / NOT EVALUATED",
        "human_listening": "PENDING",
        "module5_executed": False,
        "module7_started": False,
    }
    path = write_json(root, run / "PRE_EVALUATION_FREEZE.json", freeze)
    write_json(
        root,
        run / "PRE_EVALUATION_FREEZE_SHA.json",
        {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)},
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "freeze": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "source_formats": dict(counts),
            },
            indent=2,
        ),
        flush=True,
    )
    return path


def verify_freeze(
    root: Path, path: Path, expected_sha: str
) -> tuple[dict[str, Any], list[AudioJob]]:
    verify_checksum(path, expected_sha)
    freeze = read_json(path)
    if freeze.get("schema") != "m6_final_freeze_v2" or any(
        freeze.get(k) != "PASS" for k in ("status", "phase2", "phase3", "phase4")
    ):
        raise ValueError("all freeze gates must be PASS")
    pf = freeze["preflight"]
    if (
        pf["status"] != "PASS"
        or pf["code_identity"] != code_identity(root)
        or pf["quality"] != _quality(root)
        or pf["environment"] != _environment(root)
        or pf["preservation"]["status"] != "PASS"
    ):
        raise ValueError("freeze code/environment/quality/preservation mismatch")
    config = load_config(root / "configs/module6/pipeline.yaml", root)
    if (
        config.identity != pf["config_identity"]
        or config.identity != CONFIG_SHA
        or pf["normalizer_sha256"] != NORMALIZATION_SHA
        or pf["metric_sha256"] != METRIC_SHA
        or pf["model_sha256"] != MODEL_SHA
        or digest(pf["panel_ids"]) != PANEL_SHA
    ):
        raise ValueError("frozen scientific identity mismatch")
    verify_checksum(config.checkpoint, MODEL_SHA)
    for filename, field in (
        ("source_validation.json", "source_validation_sha256"),
        ("recognition_jobs.json", "recognition_jobs_sha256"),
        ("RECOVERY_REUSE.json", "recovery_reuse_sha256"),
    ):
        verify_checksum(path.parent / filename, freeze[field])
    jobs = [
        AudioJob.from_dict(j)
        for j in json.loads((path.parent / "recognition_jobs.json").read_bytes())
    ]
    if [j.logical_id for j in jobs] != pf["panel_ids"]:
        raise ValueError("freeze ordered job mismatch")
    reader = SourceReader(root)
    actual = [reader.load(j)[1] for j in jobs]
    if actual != json.loads((path.parent / "source_validation.json").read_bytes()):
        raise ValueError("source validation drift after freeze")
    if _cuda() != pf["cuda"]:
        raise ValueError("CUDA identity drift")
    return freeze, jobs


def _evaluate(
    root: Path, run: Path, identity: dict[str, Any], jobs: list[AudioJob], freeze: dict[str, Any]
) -> dict[str, Any]:
    from .metrics import aggregate, score
    from .phase234_compat import build_reference_inventory, inherited_group, load_manifest

    records = require_terminal(root, run, identity, jobs)
    manifest = load_manifest(root)
    inventory = build_reference_inventory(root, manifest)
    if (
        digest([inventory[j.logical_id].to_dict() for j in jobs])
        != freeze["preflight"]["reference_inventory_sha256"]
    ):
        raise ValueError("evaluator reference identity drift")
    rows = []
    for job, record in zip(jobs, records, strict=True):
        row = manifest[job.logical_id]
        reference = inventory[job.logical_id].to_dict()
        metric = score(reference, record["raw_text"], asr_status=record["status"])
        directory = run / "records" / job.logical_id / "evaluation"
        persist(root, directory / "reference.json", reference)
        persist(root, directory / "metrics_raw.json", metric)
        rows.append(
            {
                "logical_id": job.logical_id,
                "speaker_id": row["speaker_id"],
                "condition": row["condition"],
                "prompt_type": row["prompt_type"],
                "inherited_group": inherited_group(row["speaker_id"]),
                "metrics": metric,
            }
        )
    summary = aggregate(rows, scope="FINAL FROZEN 130-RECORD EVALUATION; RAW ASR ONLY")
    successful = [r for r in records if r["status"] == "COMPLETE"]
    times = [r["elapsed_seconds"] for r in successful]
    summary["execution_counts"] = {
        "requested": len(jobs),
        "attempted": len(records),
        "successfully_transcribed": len(successful),
        "failed": len(records) - len(successful),
        "scored": summary["corpus"]["scored_records"],
        "exclusions": sum(
            not r["metrics"]["scored"] and r["metrics"]["exclusion_reason"] != "ASR_FAILURE"
            for r in rows
        ),
    }
    probes = [read_json(p) for p in sorted(run.glob("model_probe_*.json"))]
    summary["runtime"] = {
        "total_successful_asr_seconds": sum(times),
        "min_seconds": min(times) if times else None,
        "median_seconds": statistics.median(times) if times else None,
        "mean_seconds": statistics.mean(times) if times else None,
        "max_seconds": max(times) if times else None,
        "model_load_seconds": sum(p["model_load_seconds"] for p in probes),
        "model_loads": len(probes),
        "per_record_seconds": {r["logical_id"]: r.get("elapsed_seconds") for r in records},
    }
    summary["run_identity"] = identity
    summary["source_exceptions"] = sorted(EXCEPTION_IDS)
    summary["references_entered_recognition"] = False
    summary["correction_used"] = False
    persist(root, run / "metrics_raw.json", summary)
    return summary


def final_evaluation(root: Path, freeze_path: Path, expected_sha: str) -> Path:
    from .asr import load_frozen_model, transcribe_audio
    from .phase234 import _preservation

    freeze, jobs = verify_freeze(root, freeze_path, expected_sha)
    pf = freeze["preflight"]
    identity = {
        "schema": "m6_final_run_v2",
        "freeze_sha256": expected_sha,
        "freeze_path": freeze_path.relative_to(root).as_posix(),
        "panel_ids": pf["panel_ids"],
        "panel_sha256": PANEL_SHA,
        "code_identity": pf["code_identity"],
        "config_identity": pf["config_identity"],
        "environment": pf["environment"],
        "environment_lock_sha256": LOCK_SHA,
        "model_sha256": MODEL_SHA,
        "normalizer_sha256": NORMALIZATION_SHA,
        "metric_sha256": METRIC_SHA,
        "device": "cuda",
        "dtype": "float16",
        "gpu": GPU,
        "decoding": {**DECODING, "fp16": True},
        "source_exception_policy": EXCEPTION_ID,
    }
    config = load_config(root / "configs/module6/pipeline.yaml", root)
    reader = SourceReader(root)
    model: Any = None
    with exclusive_run(root):
        run = claim_run(root, identity, jobs)
        if (run / "FINAL_EVALUATION_COMPLETE.json").exists() or (
            run / "FINAL_EVALUATION_PARTIAL.json"
        ).exists():
            raise ValueError("final evaluation already terminal; no second execution")

        def infer(job: AudioJob) -> dict[str, Any]:
            nonlocal model
            audio, source = reader.load(job)
            if model is None:
                model, probe = load_frozen_model(config)
                write_json(
                    root,
                    run
                    / ("model_probe_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"),
                    probe,
                )
            result = transcribe_audio(model, audio, device="cuda")
            if result["decoding"] != identity["decoding"]:
                raise ValueError("decoding drift")
            result["source_validation"] = source
            return result

        try:
            recognize(root, run, identity, jobs, infer)
            summary = _evaluate(root, run, identity, jobs, freeze)
            preservation = _preservation(root)
            write_json(
                root,
                run / ("preservation_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"),
                preservation,
            )
            counts = summary["execution_counts"]
            passed = (
                counts
                == {
                    "requested": 130,
                    "attempted": 130,
                    "successfully_transcribed": 130,
                    "failed": 0,
                    "scored": 130,
                    "exclusions": 0,
                }
                and preservation["status"] == "PASS"
            )
            complete = {
                "status": "PASS" if passed else "PARTIAL",
                "counts": counts,
                "identity_sha256": digest(identity),
                "metrics_sha256": sha256(run / "metrics_raw.json"),
                "recognition_terminal_sha256": sha256(run / "RECOGNITION_TERMINAL.json"),
                "preservation": preservation,
                "source_exceptions": sorted(EXCEPTION_IDS),
                "new_development_forwards": 0,
                "new_tts_synthesis": 0,
                "human_listening": "PENDING",
                "module6_signoff": "PENDING HUMAN LISTENING" if passed else "PENDING",
                "module6_computational_status": "COMPLETE" if passed else "INCOMPLETE",
                "module5_executed": False,
                "module7_started": False,
            }
            filename = (
                "FINAL_EVALUATION_COMPLETE.json" if passed else "FINAL_EVALUATION_PARTIAL.json"
            )
            persist(root, run / filename, complete)
            print(
                json.dumps(
                    {
                        "run": run.relative_to(root).as_posix(),
                        "status": complete["status"],
                        "counts": counts,
                        "corpus": summary["corpus"],
                        "runtime": summary["runtime"],
                    },
                    indent=2,
                ),
                flush=True,
            )
        except BaseException as exc:
            write_json(
                root,
                run / ("INTERRUPTION_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"),
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "identity_sha256": digest(identity),
                    "resume_policy": "reuse validated successes; "
                    "claimed incomplete records become failures; execute only never-claimed jobs",
                },
            )
            raise
    return run
