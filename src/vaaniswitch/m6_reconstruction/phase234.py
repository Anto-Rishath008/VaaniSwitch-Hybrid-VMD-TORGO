"""Module 6 Phases 2-4 and frozen 130-record evaluation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import (
    canonical_bytes,
    digest,
    output_path,
    sha256,
    verify_checksum,
    write_bytes,
    write_json,
)
from .asr import load_frozen_model, transcribe_audio
from .audio import decode_pcm, validate_generated_wav
from .config import MODEL_SHA, VOICE_ID, load_config
from .phase234_compat import (
    NORMALIZER_SHA256,
    NORMALIZER_SPEC,
    aggregate,
    audio_job,
    inherited_group,
    score,
)

DEV_IDS = (
    "M03/Session2/0002",
    "M03/Session2/0013",
    "MC02/Session1/0007",
    "MC02/Session1/0006",
)
PHASE1_REUSE_ID = "M03/Session2/0013"
DEMO_IDS = ("M03/Session2/0013", "MC02/Session1/0006")


def _phase0(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        (root / "docs/modules/6_english_asr_tts/PHASE0_EVIDENCE.json").read_bytes()
    )
    return result


def _write_jsonl(root: Path, path: Path, rows: list[dict[str, Any]]) -> Path:
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    return write_bytes(root, path, payload)


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "logical_id": row["logical_id"],
        "speaker_id": row.get("speaker_id"),
        "speaker_group": inherited_group(str(row.get("speaker_id"))),
        "condition": row.get("condition"),
        "gender": row.get("gender"),
        "session_id": row.get("session_id"),
        "prompt_id": row.get("prompt_id"),
        "prompt_type": row.get("prompt_type"),
        "primary_microphone": row.get("primary_microphone"),
        "duration_seconds": row.get("duration_seconds"),
    }


def _validate_audio(root: Path, row: dict[str, Any], expected_sha: str | None = None) -> bytes:
    path = root / str(row["primary_audio_path"])
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = expected_sha or row["primary_audio_sha256"]
    if actual != expected or actual != row["primary_audio_sha256"]:
        raise ValueError(f"audio hash mismatch: {row['logical_id']}")
    audio, meta = decode_pcm(payload, source=True)
    if len(audio) != int(row["frame_count"]):
        raise ValueError(f"frame count mismatch: {row['logical_id']}")
    if meta["sample_rate"] != 16000 or meta["channels"] != 1 or meta["bit_depth"] != 16:
        raise ValueError(f"unexpected audio format: {row['logical_id']}")
    return payload


def _find_phase1_reuse(root: Path) -> Path:
    candidates: list[Path] = []
    for complete_path in (root / "artifacts/module6").rglob("RUNTIME_COMPLETE.json"):
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if complete.get("status") != "PASS" or PHASE1_REUSE_ID not in complete.get(
                "asr_records", []
            ):
                continue
            run = complete_path.parent
            asr = json.loads((run / "asr_smoke.json").read_text(encoding="utf-8"))
            if asr.get("record_id") != PHASE1_REUSE_ID or asr.get("checkpoint_sha256") != MODEL_SHA:
                continue
            if asr.get("device") != "cuda" or asr.get("reference_context_used") is not False:
                continue
            if (run / "raw_asr.txt").read_bytes() != asr["raw_text"].encode("utf-8"):
                continue
            candidates.append(run)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    if not candidates:
        raise RuntimeError("no valid retained Phase-1 ASR found for M03/Session2/0013")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def _persist_raw(
    root: Path,
    record_dir: Path,
    logical_id: str,
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    raw = write_bytes(root, record_dir / "raw_asr.txt", result["raw_text"].encode("utf-8"))
    backend = write_json(root, record_dir / "backend_result.json", result["backend_result"])
    record = {
        "logical_id": logical_id,
        "raw_text": result["raw_text"],
        "raw_text_sha256": sha256(raw),
        "backend_result_sha256": sha256(backend),
        "device": result["device"],
        "dtype": result["dtype"],
        "elapsed_seconds": result["elapsed_seconds"],
        "input_frames": result["input_frames"],
        "input_float32_sha256": result["input_float32_sha256"],
        "reference_context_used": result["reference_context_used"],
        "source_audio_sha256": row["primary_audio_sha256"],
        "checkpoint_sha256": MODEL_SHA,
        "provenance": provenance,
    }
    write_json(root, record_dir / "raw_asr.json", record)
    return record


def _reuse_phase1(root: Path, record_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    source = _find_phase1_reuse(root)
    asr = json.loads((source / "asr_smoke.json").read_text(encoding="utf-8"))
    raw_bytes = (source / "raw_asr.txt").read_bytes()
    backend_bytes = (source / "backend_result.json").read_bytes()
    if raw_bytes != asr["raw_text"].encode("utf-8"):
        raise ValueError("retained Phase-1 raw text mismatch")
    raw = write_bytes(root, record_dir / "raw_asr.txt", raw_bytes)
    backend = write_bytes(root, record_dir / "backend_result.json", backend_bytes)
    record = {
        "logical_id": PHASE1_REUSE_ID,
        "raw_text": asr["raw_text"],
        "raw_text_sha256": sha256(raw),
        "backend_result_sha256": sha256(backend),
        "device": asr["device"],
        "dtype": asr["dtype"],
        "elapsed_seconds": asr["elapsed_seconds"],
        "input_frames": asr["input_frames"],
        "input_float32_sha256": asr["input_float32_sha256"],
        "reference_context_used": asr["reference_context_used"],
        "source_audio_sha256": row["primary_audio_sha256"],
        "checkpoint_sha256": MODEL_SHA,
        "provenance": {
            "type": "PHASE1_REUSE",
            "source_run": source.relative_to(root).as_posix(),
            "source_asr_sha256": sha256(source / "asr_smoke.json"),
            "new_forward": False,
        },
    }
    write_json(root, record_dir / "raw_asr.json", record)
    return record


def _run_asr_for_ids(
    root: Path,
    run: Path,
    ids: list[str],
    manifest: dict[str, dict[str, Any]],
    *,
    reuse_phase1: bool,
) -> list[dict[str, Any]]:
    config = load_config(root / "configs/module6/pipeline.yaml", root)
    jobs = [audio_job(manifest[logical_id]) for logical_id in ids]
    _write_jsonl(root, run / "recognition_jobs.jsonl", jobs)
    for job in jobs:
        if set(job) != {"logical_id", "primary_audio_path", "primary_audio_sha256", "frame_count"}:
            raise AssertionError("recognition job leaked non-audio fields")
    results: list[dict[str, Any]] = []
    model: Any | None = None
    probe: dict[str, Any] | None = None
    for logical_id in ids:
        row = manifest[logical_id]
        record_dir = run / "records" / logical_id
        if reuse_phase1 and logical_id == PHASE1_REUSE_ID:
            results.append(_reuse_phase1(root, record_dir, row))
            continue
        payload = _validate_audio(root, row)
        audio, _ = decode_pcm(payload, source=True)
        if model is None:
            model, probe = load_frozen_model(config)
            write_json(root, run / "model_probe.json", probe)
        result = transcribe_audio(model, audio, device=config.device)
        results.append(
            _persist_raw(
                root,
                record_dir,
                logical_id,
                row,
                result,
                provenance={"type": "NEW_FROZEN_ASR_FORWARD", "new_forward": True},
            )
        )
    return results


def _evaluate_ids(
    root: Path,
    run: Path,
    ids: list[str],
    manifest: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    scored_rows: list[dict[str, Any]] = []
    for logical_id in ids:
        row = manifest[logical_id]
        record_dir = run / "records" / logical_id
        raw = (record_dir / "raw_asr.txt").read_text(encoding="utf-8")
        reference = inventory[logical_id]
        ref_dir = record_dir / "evaluation"
        ref_payload = reference.to_dict()
        write_json(root, ref_dir / "reference.json", ref_payload)
        if reference.source_text is not None:
            write_bytes(
                root, ref_dir / "reference_source.txt", reference.source_text.encode("utf-8")
            )
        if reference.scoring_text is not None:
            write_bytes(root, ref_dir / "reference.txt", reference.scoring_text.encode("utf-8"))
        metric = score(
            logical_id,
            reference.scoring_text if reference.eligible else None,
            raw,
            exclusion_reason=reference.exclusion_reason,
        ).to_dict()
        write_json(root, ref_dir / "metrics_raw.json", metric)
        scored_rows.append({"metadata": _metadata(row), "metric": metric})
    summary = {
        "records_requested": len(ids),
        "records_scored": sum(1 for x in scored_rows if x["metric"]["eligible"]),
        "micro": aggregate(scored_rows),
        "per_speaker": aggregate(scored_rows, ("speaker_id",)),
        "per_prompt_type": aggregate(scored_rows, ("prompt_type",)),
        "per_condition": aggregate(scored_rows, ("condition",)),
        "per_group": aggregate(scored_rows, ("speaker_group",)),
        "normalizer": {"spec": NORMALIZER_SPEC, "sha256": NORMALIZER_SHA256},
    }
    write_json(root, run / "metrics_raw.json", {"records": scored_rows, "summary": summary})
    _write_metrics_csv(root, run / "metrics_raw.csv", scored_rows)
    return summary


def _write_metrics_csv(root: Path, path: Path, rows: list[dict[str, Any]]) -> None:
    import io

    buffer = io.StringIO(newline="")
    fieldnames = [
        "logical_id",
        "speaker_id",
        "speaker_group",
        "condition",
        "prompt_type",
        "eligible",
        "wer",
        "cer",
        "exclusion_reason",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        meta, metric = row["metadata"], row["metric"]
        writer.writerow(
            {
                "logical_id": meta["logical_id"],
                "speaker_id": meta["speaker_id"],
                "speaker_group": meta["speaker_group"],
                "condition": meta["condition"],
                "prompt_type": meta["prompt_type"],
                "eligible": metric["eligible"],
                "wer": metric["wer"],
                "cer": metric["cer"],
                "exclusion_reason": metric["exclusion_reason"],
            }
        )
    write_bytes(root, path, buffer.getvalue().encode("utf-8"))


def _tts_worker(root: Path, run: Path, logical_id: str) -> None:
    import importlib.metadata

    import pyttsx3  # type: ignore[import-untyped]

    from .tts import select_required_voice, voice_record

    record_dir = run / "records" / logical_id
    text = (record_dir / "raw_asr.txt").read_text(encoding="utf-8")
    tts_dir = record_dir / "tts_raw"
    if not text.strip():
        write_json(
            root,
            tts_dir / "status.json",
            {
                "status": "NO_TEXT_FOR_TTS",
                "logical_id": logical_id,
                "reference_substitution_used": False,
            },
        )
        return
    if importlib.metadata.version("pyttsx3") != "2.99":
        raise ValueError("pyttsx3 version drift")
    engine = pyttsx3.init(driverName="sapi5")
    voices = engine.getProperty("voices")
    selected = select_required_voice(voices)
    engine.setProperty("voice", VOICE_ID)
    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)
    engine.startLoop(False)
    try:
        engine.iterate()
    finally:
        engine.endLoop()
    actual = {
        "voice_id": engine.getProperty("voice"),
        "rate": engine.getProperty("rate"),
        "volume": engine.getProperty("volume"),
    }
    if actual != {"voice_id": VOICE_ID, "rate": 175, "volume": 1.0}:
        raise ValueError(f"SAPI settings mismatch: {actual}")
    write_bytes(root, tts_dir / "synthesis_text.txt", text.encode("utf-8"))
    pending = output_path(root, tts_dir / "clear.pending.wav")
    final = output_path(root, tts_dir / "clear.wav")
    engine.save_to_file(text, str(pending))
    engine.runAndWait()
    engine.stop()
    meta = validate_generated_wav(pending)
    pending.rename(final)
    write_json(
        root,
        tts_dir / "synthesis.json",
        {
            "logical_id": logical_id,
            "backend": "pyttsx3",
            "package_version": "2.99",
            "driver": "sapi5",
            "selected_voice": voice_record(selected),
            "actual_settings": actual,
            "synthesis_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "wav_path": final.relative_to(root).as_posix(),
            "wav_sha256": sha256(final),
            "validation": meta,
            "listening_check": "NOT PERFORMED",
            "source": "RAW_ASR_TRANSCRIPT",
            "reference_used_for_synthesis": False,
        },
    )


def _run_tts(root: Path, run: Path, logical_id: str) -> None:
    command = [
        sys.executable,
        "-B",
        "-m",
        "vaaniswitch.m6_reconstruction.phase234",
        "tts-worker",
        "--root",
        str(root),
        "--run",
        str(run),
        "--record-id",
        logical_id,
    ]
    subprocess.run(
        command,
        cwd=root,
        check=True,
        timeout=60,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root / "src")},
    )


def _panel(root: Path, manifest: dict[str, dict[str, Any]]) -> list[str]:
    phase0 = _phase0(root)
    ids = [str(x) for x in phase0["proposed_panel"]["evaluation_ids"]]
    if len(ids) != 130 or len(set(ids)) != 130:
        raise ValueError("evaluation panel must contain 130 unique IDs")
    if any(x.split("/")[0] in {"M03", "MC02"} for x in ids):
        raise ValueError("development speaker leaked into evaluation panel")
    if (
        phase0["proposed_panel"]["ids_sha256"]
        != "15703d20490c83ac814664c48c3d8c8bf1197d2054d7d9d5ce495c68804a536b"
    ):
        raise ValueError("Phase-0 panel identity drift")
    if digest(ids) != phase0["proposed_panel"]["ids_sha256"]:
        raise ValueError("ordered evaluation panel identity drift")
    missing = [x for x in ids if x not in manifest]
    if missing:
        raise ValueError(f"panel IDs absent from manifest: {missing[:3]}")
    conditions: dict[str, int] = {}
    prompt_types: dict[str, int] = {}
    speakers = set()
    for logical_id in ids:
        row = manifest[logical_id]
        speakers.add(str(row["speaker_id"]))
        conditions[str(row["condition"])] = conditions.get(str(row["condition"]), 0) + 1
        prompt_types[str(row["prompt_type"])] = prompt_types.get(str(row["prompt_type"]), 0) + 1
    if len(speakers) != 13:
        raise ValueError(f"expected 13 evaluation speakers, got {len(speakers)}")
    if conditions != {"control": 60, "dysarthric": 70}:
        raise ValueError(f"condition counts drift: {conditions}")
    if prompt_types != {"sentence": 65, "word": 65}:
        raise ValueError(f"prompt-type counts drift: {prompt_types}")
    return ids


def _preservation(root: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-B", "scripts/module6_preservation.py", "--root", str(root)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root / "src")},
    )
    result: dict[str, Any] = json.loads(proc.stdout)
    if result["status"] != "PASS":
        raise RuntimeError(f"upstream preservation failed: {result}")
    if (
        result["frozen_scientific_code_sha256"]
        != "7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8"
    ):
        raise RuntimeError("frozen Modules 0-5 code identity drift")
    return result


def _code_identity(root: Path) -> dict[str, Any]:
    source = {}
    for path in sorted((root / "src/vaaniswitch/m6_reconstruction").glob("*.py")):
        source[path.relative_to(root).as_posix()] = sha256(path)
    return {"files": source, "sha256": digest(source)}


def _preflight(
    root: Path, manifest: dict[str, dict[str, Any]], inventory: dict[str, Any]
) -> dict[str, Any]:
    config = load_config(root / "configs/module6/pipeline.yaml", root)
    verify_checksum(config.checkpoint, MODEL_SHA)
    ids = _panel(root, manifest)
    phase0 = _phase0(root)
    expected_audio = phase0["panel_audio_verification"]["primary_audio_hashes"]
    for logical_id in ids:
        if not inventory[logical_id].eligible:
            raise ValueError(f"evaluation reference is not eligible: {logical_id}")
        _validate_audio(root, manifest[logical_id], expected_audio[logical_id])
    preservation = _preservation(root)
    import torch

    if not torch.version.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable at final-evaluation preflight")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4070 Laptop GPU":
        raise RuntimeError("unexpected CUDA device")
    tensor = torch.ones(4, device="cuda")
    torch.cuda.synchronize()
    if tensor.sum().item() != 4:
        raise RuntimeError("CUDA allocation check failed")
    del tensor
    return {
        "status": "PASS",
        "panel_records": len(ids),
        "panel_identity_from_phase0": phase0["proposed_panel"]["ids_sha256"],
        "evaluation_speakers": 13,
        "development_speakers_excluded": ["M03", "MC02"],
        "conditions": {"dysarthric": 70, "control": 60},
        "prompt_types": {"word": 65, "sentence": 65},
        "model_sha256": MODEL_SHA,
        "config_identity": config.identity,
        "normalizer_sha256": NORMALIZER_SHA256,
        "code": _code_identity(root),
        "preservation": preservation,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "evaluation_panel_forwards": 0,
    }


def _markdown_table_dev(run: Path, ids: list[str]) -> str:
    lines = ["| Record | Raw ASR | WER | CER |", "|---|---|---:|---:|"]
    metrics = json.loads((run / "metrics_raw.json").read_text(encoding="utf-8"))
    by_id = {row["metadata"]["logical_id"]: row["metric"] for row in metrics["records"]}
    for logical_id in ids:
        raw = (run / "records" / logical_id / "raw_asr.txt").read_text(encoding="utf-8")
        m = by_id[logical_id]
        lines.append(
            f"| `{logical_id}` | `{raw.replace('|', '¦')}` | {m['wer']:.6f} | {m['cer']:.6f} |"
        )
    return "\n".join(lines)


def pre_evaluation(root: Path) -> Path:
    """Recover existing development output and apply the current complete gate set."""
    from .final_pipeline import prepare

    return prepare(root)


def _verify_freeze(root: Path, freeze_path: Path, expected_sha: str) -> dict[str, Any]:
    if sha256(freeze_path) != expected_sha:
        raise ValueError("PRE_EVALUATION_FREEZE SHA mismatch")
    freeze: dict[str, Any] = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS" or freeze.get("final_evaluation_status") != "NOT STARTED":
        raise ValueError("freeze is not in approved pre-evaluation state")
    if freeze["model_sha256"] != MODEL_SHA or freeze["normalizer_sha256"] != NORMALIZER_SHA256:
        raise ValueError("frozen model/normalizer identity mismatch")
    if freeze["code_identity"] != _code_identity(root):
        raise ValueError("Module 6 code changed after pre-evaluation freeze")
    if freeze["environment_lock_sha256"] != sha256(
        root / "configs/module6/environment/requirements-win-py312.lock"
    ):
        raise ValueError("Module 6 environment lock changed")
    config = load_config(root / "configs/module6/pipeline.yaml", root)
    if freeze["config_identity"] != config.identity:
        raise ValueError("Module 6 config changed")
    verify_checksum(config.checkpoint, MODEL_SHA)
    preservation = _preservation(root)
    if preservation["status"] != "PASS":
        raise ValueError("upstream preservation failed")
    return freeze


def final_evaluation(root: Path, freeze_path: Path, expected_sha: str) -> Path:
    """All public final entrypoints use the immutable once-only runner."""
    from .final_pipeline import final_evaluation as run_final

    return run_final(root, freeze_path, expected_sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pre-eval", "final-eval", "tts-worker"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path)
    parser.add_argument("--record-id")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--freeze-sha256")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "pre-eval":
        pre_evaluation(root)
    elif args.action == "final-eval":
        if args.freeze is None or not args.freeze_sha256:
            parser.error("--freeze and --freeze-sha256 are required")
        freeze = args.freeze if args.freeze.is_absolute() else root / args.freeze
        final_evaluation(root, freeze, args.freeze_sha256)
    else:
        if args.run is None or not args.record_id:
            parser.error("--run and --record-id are required")
        run = args.run if args.run.is_absolute() else root / args.run
        _tts_worker(root, run, args.record_id)


if __name__ == "__main__":
    main()
