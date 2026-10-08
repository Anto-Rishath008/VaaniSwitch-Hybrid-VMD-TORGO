"""Explicit Phase-1 acquisition and bounded mechanical smoke commands only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import (
    complete_json,
    digest,
    output_path,
    sha256,
    verify_checksum,
    write_bytes,
    write_json,
)
from .audio import decode_pcm
from .config import MODEL_SHA, MODEL_SOURCE, RECORD_ID, RuntimeConfig, load_config
from .provenance import environment_record, resolve_smoke_audio


def bounded_worker(root: Path, command: list[str], log: Path, timeout: int) -> None:
    logfile = output_path(root, log)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("xb") as stream:
        subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
            cwd=root,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root / "src")},
        )


def asr_worker(config: RuntimeConfig, run: Path) -> None:
    from .asr import load_frozen_model, transcribe_audio

    job = json.loads((run / "recognition_audio.json").read_bytes())
    if set(job) != {"logical_id", "primary_audio_path", "primary_audio_sha256", "frame_count"}:
        raise ValueError("recognition job must contain audio bindings only")
    if job["logical_id"] != RECORD_ID:
        raise ValueError("Phase 1 permits only the predetermined development record")
    payload = (config.root / job["primary_audio_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != job["primary_audio_sha256"]:
        raise ValueError("source audio changed before decoding")
    audio, metadata = decode_pcm(payload, source=True)
    if len(audio) != job["frame_count"]:
        raise ValueError("complete source frame count mismatch")
    model, probe = load_frozen_model(config)
    write_json(config.root, run / "cuda_model_probe.json", probe)
    result = transcribe_audio(model, audio, device=config.device)
    # Persist exact backend text before any possible later processing.
    text_path = write_bytes(config.root, run / "raw_asr.txt", result["raw_text"].encode("utf-8"))
    backend_path = write_json(config.root, run / "backend_result.json", result["backend_result"])
    result.update(
        record_id=RECORD_ID,
        source=job,
        audio=metadata,
        raw_text_sha256=sha256(text_path),
        backend_json_sha256=sha256(backend_path),
        model_probe=probe,
        checkpoint_sha256=MODEL_SHA,
        model_source=MODEL_SOURCE,
        checkpoint_path=str(config.checkpoint),
        whisper_package_version="20250625",
    )
    write_json(config.root, run / "asr_smoke.json", result)


def retained_asr(config: RuntimeConfig, previous: Path) -> dict[str, bytes]:
    """Validate an interrupted run before reusing its exact ASR bytes; no inference."""
    previous = output_path(config.root, previous)
    if not (previous / "FAILURE.json").is_file() or (previous / "RUNTIME_COMPLETE.json").exists():
        raise ValueError("TTS resume requires a retained failed runtime attempt")
    result = json.loads((previous / "asr_smoke.json").read_bytes())
    backend = json.loads((previous / "backend_result.json").read_bytes())
    job = resolve_smoke_audio(config.root)
    if result["source"] != job or result["device"] != config.device:
        raise ValueError("retained ASR source/device mismatch")
    if json.loads((previous / "config.resolved.json").read_bytes()) != config.values:
        raise ValueError("retained configuration mismatch")
    if result["checkpoint_sha256"] != MODEL_SHA:
        raise ValueError("retained ASR model mismatch")
    verify_checksum(config.checkpoint, MODEL_SHA)
    verify_checksum(previous / "raw_asr.txt", result["raw_text_sha256"])
    verify_checksum(previous / "backend_result.json", result["backend_json_sha256"])
    if (previous / "raw_asr.txt").read_bytes() != result["raw_text"].encode("utf-8"):
        raise ValueError("retained raw text mismatch")
    if backend != result["backend_result"] or backend["text"] != result["raw_text"]:
        raise ValueError("retained backend mismatch")
    old = json.loads((previous / "reproducibility.json").read_bytes())
    for path, expected in old["source_sha256"].items():
        if Path(path).name not in {"tts.py", "runtime.py"}:
            verify_checksum(config.root / path, expected)
    lock = config.root / "configs/module6/environment/requirements-win-py312.lock"
    verify_checksum(lock, old["identity"]["lock"])
    names = ["raw_asr.txt", "backend_result.json", "asr_smoke.json", "cuda_model_probe.json"]
    return {name: (previous / name).read_bytes() for name in names}


def smoke(config: RuntimeConfig, config_path: Path, previous: Path | None = None) -> Path:
    root = config.root
    environment = environment_record(root)
    job = resolve_smoke_audio(root)
    reused = retained_asr(config, previous) if previous else None
    run_id = "phase1_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run = output_path(root, f"artifacts/module6/{config.identity}/{run_id}")
    run.mkdir(parents=True, exist_ok=False)
    source_files = sorted((root / "src/vaaniswitch/m6_reconstruction").glob("*.py"))
    code = {p.relative_to(root).as_posix(): sha256(p) for p in source_files}
    identity = {
        "config": config.identity,
        "code": digest(code),
        "lock": environment["lock_sha256"],
        "environment": environment["identity_sha256"],
        "source_audio": job["primary_audio_sha256"],
        "model": MODEL_SHA,
        "device": config.device,
        "dtype": config.values["execution"]["dtype"],
        "phase": 1,
        "reference_context": False,
        "metrics": False,
        "correction": False,
    }
    write_json(root, run / "config.resolved.json", config.values)
    write_json(root, run / "environment.json", environment)
    write_json(
        root,
        run / "reproducibility.json",
        {
            "identity": identity,
            "identity_sha256": digest(identity),
            "source_sha256": code,
            "seed": 42,
            "cross_device_bitwise_reproducibility": "NOT CLAIMED",
            "command": sys.argv,
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    write_json(root, run / "recognition_audio.json", job)
    if reused is not None and previous is not None:
        for name, payload in reused.items():
            write_bytes(root, run / name, payload)
        write_json(
            root,
            run / "asr_reuse.json",
            {
                "parent_run": str(previous),
                "parent_asr_sha256": sha256(previous / "asr_smoke.json"),
                "parent_environment_sha256": sha256(previous / "environment.json"),
                "parent_reproducibility_sha256": sha256(previous / "reproducibility.json"),
                "files_sha256": {name: sha256(previous / name) for name in reused},
                "reason": "TTS-only repair; successful CUDA ASR retained without another forward",
            },
        )
    command = [
        sys.executable,
        "-B",
        "-m",
        "vaaniswitch.m6_reconstruction.runtime",
        "--root",
        str(root),
        "--config",
        str(config_path),
        "--run",
        str(run),
    ]
    started = time.perf_counter()
    try:
        if reused is None:
            bounded_worker(
                root,
                [*command, "asr-worker"],
                run / "asr_worker.log",
                config.values["execution"]["asr_timeout_seconds"],
            )
        bounded_worker(
            root,
            [*command, "tts-worker"],
            run / "tts_worker.log",
            config.values["tts"]["timeout_seconds"],
        )
        hashes = {p.relative_to(run).as_posix(): sha256(p) for p in run.rglob("*") if p.is_file()}
        complete_json(
            root,
            run / "RUNTIME_COMPLETE.json",
            {
                "status": "PASS" if config.device == "cuda" else "CPU-FALLBACK",
                "scope": "mechanical Phase-1 runtime smoke only; not Module 6 sign-off",
                "identity": identity,
                "artifact_sha256": hashes,
                "elapsed_seconds": time.perf_counter() - started,
                "asr_records": [RECORD_ID],
                "asr_reused": reused is not None,
                "new_asr_transcriptions": 0 if reused is not None else 1,
                "evaluation_records": 0,
                "wer_cer_computed": False,
                "module5_executed": False,
                "module7_started": False,
            },
        )
    except Exception as exc:
        write_json(
            root,
            run / "FAILURE.json",
            {
                "error": repr(exc),
                "elapsed_seconds": time.perf_counter() - started,
                "device": config.device,
            },
        )
        raise
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["acquire", "smoke", "resume-tts", "asr-worker", "tts-worker"]
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/module6/pipeline.yaml"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(path, root)
    if Path(sys.prefix).resolve() != root / ".venv-m6":
        raise RuntimeError("use .venv-m6/Scripts/python.exe")
    if args.action == "acquire":
        from .asr import acquire_model

        record = acquire_model(config)
        write_json(root, "artifacts/module6/models/whisper/acquisition.json", record)
        print(json.dumps(record))
    elif args.action == "smoke":
        print(smoke(config, path))
    elif args.action == "resume-tts":
        if args.run is None:
            parser.error("--run must identify the retained failed attempt")
        print(smoke(config, path, output_path(root, args.run)))
    else:
        if args.run is None:
            parser.error("--run is required for internal workers")
        run = output_path(root, args.run)
        if args.action == "asr-worker":
            asr_worker(config, run)
        else:
            from .tts import synthesize_smoke

            synthesize_smoke(config, run)


if __name__ == "__main__":
    main()
