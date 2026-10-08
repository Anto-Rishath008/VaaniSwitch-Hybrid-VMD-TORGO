"""Immutable final-ASR claims and terminal states; no references or metrics enter this module."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from .artifacts import (
    canonical_bytes,
    complete_json,
    digest,
    output_path,
    sha256,
    verify_checksum,
    write_bytes,
)
from .schemas import AudioJob


def read_json(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_bytes())
    return value


def persist(root: Path, path: Path, value: Any) -> None:
    """Idempotent publication of an identical artifact; never replace different bytes."""
    if path.exists():
        if path.read_bytes() != canonical_bytes(value) + b"\n":
            raise ValueError(f"immutable artifact mismatch: {path}")
    else:
        complete_json(root, path, value)


@contextmanager
def exclusive_run(root: Path) -> Iterator[None]:
    """OS-owned lock releases on process death; no unsafe stale-PID lock deletion."""
    path = output_path(root, "artifacts/module6/final_evaluation/execution.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream: BinaryIO = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another final execution owns the lock") from exc
        else:
            raise RuntimeError("final execution requires the validated Windows locking backend")
        yield
    finally:
        stream.close()


def claim_run(root: Path, identity: dict[str, Any], jobs: list[AudioJob]) -> Path:
    if [j.logical_id for j in jobs] != identity["panel_ids"]:
        raise ValueError("ordered jobs differ from final run identity")
    if len({j.logical_id for j in jobs}) != len(jobs):
        raise ValueError("duplicate final job")
    key = digest(identity)
    directory = output_path(root, "artifacts/module6/final_evaluation")
    claim = directory / "CLAIM.json"
    run = directory / key
    value = {
        "identity": identity,
        "identity_sha256": key,
        "jobs_sha256": digest([j.to_dict() for j in jobs]),
        "run": run.relative_to(root).as_posix(),
    }
    if not claim.exists() and (
        list(directory.glob("*/FINAL_EVALUATION_COMPLETE.json"))
        or any(p.is_dir() for p in directory.iterdir())
    ):
        raise ValueError("unclaimed existing final artifacts prohibit a new run")
    persist(root, claim, value)
    run.mkdir(parents=True, exist_ok=True)
    persist(root, run / "identity.json", value)
    persist(root, run / "recognition_jobs.json", [j.to_dict() for j in jobs])
    return run


def terminal(root: Path, run: Path, key: str, job: AudioJob) -> dict[str, Any] | None:
    directory = run / "records" / job.logical_id
    seal, failure = directory / "raw_asr.json", directory / "ASR_FAILURE.json"
    attempt = directory / "ATTEMPT.json"
    if not attempt.exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("record has artifacts without an immutable attempt claim")
        return None
    claim = read_json(attempt)
    if claim["identity_sha256"] != key or claim["audio_job"] != job.to_dict():
        raise ValueError("attempt identity mismatch")
    if seal.exists() and failure.exists():
        raise ValueError("conflicting success/failure terminal states")
    if seal.exists():
        record = read_json(seal)
        if (
            record["identity_sha256"] != key
            or record["audio_job"] != job.to_dict()
            or record["status"] != "COMPLETE"
            or record["reference_context_used"] is not False
        ):
            raise ValueError("retained raw identity mismatch")
        verify_checksum(directory / "raw_asr.txt", record["raw_text_sha256"])
        verify_checksum(directory / "backend_result.json", record["backend_result_sha256"])
        verify_checksum(directory / "source_validation.json", record["source_validation_sha256"])
        source = read_json(directory / "source_validation.json")
        identity = read_json(run / "identity.json")["identity"]
        if (
            record["device"] != identity["device"]
            or record["dtype"] != identity["dtype"]
            or source["logical_id"] != job.logical_id
            or source["path"] != job.primary_audio_path
            or source["complete_file_sha256"] != job.primary_audio_sha256
            or record["input_float32_sha256"] != source["decoded_float32_sha256"]
            or ("decoding" in identity and record.get("decoding") != identity["decoding"])
        ):
            raise ValueError("retained scientific metadata mismatch")
        raw = (directory / "raw_asr.txt").read_bytes()
        if raw != record["raw_text"].encode("utf-8") or raw != read_json(
            directory / "backend_result.json"
        )["text"].encode("utf-8"):
            raise ValueError("retained backend/raw text mismatch")
        if record["input_frames"] != job.frame_count:
            raise ValueError("retained complete frame count mismatch")
        if not math.isfinite(record["elapsed_seconds"]) or record["elapsed_seconds"] < 0:
            raise ValueError("invalid recorded ASR runtime")
        return record
    if failure.exists():
        record = read_json(failure)
        if (
            record["identity_sha256"] != key
            or record["audio_job"] != job.to_dict()
            or record["status"] != "ASR_FAILURE"
        ):
            raise ValueError("failure identity mismatch")
        return record
    return None


def _failure(
    root: Path, directory: Path, key: str, job: AudioJob, error_type: str, error: str
) -> dict[str, Any]:
    record = {
        "status": "ASR_FAILURE",
        "identity_sha256": key,
        "audio_job": job.to_dict(),
        "logical_id": job.logical_id,
        "error_type": error_type,
        "error": error,
        "raw_text": None,
        "retry_allowed": False,
    }
    complete_json(root, directory / "ASR_FAILURE.json", record)
    return record


def recognize(
    root: Path,
    run: Path,
    identity: dict[str, Any],
    jobs: list[AudioJob],
    infer: Callable[[AudioJob], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Caller holds exclusive_run. Resume executes only never-claimed records.

    A crashed in-flight claim becomes an explicit failure, not another model forward.
    All successful seals are verified before ANY new work is allowed.
    """
    key = digest(identity)
    if read_json(run / "identity.json")["identity"] != identity:
        raise ValueError("run identity mismatch")
    if (
        read_json(run / "identity.json")["jobs_sha256"] != digest([j.to_dict() for j in jobs])
        or [j.logical_id for j in jobs] != identity["panel_ids"]
    ):
        raise ValueError("frozen job list mismatch")
    retained = [terminal(root, run, key, job) for job in jobs]
    results = []
    for index, (job, previous) in enumerate(zip(jobs, retained, strict=True)):
        directory = run / "records" / job.logical_id
        if previous is not None:
            results.append(previous)
            continue
        if (directory / "ATTEMPT.json").exists():
            results.append(
                _failure(
                    root,
                    directory,
                    key,
                    job,
                    "INTERRUPTED_ATTEMPT",
                    "Prior attempt has no validated successful seal; not rerun",
                )
            )
            continue
        complete_json(
            root,
            directory / "ATTEMPT.json",
            {
                "identity_sha256": key,
                "audio_job": job.to_dict(),
                "claimed_utc": datetime.now(UTC).isoformat(),
                "attempt_number": 1,
            },
        )
        try:
            result = infer(job)
        except Exception as exc:
            results.append(_failure(root, directory, key, job, type(exc).__name__, str(exc)))
            continue
        # Publication errors propagate and cannot trigger another inference on resume.
        if (
            not isinstance(result.get("raw_text"), str)
            or result["backend_result"].get("text") != result["raw_text"]
            or result["reference_context_used"] is not False
            or result["input_frames"] != job.frame_count
            or result["device"] != identity["device"]
            or result["dtype"] != identity["dtype"]
        ):
            results.append(
                _failure(
                    root,
                    directory,
                    key,
                    job,
                    "INVALID_BACKEND_RESULT",
                    "Backend violated frozen audio/text/device contract",
                )
            )
            continue
        raw = write_bytes(root, directory / "raw_asr.txt", result["raw_text"].encode("utf-8"))
        persist(root, directory / "backend_result.json", result["backend_result"])
        persist(root, directory / "source_validation.json", result["source_validation"])
        record = {
            k: v for k, v in result.items() if k not in ("backend_result", "source_validation")
        }
        record.update(
            status="COMPLETE",
            logical_id=job.logical_id,
            identity_sha256=key,
            audio_job=job.to_dict(),
            raw_text_sha256=sha256(raw),
            backend_result_sha256=sha256(directory / "backend_result.json"),
            source_validation_sha256=sha256(directory / "source_validation.json"),
        )
        complete_json(root, directory / "raw_asr.json", record)
        checked = terminal(root, run, key, job)
        assert checked is not None
        results.append(checked)
        print(
            json.dumps(
                {
                    "terminal_records": index + 1,
                    "requested": len(jobs),
                    "latest_id": job.logical_id,
                    "status": "COMPLETE",
                }
            ),
            flush=True,
        )
    seals = {
        j.logical_id: sha256(
            run
            / "records"
            / j.logical_id
            / ("raw_asr.json" if r["status"] == "COMPLETE" else "ASR_FAILURE.json")
        )
        for j, r in zip(jobs, results, strict=True)
    }
    persist(
        root,
        run / "RECOGNITION_TERMINAL.json",
        {
            "identity_sha256": key,
            "requested": len(jobs),
            "terminal_records": len(results),
            "successful": sum(r["status"] == "COMPLETE" for r in results),
            "failed": sum(r["status"] == "ASR_FAILURE" for r in results),
            "seals": seals,
        },
    )
    return results


def require_terminal(
    root: Path, run: Path, identity: dict[str, Any], jobs: list[AudioJob]
) -> list[dict[str, Any]]:
    marker = read_json(run / "RECOGNITION_TERMINAL.json")
    key = digest(identity)
    if marker["identity_sha256"] != key or marker["terminal_records"] != len(jobs):
        raise ValueError("full panel has not reached terminal states")
    results = []
    for job in jobs:
        record = terminal(root, run, key, job)
        if record is None:
            raise ValueError("nonterminal record prohibits metric aggregation")
        filename = "raw_asr.json" if record["status"] == "COMPLETE" else "ASR_FAILURE.json"
        verify_checksum(
            run / "records" / job.logical_id / filename, marker["seals"][job.logical_id]
        )
        results.append(record)
    return results
