"""Mocked interruption/resume and once-only state tests. No final source/model forwards."""

import json
from pathlib import Path
from typing import Any

import pytest

from vaaniswitch.m6_reconstruction import final_state as state
from vaaniswitch.m6_reconstruction.artifacts import digest
from vaaniswitch.m6_reconstruction.metrics import aggregate, score
from vaaniswitch.m6_reconstruction.references import derive_reference
from vaaniswitch.m6_reconstruction.schemas import AudioJob


def setup() -> tuple[list[AudioJob], dict[str, Any]]:
    jobs = [
        AudioJob(
            f"M01/Session1/{i:04d}",
            f"data/raw/torgo/extracted/M/M01/Session1/wav_headMic/{i:04d}.wav",
            "0" * 64,
            12,
        )
        for i in range(1, 4)
    ]
    identity = {
        "panel_ids": [j.logical_id for j in jobs],
        "device": "cuda",
        "dtype": "float16",
        "freeze_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "code_identity": "3" * 64,
    }
    return jobs, identity


def result(job: AudioJob) -> dict[str, Any]:
    return {
        "raw_text": " obviously poor text ",
        "backend_result": {"text": " obviously poor text "},
        "reference_context_used": False,
        "input_frames": job.frame_count,
        "input_float32_sha256": "4" * 64,
        "elapsed_seconds": 0.25,
        "device": "cuda",
        "dtype": "float16",
        "source_validation": {
            "logical_id": job.logical_id,
            "path": job.primary_audio_path,
            "complete_file_sha256": job.primary_audio_sha256,
            "decoded_float32_sha256": "4" * 64,
        },
    }


def test_resume_retains_success_marks_interruption_and_only_runs_unattempted(
    tmp_path: Path,
) -> None:
    jobs, identity = setup()
    called = []

    def interrupt(job: AudioJob) -> dict[str, Any]:
        called.append(job.logical_id)
        if job == jobs[1]:
            raise KeyboardInterrupt
        return result(job)

    with state.exclusive_run(tmp_path):
        run = state.claim_run(tmp_path, identity, jobs)
        with pytest.raises(KeyboardInterrupt):
            state.recognize(tmp_path, run, identity, jobs, interrupt)
        assert not (run / "RECOGNITION_TERMINAL.json").exists()
        before = (run / "records" / jobs[0].logical_id / "raw_asr.json").read_bytes()

    def resumed(job: AudioJob) -> dict[str, Any]:
        called.append(job.logical_id)
        return result(job)

    with state.exclusive_run(tmp_path):
        assert state.claim_run(tmp_path, identity, jobs) == run
        states = state.recognize(tmp_path, run, identity, jobs, resumed)
        assert [r["status"] for r in states] == ["COMPLETE", "ASR_FAILURE", "COMPLETE"]
        assert called == [j.logical_id for j in jobs]
        assert (run / "records" / jobs[0].logical_id / "raw_asr.json").read_bytes() == before
        assert len(state.require_terminal(tmp_path, run, identity, jobs)) == 3
        state.recognize(tmp_path, run, identity, jobs, lambda job: pytest.fail("silent retry"))


def test_record_tampering_fails_before_new_work(tmp_path: Path) -> None:
    jobs, identity = setup()

    def interrupted(job: AudioJob) -> dict[str, Any]:
        if job == jobs[1]:
            raise KeyboardInterrupt
        return result(job)

    with state.exclusive_run(tmp_path):
        run = state.claim_run(tmp_path, identity, jobs)
        with pytest.raises(KeyboardInterrupt):
            state.recognize(tmp_path, run, identity, jobs, interrupted)
        (run / "records" / jobs[0].logical_id / "raw_asr.txt").write_text("changed")
        with pytest.raises(ValueError, match="SHA"):
            state.recognize(tmp_path, run, identity, jobs, lambda job: pytest.fail("forward"))


def test_identity_change_and_second_panel_rejected(tmp_path: Path) -> None:
    jobs, identity = setup()
    with state.exclusive_run(tmp_path):
        state.claim_run(tmp_path, identity, jobs)
        with pytest.raises(ValueError, match="immutable"):
            state.claim_run(tmp_path, {**identity, "freeze_sha256": "9" * 64}, jobs)
        with pytest.raises(ValueError, match="ordered"):
            state.claim_run(tmp_path, identity, list(reversed(jobs)))


def test_concurrent_execution_rejected(tmp_path: Path) -> None:
    with (
        state.exclusive_run(tmp_path),
        pytest.raises((RuntimeError, BlockingIOError)),
        state.exclusive_run(tmp_path),
    ):
        pytest.fail("second lock acquired")


def test_failed_runtime_stays_failure_and_aggregate_partial(tmp_path: Path) -> None:
    jobs, identity = setup()

    def failed(job: AudioJob) -> dict[str, Any]:
        raise RuntimeError("synthetic backend failure")

    with state.exclusive_run(tmp_path):
        run = state.claim_run(tmp_path, identity, jobs)
        with pytest.raises(FileNotFoundError):
            state.require_terminal(tmp_path, run, identity, jobs)
        states = state.recognize(tmp_path, run, identity, jobs, failed)
        state.recognize(tmp_path, run, identity, jobs, lambda job: pytest.fail("failure retried"))
    rows = [
        {
            "logical_id": r["logical_id"],
            "speaker_id": "M01",
            "condition": "synthetic",
            "prompt_type": "word",
            "inherited_group": "2",
            "metrics": score(
                derive_reference("a", core_member=True), r["raw_text"], asr_status=r["status"]
            ),
        }
        for r in states
    ]
    assert aggregate(rows, scope="SYNTHETIC")["completion_status"] == "PARTIAL"
    assert all(r["raw_text"] is None for r in states)


def test_completed_artifacts_never_overwritten(tmp_path: Path) -> None:
    jobs, identity = setup()
    with state.exclusive_run(tmp_path):
        run = state.claim_run(tmp_path, identity, jobs)
        state.recognize(tmp_path, run, identity, jobs, result)
        state.persist(tmp_path, run / "FINAL_EVALUATION_COMPLETE.json", {"status": "PASS"})
        with pytest.raises(ValueError, match="immutable"):
            state.persist(tmp_path, run / "FINAL_EVALUATION_COMPLETE.json", {"status": "different"})
        with pytest.raises(ValueError, match="identity"):
            state.recognize(tmp_path, run, {**identity, "dtype": "float32"}, jobs, result)
        assert state.read_json(run / "RECOGNITION_TERMINAL.json")["identity_sha256"] == digest(
            identity
        )


def test_changed_job_list_rejected_before_any_forward(tmp_path: Path) -> None:
    jobs, identity = setup()
    with state.exclusive_run(tmp_path):
        run = state.claim_run(tmp_path, identity, jobs)
        with pytest.raises(ValueError, match="job"):
            state.recognize(
                tmp_path,
                run,
                identity,
                jobs[:1],
                lambda job: pytest.fail("inference before job binding check"),
            )


@pytest.mark.parametrize("field", ["dtype", "input_float32_sha256"])
def test_retained_scientific_metadata_is_checked(tmp_path: Path, field: str) -> None:
    jobs, identity = setup()
    with state.exclusive_run(tmp_path):
        run = state.claim_run(tmp_path, identity, jobs)
        state.recognize(tmp_path, run, identity, jobs, result)
        path = run / "records" / jobs[0].logical_id / "raw_asr.json"
        modified = state.read_json(path)
        modified[field] = "wrong"
        path.write_text(json.dumps(modified))
        with pytest.raises(ValueError, match="scientific"):
            state.terminal(tmp_path, run, digest(identity), jobs[0])
