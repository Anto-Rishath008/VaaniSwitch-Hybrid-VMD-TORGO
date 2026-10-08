"""Public entrypoints and evaluator/preflight must fail closed before inference."""

import json
from pathlib import Path
from typing import Any

import pytest

from vaaniswitch.m6_reconstruction import final_pipeline, phase234, phase234_compat
from vaaniswitch.m6_reconstruction.artifacts import sha256


def test_blocked_freeze_refused_before_runtime(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    path.write_text('{"status":"BLOCKED"}')
    with pytest.raises(ValueError, match="gates"):
        final_pipeline.final_evaluation(tmp_path, path, sha256(path))


def test_missing_quality_gates_prevent_prepare(tmp_path: Path) -> None:
    path = tmp_path / final_pipeline.BASE / "QUALITY_GATES.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"checks": {"pytest": "PASS"}}))
    with pytest.raises(ValueError, match="quality"):
        final_pipeline.prepare(tmp_path)


def test_scoring_waits_for_every_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        phase234_compat,
        "load_manifest",
        lambda root: pytest.fail("evaluator accessed before terminal gate"),
    )
    with pytest.raises(FileNotFoundError):
        final_pipeline._evaluate(tmp_path, tmp_path, {}, [], {})


def test_old_public_entries_use_new_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def prepared(root: Path) -> Path:
        calls.append("prepare")
        return root

    def executed(root: Path, freeze_path: Path, expected_sha: str) -> Path:
        calls.append("execute")
        return root

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("old development runtime repeated")

    monkeypatch.setattr(final_pipeline, "prepare", prepared)
    monkeypatch.setattr(final_pipeline, "final_evaluation", executed)
    monkeypatch.setattr(phase234, "_run_asr_for_ids", forbidden)
    assert phase234.pre_evaluation(tmp_path) == tmp_path
    assert phase234.final_evaluation(tmp_path, tmp_path / "freeze", "x") == tmp_path
    assert calls == ["prepare", "execute"]
