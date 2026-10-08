"""Independent contract and recovery-gate regression checks; no real model calls."""

from pathlib import Path
from typing import Any

import pytest

from vaaniswitch.m6_reconstruction import phase234, phase234_compat
from vaaniswitch.m6_reconstruction.artifacts import sha256
from vaaniswitch.m6_reconstruction.metrics import aggregate, score
from vaaniswitch.m6_reconstruction.normalization import normalize
from vaaniswitch.m6_reconstruction.references import derive_reference, read_reference
from vaaniswitch.m6_reconstruction.schemas import AudioJob

ROOT = Path(__file__).resolve().parents[2]


def test_blocked_freeze_cannot_authorize_evaluation(tmp_path: Path) -> None:
    freeze = tmp_path / "PRE_EVALUATION_FREEZE.json"
    freeze.write_text('{"status":"BLOCKED","final_evaluation_status":"NOT STARTED"}')
    with pytest.raises(ValueError, match="approved pre-evaluation state"):
        phase234._verify_freeze(ROOT, freeze, sha256(freeze))


def test_panel_order_is_hashed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = phase234._phase0(ROOT)
    original["proposed_panel"]["evaluation_ids"].reverse()
    monkeypatch.setattr(phase234, "_phase0", lambda root: original)
    with pytest.raises(ValueError, match="identity"):
        phase234._panel(ROOT, phase234_compat.load_manifest(ROOT))


def test_core_comes_from_signed_selection_not_recomputed_flags() -> None:
    manifest = phase234_compat.load_manifest(ROOT)
    manifest["M03/Session2/0002"]["normal_speech_eligible"] = False
    core = phase234_compat.signed_core_ids(ROOT, manifest)
    assert len(core) == 8673
    assert "M03/Session2/0002" in core


def test_recognition_job_rejects_reference_and_mismatched_identity() -> None:
    job = phase234_compat.audio_job(phase234_compat.load_manifest(ROOT)["M03/Session2/0002"])
    with pytest.raises(ValueError, match="audio bindings"):
        AudioJob.from_dict({**job, "reference": "now"})
    with pytest.raises(ValueError, match="composite identity"):
        AudioJob.from_dict({**job, "logical_id": "MC02/Session1/0002"})


def test_reference_join_rejects_prompt_only_collision() -> None:
    row = phase234_compat.load_manifest(ROOT)["M03/Session2/0002"]
    with pytest.raises(ValueError, match="composite identity"):
        read_reference(ROOT, {**row, "speaker_id": "MC02"}, core_member=True)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (None, "MISSING_REFERENCE"),
        ("  ", "BLANK_REFERENCE"),
        ("xxx", "PLACEHOLDER_REFERENCE"),
        ("scene.PNG", "IMAGE_STIMULUS_REFERENCE"),
        ("word [other note]", "UNSUPPORTED_BRACKET_PATTERN"),
    ],
)
def test_unavailable_reference_stays_null(text: str | None, reason: str) -> None:
    result = score(derive_reference(text, core_member=True), "a hypothesis")
    assert result["wer"] is result["cer"] is None
    assert result["exclusion_reason"] == reason


def test_normalization_has_no_number_expansion_or_repair() -> None:
    assert normalize("  \uff29\uff34\u2019s\u2014\uff19\uff13, years!\t") == "its 93 years"
    assert normalize("jarg it") == "jarg it"
    ref = derive_reference("ninety-three", core_member=True)
    assert score(ref, "93")["wer"]["ratio"] == 1.0


def test_asr_failure_is_not_empty_hypothesis() -> None:
    ref = derive_reference("a b", core_member=True)
    empty = score(ref, "")
    assert empty["wer"]["deletions"] == 2
    failure = score(ref, None, asr_status="ASR_FAILURE")
    assert failure["hypothesis_original"] is None
    assert failure["wer"] is None and failure["exclusion_reason"] == "ASR_FAILURE"


def test_count_weighted_corpus_and_equal_group_sample_sd() -> None:
    rows: list[dict[str, Any]] = []
    for group, ref, hyp in [("1", "a", "a"), ("2", "a b", ""), ("3", "a", "a b c")]:
        rows.append(
            {
                "logical_id": group,
                "speaker_id": group,
                "condition": "synthetic",
                "prompt_type": "word",
                "inherited_group": group,
                "metrics": score(derive_reference(ref, core_member=True), hyp),
            }
        )
    result = aggregate(rows, scope="SYNTHETIC")
    assert result["corpus"]["wer"]["ratio"] == 1.0  # 4 errors / 4 words
    assert result["three_group_mean_sample_sd"]["wer"] == {"mean": 1.0, "sample_sd": 1.0}
    assert rows[-1]["metrics"]["wer"]["percentage"] == 200.0
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(rows + rows[:1], scope="SYNTHETIC")
