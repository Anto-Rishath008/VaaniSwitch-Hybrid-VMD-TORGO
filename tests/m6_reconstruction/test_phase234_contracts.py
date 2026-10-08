from __future__ import annotations

from vaaniswitch.m6_reconstruction.normalization import NORMALIZATION_SHA, normalize
from vaaniswitch.m6_reconstruction.phase234_compat import (
    inherited_group,
    sample_mean_sd,
    score,
)
from vaaniswitch.m6_reconstruction.references import derive_reference
from vaaniswitch.m6_reconstruction.schemas import PANEL_SHA, split_id


def test_frozen_panel_and_session23() -> None:
    assert PANEL_SHA == "15703d20490c83ac814664c48c3d8c8bf1197d2054d7d9d5ce495c68804a536b"
    assert split_id("M01/Session2_3/0001") == ("M01", "Session2_3", "0001")
    assert inherited_group("F03") == "1"
    assert inherited_group("MC02") == "2"
    assert inherited_group("M03") == "3"


def test_normalization_contract() -> None:
    assert normalize("  It\u2019s,  93-years!  ") == "its 93 years"
    assert normalize("DON'T") == "dont"
    assert len(NORMALIZATION_SHA) == 64


def test_reference_contract() -> None:
    sense = derive_reference("tear [as in tear up that paper]", core_member=True)
    assert sense["eligible"] and sense["scoring_reference"] == "tear"
    image = derive_reference("input/images/kitchen.jpg", core_member=True)
    assert not image["eligible"] and image["exclusion_reason"] == "IMAGE_STIMULUS_REFERENCE"
    blank = derive_reference("   ", core_member=True)
    assert not blank["eligible"] and blank["exclusion_reason"] == "BLANK_REFERENCE"
    bad = derive_reference("word [other note]", core_member=True)
    assert not bad["eligible"] and bad["eligibility_state"] == "REVIEW_REQUIRED"


def test_metric_identity_insertions_empty_and_unavailable() -> None:
    same = score("x", "hello world", "hello world").to_dict()
    assert same["wer"] == 0.0 and same["cer"] == 0.0
    insertion = score("x", "a", "a b c").to_dict()
    assert insertion["wer"] == 2.0
    empty = score("x", "a b", "").to_dict()
    assert empty["wer"] == 1.0
    unavailable = score("x", None, "hyp", exclusion_reason="IMAGE_STIMULUS_REFERENCE").to_dict()
    assert unavailable["wer"] is None and unavailable["cer"] is None
    assert not unavailable["eligible"]


def test_sample_sd() -> None:
    result = sample_mean_sd([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["sample_sd"] == 1.0
