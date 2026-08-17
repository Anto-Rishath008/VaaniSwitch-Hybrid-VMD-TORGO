"""Metadata, transcript, and prompt-category regression tests."""

from pathlib import Path

import pytest

from tests.data.helpers import make_config
from components.data.metadata import (
    MetadataError,
    parse_prompt_id,
    parse_session_id,
    parse_speaker,
    read_transcript,
)
from components.data.prompt_types import classify_prompt
from components.data.schemas import Condition, Gender, PromptType


def test_speaker_group_parsing_assigns_condition_and_gender(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert parse_speaker("F", "F01", config).condition is Condition.DYSARTHRIC
    assert parse_speaker("M", "M05", config).gender is Gender.MALE
    assert parse_speaker("FC", "FC03", config).condition is Condition.CONTROL
    assert parse_speaker("MC", "MC04", config).gender is Gender.MALE


def test_speaker_must_match_exact_group(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(MetadataError, match="does not match"):
        parse_speaker("F", "FC01", config)


def test_session_parsing_preserves_combined_session_id(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert parse_session_id("Session2_3", config) == "Session2_3"
    with pytest.raises(MetadataError, match="invalid TORGO session"):
        parse_session_id("Session_2", config)


def test_prompt_id_parsing_uses_real_four_digit_stem(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert parse_prompt_id(Path("0007.txt"), config) == "0007"
    with pytest.raises(MetadataError, match="invalid TORGO prompt"):
        parse_prompt_id(Path("7.txt"), config)


def test_transcript_is_read_from_prompt_file(tmp_path: Path) -> None:
    prompt = tmp_path / "0042.txt"
    prompt.write_text("  A real transcript.  \n", encoding="utf-8")

    result = read_transcript(prompt, ("utf-8-sig",))

    assert result.transcript == "A real transcript."
    assert result.error is None


def test_missing_and_malformed_prompt_handling(tmp_path: Path) -> None:
    missing = read_transcript(tmp_path / "missing.txt", ("utf-8",))
    malformed_path = tmp_path / "bad.txt"
    malformed_path.write_bytes(b"\xff\xfe")
    malformed = read_transcript(malformed_path, ("utf-8",))

    assert missing.transcript is None
    assert missing.error is not None and missing.error.startswith("prompt_read_error")
    assert malformed.transcript is None
    assert malformed.error is not None and malformed.error.startswith("prompt_decode_error")


def test_blank_prompt_is_not_treated_as_a_transcript(tmp_path: Path) -> None:
    prompt = tmp_path / "blank.txt"
    prompt.write_text("  \r\n\t", encoding="utf-8")

    result = read_transcript(prompt, ("utf-8",))

    assert result.transcript is None
    assert result.error == "blank_prompt_text"


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("fork", PromptType.WORD),
        ("tear [as in tear up that paper]", PromptType.WORD),
        ("Nothing has been done yet.", PromptType.SENTENCE),
        ("[say Pah-Tah-Kah repeatedly]", PromptType.ARTICULATION_TASK),
        (
            "[relax your mouth in its normal position]",
            PromptType.INSTRUCTION_OR_NONSTANDARD,
        ),
        ("xxx", PromptType.UNKNOWN),
    ],
)
def test_prompt_type_classification(transcript: str, expected: PromptType) -> None:
    assert classify_prompt(transcript) is expected


def test_terminal_escape_prompt_is_flagged_unknown(tmp_path: Path) -> None:
    prompt = tmp_path / "0009.txt"
    prompt.write_text("rare lie\x1b[D", encoding="utf-8")

    result = read_transcript(prompt, ("utf-8",))

    assert "malformed_transcript_control_characters" in result.flags
    assert classify_prompt(result.transcript, result.flags) is PromptType.UNKNOWN
