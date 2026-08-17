"""Conservative, transparent TORGO prompt-type classification."""

from __future__ import annotations

import re

from components.data.schemas import PromptType

_FULL_BRACKETED = re.compile(r"^\s*\[(?P<body>.*)]\s*$", re.DOTALL)
_WORD_WITH_SENSE_CLARIFIER = re.compile(
    r"^\s*[^\s\[]+\s+\[as\s+in\s+.+]\s*$",
    re.IGNORECASE | re.DOTALL,
)


def classify_prompt(transcript: str | None, transcript_flags: tuple[str, ...] = ()) -> PromptType:
    """Classify only patterns supported by the prompt text itself.

    Fully bracketed ``say`` tasks are articulation tasks; non-vocal bracketed
    directions are instructions. TORGO sense clarifiers such as
    ``tear [as in tear up that paper]`` still identify a single target word.
    Placeholder or malformed text remains unknown.
    """

    if transcript is None or not transcript.strip() or transcript_flags:
        return PromptType.UNKNOWN
    normalized = transcript.strip()
    if normalized.casefold() == "xxx":
        return PromptType.UNKNOWN

    bracketed = _FULL_BRACKETED.fullmatch(normalized)
    if bracketed is not None:
        body = bracketed.group("body").strip().casefold()
        if body.startswith("say "):
            return PromptType.ARTICULATION_TASK
        return PromptType.INSTRUCTION_OR_NONSTANDARD

    if _WORD_WITH_SENSE_CLARIFIER.fullmatch(normalized) is not None:
        return PromptType.WORD
    if len(normalized.split()) == 1:
        return PromptType.WORD
    return PromptType.SENTENCE
