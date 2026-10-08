"""m6_text_v1: explicit symmetric normalization, without lexical substitutions."""

from __future__ import annotations

import unicodedata

from .artifacts import digest

NORMALIZATION = {
    "id": "m6_text_v1",
    "unicode": "NFKC",
    "case": "lower",
    "apostrophes": "U+2018/U+2019 -> U+0027; then remove U+0027",
    "punctuation": "replace remaining Unicode category P characters with spaces",
    "whitespace": "collapse and trim",
    "digits": "unchanged",
    "cer": "remove normalized spaces before character alignment",
    "lexical_substitutions": False,
}
NORMALIZATION_SHA = digest(NORMALIZATION)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = value.replace("\u2018", "'").replace("\u2019", "'").replace("'", "")
    value = "".join(" " if unicodedata.category(c).startswith("P") else c for c in value)
    return " ".join(value.split())
