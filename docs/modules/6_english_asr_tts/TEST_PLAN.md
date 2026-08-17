# Test Plan

Required:
- frozen ASR
- no gradient link
- WER/CER
- transcript immutability
- TTS WAV
- no voice clone

Tools: pytest, pytest-cov, Ruff, Pyright/mypy, Hypothesis where useful, Bandit/pip-audit when relevant.

Every defect: reproduce → failing regression test → fix → targeted test → module regression → log.
