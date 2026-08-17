# Test Plan

Required:
- logical ID
- mic priority/fallback
- no-audio exclusion
- speaker folds
- mic leakage
- WAV audit

Tools: pytest, pytest-cov, Ruff, Pyright/mypy, Hypothesis where useful, Bandit/pip-audit when relevant.

Every defect: reproduce → failing regression test → fix → targeted test → module regression → log.
