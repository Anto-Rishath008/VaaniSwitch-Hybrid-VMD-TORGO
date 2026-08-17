# Testing Strategy

Reject any result if speakers overlap, microphone copies cross folds, evaluation statistics leak into training, or tuning uses evaluation folds.

Tools: pytest, pytest-cov, Ruff, Pyright/mypy, Hypothesis where useful, Playwright, k6, Bandit, pip-audit.

Do not run full VMD caching/training inside unit tests. Use synthetic/tiny fixtures.

Defect lifecycle: `OPEN → FIXING → RETEST → CLOSED`. Every fix gets a regression test.
