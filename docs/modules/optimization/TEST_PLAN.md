# Test Plan

Required:
- train-only augmentation
- tuning isolation
- fold aggregation
- robustness
- ablation runner

Tools: pytest, pytest-cov, Ruff, Pyright/mypy, Hypothesis where useful, Bandit/pip-audit when relevant.

Every defect: reproduce → failing regression test → fix → targeted test → module regression → log.
