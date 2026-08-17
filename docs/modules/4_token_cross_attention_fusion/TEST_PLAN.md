# Test Plan

Required:
- Q/KV shapes
- attention backward
- residual norm
- ablation path
- probability
- metrics

Tools: pytest, pytest-cov, Ruff, Pyright/mypy, Hypothesis where useful, Bandit/pip-audit when relevant.

Every defect: reproduce → failing regression test → fix → targeted test → module regression → log.
