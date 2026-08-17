# Test Plan

Required:
- Mel [B,16,D]
- graph [B,1,D]
- RP [B,16,D]
- nonlinear [B,17,D]
- gradient flow

Tools: pytest, pytest-cov, Ruff, Pyright/mypy, Hypothesis where useful, Bandit/pip-audit when relevant.

Every defect: reproduce → failing regression test → fix → targeted test → module regression → log.
