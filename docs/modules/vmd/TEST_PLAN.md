# Test Plan

Required:
- VMD config
- multitone decomposition
- all-mode reconstruction
- cache
- short audio

Tools: pytest, pytest-cov, Ruff, Pyright/mypy, Hypothesis where useful, Bandit/pip-audit when relevant.

Every defect: reproduce → failing regression test → fix → targeted test → module regression → log.
