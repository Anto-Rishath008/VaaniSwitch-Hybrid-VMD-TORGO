# Test Plan

Required:
- API
- invalid upload
- Gradio E2E
- Playwright
- k6
- Bandit/pip-audit
- full regression

Tools: pytest, pytest-cov, Ruff, Pyright/mypy, Hypothesis where useful, Playwright, k6, Bandit, pip-audit.

Every defect: reproduce → failing regression test → fix → targeted test → module regression → log.
