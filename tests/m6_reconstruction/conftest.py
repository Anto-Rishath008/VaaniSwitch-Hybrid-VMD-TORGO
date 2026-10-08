from pathlib import Path
from typing import Any

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "runtime: opt-in validation of actual retained runtime smoke"
    )


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def config_values(project_root: Path) -> dict[str, Any]:
    import yaml

    return dict(yaml.safe_load((project_root / "configs/module6/pipeline.yaml").read_text("utf-8")))
