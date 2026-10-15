import json
import pytest
from backend.settings import REGISTRY
from backend.worker import verify_runtime_versions

def test_frozen_runtime_versions_match():
    registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
    verify_runtime_versions(registry['runtime_versions'])

def test_dependency_drift_fails_before_model_loading(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'unverified')
    with pytest.raises(ValueError,match='Runtime dependency drift'):
        verify_runtime_versions({'torch':'2.13.0+cu126'})
