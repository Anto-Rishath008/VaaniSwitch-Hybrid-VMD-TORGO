"""Tests for the preprocessing pipeline script."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from components.preprocess import main


def test_preprocess_action_run() -> None:
    """Test preprocess run pipeline execution by mocking config loaders and pipeline runs."""
    with patch("components.preprocess.TorgoConfig.from_yaml") as mock_cfg0, \
         patch("components.preprocess.Module1Config.from_yaml") as mock_cfg1, \
         patch("components.preprocess.Module2Config.from_yaml") as mock_cfg2, \
         patch("components.preprocess.run_module0") as mock_m0, \
         patch("components.preprocess.run_module1") as mock_m1, \
         patch("components.preprocess.run_module2") as mock_m2:

        # Set return values for mocks
        mock_cfg0.return_value = MagicMock()
        mock_cfg1.return_value = MagicMock()
        mock_cfg2.return_value = MagicMock()
        mock_m2.return_value = {"status": "success"}

        sys_argv = [
            "preprocess.py",
            "run",
            "--m2-config", "configs/representations/default.yaml",
            "--m0-config", "configs/data/torgo.yaml",
            "--m1-config", "configs/vmd/default.yaml",
            "--limit", "2",
        ]

        with patch.object(sys, "argv", sys_argv):
            assert main() == 0

        mock_cfg0.assert_called_once()
        mock_cfg1.assert_called_once()
        mock_cfg2.assert_called_once()
        mock_m0.assert_called_once()
        mock_m1.assert_called_once()
        mock_m2.assert_called_once()


def test_preprocess_action_validate() -> None:
    """Test preprocess validate execution by mocking config loaders and validation runs."""
    with patch("components.preprocess.TorgoConfig.from_yaml") as mock_cfg0, \
         patch("components.preprocess.Module1Config.from_yaml") as mock_cfg1, \
         patch("components.preprocess.Module2Config.from_yaml") as mock_cfg2, \
         patch("components.preprocess.validate_generated_outputs") as mock_m0, \
         patch("components.preprocess.validate_module1") as mock_m1, \
         patch("components.preprocess.validate_module2") as mock_m2:

        # Set return values for mocks
        mock_cfg0.return_value = MagicMock()
        mock_cfg1.return_value = MagicMock()
        mock_cfg2.return_value = MagicMock()
        mock_m2.return_value = {"status": "success"}

        sys_argv = [
            "preprocess.py",
            "validate",
            "--m2-config", "configs/representations/default.yaml",
            "--m0-config", "configs/data/torgo.yaml",
            "--m1-config", "configs/vmd/default.yaml",
            "--limit", "2",
        ]

        with patch.object(sys, "argv", sys_argv):
            assert main() == 0

        mock_cfg0.assert_called_once()
        mock_cfg1.assert_called_once()
        mock_cfg2.assert_called_once()
        mock_m0.assert_called_once()
        mock_m1.assert_called_once()
        mock_m2.assert_called_once()
