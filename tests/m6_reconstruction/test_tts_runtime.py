import ast
import io
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vaaniswitch.m6_reconstruction.config import VOICE_ID, RuntimeConfig
from vaaniswitch.m6_reconstruction.runtime import bounded_worker
from vaaniswitch.m6_reconstruction.tts import select_required_voice, synthesize_smoke


def test_exact_zira_required() -> None:
    zira = SimpleNamespace(id=VOICE_ID, name="Microsoft Zira Desktop - English (United States)")
    david = SimpleNamespace(id=VOICE_ID.replace("ZIRA", "DAVID"), name="Microsoft David Desktop")
    assert select_required_voice([david, zira]) is zira
    for voices in ([], [david], [zira, zira], [SimpleNamespace(id=VOICE_ID, name="David")]):
        with pytest.raises(ValueError, match="Zira"):
            select_required_voice(voices)


def test_sapi_save_is_queued_before_run_loop(
    tmp_path: Path, config_values: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model pyttsx3 2.99's idle proxy: an idle save precedes and strands runAndWait."""
    voice = SimpleNamespace(
        id=VOICE_ID,
        name="Microsoft Zira Desktop",
        languages=["en-US"],
        gender="Female",
        age="Adult",
    )

    class Engine:
        def __init__(self) -> None:
            self.busy = True
            self.props: dict[str, Any] = {"voices": [voice]}
            self.queued: tuple[str, str] | None = None

        def getProperty(self, key: str) -> Any:
            return self.props[key]

        def setProperty(self, key: str, value: Any) -> None:
            self.props[key] = value

        def startLoop(self, useDriverLoop: bool) -> None:
            assert not useDriverLoop

        def iterate(self) -> None:
            self.busy = False

        def endLoop(self) -> None:
            self.busy = True

        def save_to_file(self, text: str, path: str) -> None:
            assert self.busy, "save on idle proxy executes before runAndWait and strands loop"
            self.queued = (text, path)

        def runAndWait(self) -> None:
            if self.queued:
                stream = io.BytesIO()
                with wave.open(stream, "wb") as wav:
                    wav.setparams((1, 2, 22050, 100, "NONE", "not compressed"))
                    wav.writeframes(b"\x01\x02" * 100)
                Path(self.queued[1]).write_bytes(stream.getvalue())
            self.busy = False

        def stop(self) -> None:
            pass

    import pyttsx3  # type: ignore[import-untyped]

    monkeypatch.setattr(pyttsx3, "init", lambda **kwargs: Engine())
    result = synthesize_smoke(
        RuntimeConfig(tmp_path, config_values), tmp_path / "artifacts/module6/synthetic"
    )
    assert result["actual_settings"]["voice_id"] == VOICE_ID
    assert result["validation"]["nonzero_samples"] == 100


def test_worker_timeout_preserves_log(tmp_path: Path) -> None:
    path = tmp_path / "artifacts/module6/run/timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        bounded_worker(tmp_path, [sys.executable, "-c", "import time; time.sleep(10)"], path, 1)
    assert path.exists()


def test_no_upstream_or_metrics_imports(project_root: Path) -> None:
    forbidden = (
        "vaaniswitch.m0",
        "vaaniswitch.m1",
        "vaaniswitch.m2",
        "vaaniswitch.m3",
        "vaaniswitch.m4",
        "vaaniswitch.m5",
        "vaaniswitch.m7",
    )
    for path in (project_root / "src/vaaniswitch/m6_reconstruction").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith(forbidden) for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden)
