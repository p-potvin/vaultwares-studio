"""Use clean processes: an earlier test importing USD would hide this failure."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["gui_app", "gui.viewport"])
def test_gui_entrypoint_can_use_usd_after_import(module):
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}; from pxr import Usd; assert Usd.Stage.CreateInMemory()"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
