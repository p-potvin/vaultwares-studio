"""Shared Qt signal bundle for cross-tab dashboard updates.

Lives in ``gui.signals`` so widgets in any submodule can ``from gui.signals
import TaskSignals`` without pulling the whole gui_app entry point.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class TaskSignals(QObject):
    log = Signal(str)
    manifest_changed = Signal(object)
    running_changed = Signal(bool)
    api_test_finished = Signal(bool, str)
