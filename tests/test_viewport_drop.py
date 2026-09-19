"""Dropping a .ply onto the window opens it, with or without a job.

The console's artifacts are not in the job layout — ``combined_pcd.ply`` is a
loose file in a folder — and looking at one should not require building a job
around it. These tests cover the file-selection half, which is where the
mistakes are: the wrong suffix, a remote URL, several files at once.

Run in a clean subprocess, like ``test_gui_usd_import``: the panel imports
QtWebEngine, which on Windows needs USD loaded first, and by the time pytest
reaches this module another test may already have imported Qt.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

PROBE = textwrap.dedent(
    """
    import sys, json
    sys.path.insert(0, {root!r})
    import gui  # USD before Qt, as gui_app.py does
    from PySide6.QtCore import QMimeData, QUrl
    from gui.viewport import ViewportPanel

    def check(*urls):
        data = QMimeData()
        data.setUrls([QUrl.fromLocalFile(u) if not u.startswith("http") else QUrl(u) for u in urls])
        found = ViewportPanel.droppable_path(data)
        return None if found is None else found.name

    print(json.dumps({body}))
    """
)


def _probe(expression: str, root: str) -> object:
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(root=root, body=expression)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    import json

    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def root() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1])


def test_point_cloud_and_splat_suffixes_are_accepted(root, tmp_path):
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    names = ["combined_pcd.ply", "cloud.splat", "cloud.ksplat", "CLOUD.PLY"]
    for name in names:
        (tmp_path / name).write_bytes(b"ply\n")
    got = _probe(
        "[check(p) for p in %r]" % [str(tmp_path / n) for n in names], root
    )
    assert got == names


def test_other_files_and_empty_drops_are_refused(root, tmp_path):
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    other = tmp_path / "notes.txt"
    other.write_text("not a cloud")
    assert _probe("[check(%r), check()]" % str(other), root) == [None, None]


def test_the_first_cloud_in_a_multi_file_drop_wins(root, tmp_path):
    """Dragging a folder's worth of files should open one, not the first file."""
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.ply").write_bytes(b"ply\n")
    got = _probe("check(%r, %r)" % (str(tmp_path / "a.txt"), str(tmp_path / "b.ply")), root)
    assert got == "b.ply"


def test_a_remote_url_is_refused(root):
    """A drag from a browser carries an http URL; there is no local file to open."""
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    assert _probe("check('https://example.com/cloud.ply')", root) is None
