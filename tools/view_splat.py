"""Standalone splat viewer — the Viewport in its own window.

The exact widget the app embeds, opened top-level (this is what the
debugging screenshots used, and it renders reliably where the embedded tab
misbehaves). Camera capture works here too: poses land in the job's
usd/captured_cameras.json.

Usage:
    .venv\\Scripts\\python.exe tools\\view_splat.py                # latest started job
    .venv\\Scripts\\python.exe tools\\view_splat.py <job-dir>      # specific job
    .venv\\Scripts\\python.exe tools\\view_splat.py <file.ply>     # any 3DGS .ply
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--enable-unsafe-swiftshader")

from gui.viewport import ViewportTab, register_viewer_scheme  # noqa: E402

register_viewer_scheme()
from PySide6.QtWidgets import QApplication  # noqa: E402

from vaultwares_studio.pipeline import (  # noqa: E402
    completed_stage_count,
    list_job_manifests,
    load_job_manifest,
)


def _latest_started_job_dir() -> Path | None:
    for manifest_path in list_job_manifests():
        try:
            manifest = load_job_manifest(manifest_path)
        except Exception:  # noqa: BLE001
            continue
        if completed_stage_count(manifest) > 0:
            return Path(manifest.output_dir)
    return None


def main() -> int:
    app = QApplication(["VaultWares Splat Viewer"])
    tab = ViewportTab(translate=lambda key: key)
    tab.setWindowTitle("VaultWares Splat Viewer")
    tab.log.connect(lambda message: print(message, flush=True))
    tab.resize(1600, 950)

    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _latest_started_job_dir()
    if target is None:
        print("No job with completed stages found. Pass a job dir or .ply path.")
        return 1

    if target.is_file() and target.suffix.lower() == ".ply":
        # Serve the file's parent as the job root and load it directly.
        tab._server.job_root = target.parent
        tab._job_dir = target.parent
        from PySide6.QtCore import QUrl

        url = QUrl(tab._server.url())
        url.setQuery(f"scene=job/{target.name}")
        tab.show()
        tab.web_view.load(url)
    else:
        tab.show()
        tab.set_job(target)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
