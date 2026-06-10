"""Interactive 3D splat viewport: QWebEngineView + vendored GaussianSplats3D.

The web side (vaultwares_studio/webviewer/) renders the gaussian splat with
three.js via the browser GPU path — the local CUDA GPU stays free. Assets are
served through a custom ``vw://`` URL scheme (no web server, no file:// CORS
problems):

    vw://app/...   -> vaultwares_studio/webviewer/
    vw://job/...   -> the active job's output directory

Python <-> JS via QWebChannel ("bridge" object). Captured camera poses are
appended to <job>/usd/captured_cameras.json; the CameraEntity/keyframe
integration extends this in the next M2 slice.

register_viewer_scheme() MUST be called before QApplication is constructed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

try:
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import (
        QWebEngineUrlRequestJob,
        QWebEngineUrlScheme,
        QWebEngineUrlSchemeHandler,
    )
    from PySide6.QtWebEngineWidgets import QWebEngineView

    WEBENGINE_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without QtWebEngine
    WEBENGINE_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
WEBVIEWER_DIR = ROOT / "vaultwares_studio" / "webviewer"
SCHEME = b"vw"

_MIME_TYPES = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".ply": "application/octet-stream",
    ".ksplat": "application/octet-stream",
    ".splat": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".wasm": "application/wasm",
}

_scheme_registered = False


def register_viewer_scheme() -> None:
    """Register the vw:// scheme. Call BEFORE constructing QApplication."""
    global _scheme_registered
    if _scheme_registered or not WEBENGINE_AVAILABLE:
        return
    scheme = QWebEngineUrlScheme(SCHEME)
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.HostAndPort)
    scheme.setFlags(
        QWebEngineUrlScheme.Flag.SecureScheme
        | QWebEngineUrlScheme.Flag.LocalAccessAllowed
        | QWebEngineUrlScheme.Flag.CorsEnabled
    )
    QWebEngineUrlScheme.registerScheme(scheme)
    _scheme_registered = True


if WEBENGINE_AVAILABLE:

    class _VwSchemeHandler(QWebEngineUrlSchemeHandler):
        """Serves vw://app/* from the webviewer dir, vw://job/* from the job dir."""

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.roots: dict[str, Path] = {"app": WEBVIEWER_DIR}

        def set_job_root(self, job_dir: Path) -> None:
            self.roots["job"] = Path(job_dir)

        def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802
            url = job.requestUrl()
            root = self.roots.get(url.host())
            if root is None:
                job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
                return
            relative = url.path().lstrip("/")
            target = (root / relative).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
                return
            if not target.is_file():
                job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
                return
            buffer = QBuffer(job)
            buffer.setData(QByteArray(target.read_bytes()))
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            mime = _MIME_TYPES.get(target.suffix.lower(), "application/octet-stream")
            job.reply(mime.encode("ascii"), buffer)

    class ViewportBridge(QObject):
        """JS-facing object exposed through QWebChannel as ``bridge``."""

        camera_captured = Signal(dict)
        viewer_ready = Signal()
        js_message = Signal(str)

        @Slot(str)
        def cameraCaptured(self, payload: str) -> None:  # noqa: N802
            try:
                self.camera_captured.emit(json.loads(payload))
            except json.JSONDecodeError:
                self.js_message.emit(f"Bad camera payload: {payload[:120]}")

        @Slot()
        def viewerReady(self) -> None:  # noqa: N802
            self.viewer_ready.emit()

        @Slot(str)
        def jsLog(self, message: str) -> None:  # noqa: N802
            self.js_message.emit(message)


class ViewportTab(QFrame):
    """Viewport tab: fly around the reconstruction, capture camera poses."""

    log = Signal(str)

    def __init__(
        self,
        parent=None,
        translate: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setObjectName("Viewport")
        self._t = translate or (lambda key: key)
        self._job_dir: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        self.reload_btn = QPushButton(self._t("viewport_reload"), self)
        self.capture_btn = QPushButton(self._t("viewport_capture"), self)
        self.capture_btn.setEnabled(False)
        self.status_label = QLabel(self._t("viewport_no_scene"), self)
        self.status_label.setWordWrap(True)
        toolbar.addWidget(self.reload_btn)
        toolbar.addWidget(self.capture_btn)
        toolbar.addWidget(self.status_label, 1)
        layout.addLayout(toolbar)

        if not WEBENGINE_AVAILABLE:
            fallback = QLabel(self._t("viewport_no_webengine"), self)
            fallback.setWordWrap(True)
            layout.addWidget(fallback, 1)
            self.web_view = None
            return

        self.web_view = QWebEngineView(self)
        layout.addWidget(self.web_view, 1)

        self._handler = _VwSchemeHandler(self)
        self.web_view.page().profile().installUrlSchemeHandler(SCHEME, self._handler)

        self._bridge = ViewportBridge(self)
        self._channel = QWebChannel(self)
        self._channel.registerObject("bridge", self._bridge)
        self.web_view.page().setWebChannel(self._channel)

        self._bridge.camera_captured.connect(self._on_camera_captured)
        self._bridge.viewer_ready.connect(lambda: self.capture_btn.setEnabled(True))
        self._bridge.js_message.connect(self._set_status)
        self.reload_btn.clicked.connect(self.reload_scene)
        self.capture_btn.clicked.connect(
            lambda: self.web_view.page().runJavaScript("window.captureCamera();")
        )

    # -- public API ------------------------------------------------------------

    def set_job(self, job_dir: Path | str) -> None:
        """Point the viewport at a job; loads its reconstruction if present."""
        self._job_dir = Path(job_dir)
        if self.web_view is not None:
            self._handler.set_job_root(self._job_dir)
        self.reload_scene()

    def reload_scene(self) -> None:
        if self.web_view is None or self._job_dir is None:
            return
        splat = self._job_dir / "reconstruction" / "cloud.ply"
        if splat.exists():
            url = QUrl("vw://app/index.html")
            url.setQuery("scene=vw://job/reconstruction/cloud.ply")
            self._set_status(self._t("viewport_loading"))
        else:
            url = QUrl("vw://app/index.html")
            self._set_status(self._t("viewport_no_scene"))
        self.capture_btn.setEnabled(False)
        self.web_view.load(url)

    # -- internals ---------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.log.emit(f"[viewport] {message}")

    def _on_camera_captured(self, pose: dict) -> None:
        if self._job_dir is None:
            return
        usd_dir = self._job_dir / "usd"
        usd_dir.mkdir(parents=True, exist_ok=True)
        store = usd_dir / "captured_cameras.json"
        cameras: list[dict] = []
        if store.exists():
            try:
                cameras = json.loads(store.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cameras = []
        pose["name"] = f"Captured {len(cameras) + 1}"
        cameras.append(pose)
        store.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
        self._set_status(self._t("viewport_captured").format(count=len(cameras)))
