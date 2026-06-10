"""Interactive 3D splat viewport: QWebEngineView + vendored GaussianSplats3D.

The web side (vaultwares_studio/webviewer/) renders the gaussian splat with
three.js via the browser GPU path — the local CUDA GPU stays free. Assets are
served by a loopback-only HTTP server (127.0.0.1, random port):

    /...       -> vaultwares_studio/webviewer/
    /job/...   -> the active job's output directory

A custom URL scheme was tried first, but Chromium's Fetch API refuses custom
schemes for the splat download even with FetchApiAllowed (worker contexts
ignore scheme flags) — the loopback server makes fetch/workers/CORS behave
like the normal web.

Python <-> JS via QWebChannel ("bridge" object). Captured camera poses are
appended to <job>/usd/captured_cameras.json; the CameraEntity/keyframe
integration extends this in the next M2 slice.

register_viewer_scheme() MUST be called before QApplication is constructed
(kept for the vw:// scheme used by static assets in earlier sessions; the
viewer itself now runs over the loopback server).
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

# Readable fallbacks when no translator is supplied (standalone viewer).
_DEFAULT_STRINGS = {
    "viewport_reload": "Reload Scene",
    "viewport_capture": "Capture Camera",
    "viewport_loading": "Loading reconstruction…",
    "viewport_no_scene": "No reconstruction yet — run a job, then reload.",
    "viewport_no_webengine": "QtWebEngine is unavailable on this system.",
    "viewport_captured": "Camera captured ({count} total) — saved to captured_cameras.json.",
    "viewport_cameras": "Captured cameras",
    "viewport_move_up": "Move Up",
    "viewport_move_down": "Move Down",
    "viewport_delete": "Delete",
    "viewport_preview_path": "Preview Path",
    "viewport_need_two": "Capture at least 2 cameras to preview a path.",
}

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


class _ViewerRequestHandler(SimpleHTTPRequestHandler):
    """Serves the viewer app, with /job/* mapped into the active job dir."""

    server_version = "VWStudioViewer/1.0"

    def __init__(self, *args, viewer_server=None, **kwargs):
        self._viewer_server = viewer_server
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        if clean.startswith("job/"):
            job_root = self._viewer_server.job_root
            if job_root is None:
                return str(WEBVIEWER_DIR / "__missing__")
            target = (job_root / clean[len("job/"):]).resolve()
            base = job_root.resolve()
        else:
            target = (WEBVIEWER_DIR / (clean or "index.html")).resolve()
            base = WEBVIEWER_DIR.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return str(WEBVIEWER_DIR / "__missing__")
        return str(target)

    def log_message(self, *_args) -> None:  # quiet
        pass


class ViewerServer:
    """Loopback-only static server for the viewport assets and job artifacts."""

    def __init__(self) -> None:
        self.job_root: Path | None = None
        handler = partial(_ViewerRequestHandler, viewer_server=self)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self._httpd.server_address[1]
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()

    def url(self, path: str = "index.html") -> str:
        return f"http://127.0.0.1:{self.port}/{path}"

    def close(self) -> None:
        self._httpd.shutdown()


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
        # Without this, the Fetch API refuses vw:// URLs entirely and the
        # splat never loads ("URL scheme 'vw' is not supported").
        | QWebEngineUrlScheme.Flag.FetchApiAllowed
    )
    QWebEngineUrlScheme.registerScheme(scheme)
    _scheme_registered = True


if WEBENGINE_AVAILABLE:

    class _VwSchemeHandler(QWebEngineUrlSchemeHandler):
        """Serves the viewer app and the active job's artifacts.

        Everything lives under the single origin vw://app — pages and the
        splat data must share an origin or the viewer's fetch() of the scene
        is CORS-blocked (QWebEngineUrlRequestJob can't attach CORS headers).
        ``vw://app/job/...`` maps into the active job's output directory.
        """

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.app_root = WEBVIEWER_DIR
            self.job_root: Path | None = None

        def set_job_root(self, job_dir: Path) -> None:
            self.job_root = Path(job_dir)

        def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802
            url = job.requestUrl()
            if url.host() != "app":
                job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
                return
            relative = url.path().lstrip("/")
            root = self.app_root
            if relative.startswith("job/"):
                if self.job_root is None:
                    job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
                    return
                root = self.job_root
                relative = relative[len("job/"):]
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

    from PySide6.QtWebEngineCore import QWebEnginePage

    class _ViewportPage(QWebEnginePage):
        """Routes the page's JS console into the app log — no more silent failures."""

        def __init__(self, profile, parent, log) -> None:
            super().__init__(profile, parent)
            self._log = log

        def javaScriptConsoleMessage(self, level, message, line, source) -> None:  # noqa: N802
            self._log(f"[js:{level.name.replace('MessageLevel', '').lower()}] {message} ({source}:{line})")

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
        base_translate = translate or (lambda key: key)
        self._t = lambda key: (
            value if (value := base_translate(key)) != key else _DEFAULT_STRINGS.get(key, key)
        )
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

        body = QHBoxLayout()
        body.setSpacing(6)
        layout.addLayout(body, 1)

        self.web_view = QWebEngineView(self)
        # Native surface: avoids blank rendering when the parent window uses
        # DWM backdrop effects (observed with the Fluent shell's Mica).
        from PySide6.QtCore import Qt

        self.web_view.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        body.addWidget(self.web_view, 1)

        # Captured-cameras panel: reorder/delete the walkthrough stops and
        # preview the resulting path in the viewer before paying for a render.
        panel = QVBoxLayout()
        panel.setSpacing(4)
        self.cameras_label = QLabel(self._t("viewport_cameras"), self)
        panel.addWidget(self.cameras_label)
        self.camera_list = QListWidget(self)
        self.camera_list.setMaximumWidth(230)
        panel.addWidget(self.camera_list, 1)
        self.move_up_btn = QPushButton(self._t("viewport_move_up"), self)
        self.move_down_btn = QPushButton(self._t("viewport_move_down"), self)
        self.delete_btn = QPushButton(self._t("viewport_delete"), self)
        self.preview_path_btn = QPushButton(self._t("viewport_preview_path"), self)
        for button in (self.move_up_btn, self.move_down_btn, self.delete_btn, self.preview_path_btn):
            button.setMaximumWidth(230)
            panel.addWidget(button)
        body.addLayout(panel)

        self.move_up_btn.clicked.connect(lambda: self._move_camera(-1))
        self.move_down_btn.clicked.connect(lambda: self._move_camera(1))
        self.delete_btn.clicked.connect(self._delete_camera)
        self.preview_path_btn.clicked.connect(self._preview_path)

        self._page = _ViewportPage(self.web_view.page().profile(), self.web_view, self._set_status)
        self.web_view.setPage(self._page)

        self._server = ViewerServer()

        self._bridge = ViewportBridge(self)
        self._channel = QWebChannel(self)
        self._channel.registerObject("bridge", self._bridge)
        self._page.setWebChannel(self._channel)

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
            self._server.job_root = self._job_dir
            self._refresh_cameras()
        self.reload_scene()

    # -- captured-cameras panel -------------------------------------------------

    @property
    def _captured_path(self) -> Path | None:
        return self._job_dir / "usd" / "captured_cameras.json" if self._job_dir else None

    def _load_captured(self) -> list[dict]:
        path = self._captured_path
        if path is None or not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save_captured(self, cameras: list[dict]) -> None:
        path = self._captured_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
        self._refresh_cameras()

    def _refresh_cameras(self) -> None:
        if self.web_view is None:
            return
        selected = self.camera_list.currentRow()
        self.camera_list.clear()
        for camera in self._load_captured():
            self.camera_list.addItem(camera.get("name", "Camera"))
        if 0 <= selected < self.camera_list.count():
            self.camera_list.setCurrentRow(selected)

    def _move_camera(self, delta: int) -> None:
        cameras = self._load_captured()
        row = self.camera_list.currentRow()
        target = row + delta
        if row < 0 or not (0 <= target < len(cameras)):
            return
        cameras[row], cameras[target] = cameras[target], cameras[row]
        self._save_captured(cameras)
        self.camera_list.setCurrentRow(target)

    def _delete_camera(self) -> None:
        cameras = self._load_captured()
        row = self.camera_list.currentRow()
        if not (0 <= row < len(cameras)):
            return
        del cameras[row]
        self._save_captured(cameras)

    def _preview_path(self) -> None:
        from vaultwares_studio.camera_paths import build_visit_path, load_captured_entities, sample_path

        path = self._captured_path
        entities = load_captured_entities(path) if path else []
        visit = build_visit_path(entities)
        if visit is None:
            self._set_status(self._t("viewport_need_two"))
            return
        frames = [
            {"position": [float(v) for v in pos], "lookAt": [float(v) for v in target]}
            for pos, target in sample_path(visit, fps=30)
        ]
        self.web_view.page().runJavaScript(f"window.playPath({json.dumps(frames)}, 30);")

    def reload_scene(self) -> None:
        if self.web_view is None or self._job_dir is None:
            return
        splat = self._job_dir / "reconstruction" / "cloud.ply"
        url = QUrl(self._server.url())
        if splat.exists():
            query = ["scene=job/reconstruction/cloud.ply"]
            frame = self._scene_framing()
            if frame:
                query.append(frame)
            url.setQuery("&".join(query))
            self._set_status(self._t("viewport_loading"))
        else:
            self._set_status(self._t("viewport_no_scene"))
        self.capture_btn.setEnabled(False)
        self.web_view.load(url)

    def _scene_framing(self) -> str:
        """Centroid + radius from the preview cloud so the camera starts framed."""
        preview = self._job_dir / "reconstruction" / "cloud_preview.ply" if self._job_dir else None
        if preview is None or not preview.exists():
            return ""
        try:
            import numpy as np
            from plyfile import PlyData

            vertex = PlyData.read(str(preview))["vertex"]
            points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
            # Robust to outlier fliers: frame the 5th-95th percentile box.
            low, high = np.percentile(points, [5, 95], axis=0)
            center = (low + high) / 2
            radius = float(np.linalg.norm(high - low) / 2) or 1.0
            return (
                f"cx={center[0]:.3f}&cy={center[1]:.3f}&cz={center[2]:.3f}&r={radius:.3f}"
            )
        except Exception:  # noqa: BLE001 - framing is best-effort
            return ""

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
        self._refresh_cameras()
        self._set_status(self._t("viewport_captured").format(count=len(cameras)))
