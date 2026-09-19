import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.strings import t as _t

# We'll need the VaultWares themes submodule
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "vaultwares-themes" / "theme-manager" / "exports"))
from qt_exporter import QtThemeExporter

from gui.views.pipeline_workspace import PipelineWorkspace
from gui.views.settings_dialog import SettingsDialog
from gui.viewport import ViewportPanel

ICON_PATH = _REPO_ROOT / "vaultwares-themes" / "assets" / "icons" / "vaultwares-favicon-gold-filled-256.png"
LOGO_PATH = _REPO_ROOT / "vaultwares-themes" / "assets" / "logos" / "vaultwares-minimal-gold-filled.png"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VaultWares Studio")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.resize(1440, 900)

        # Initialize Exporter
        self.exporter = QtThemeExporter()

        self.setup_ui()
        self.apply_themes()
        # A .ply or .splat dropped anywhere on the window opens in the
        # viewport, job or no job. The console's raw artifacts are not in the
        # job layout, and opening one should not require making them so.
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self.viewport_panel.droppable_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        path = self.viewport_panel.droppable_path(event.mimeData())
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.pipeline_workspace.append_log(f"Opening dropped file: {path}")
        self.viewport_panel.load_file(path)
        self.content_stack.setCurrentWidget(self.viewport_panel)
        self.btn_open_viewport.setText("BACK TO PIPELINE")

    def setup_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("WarmShell")
        self.setCentralWidget(central_widget)

        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Top Navigation Bar (Warm Mode)
        self.nav_bar = QFrame(self)
        self.nav_bar.setObjectName("TitleBar")
        self.nav_bar.setFixedHeight(60)

        nav_layout = QHBoxLayout(self.nav_bar)
        nav_layout.setContentsMargins(30, 0, 30, 0)
        nav_layout.setSpacing(24)

        # Brand
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(8)
        if LOGO_PATH.exists():
            logo_pixmap = QPixmap(str(LOGO_PATH))
            logo_label = QLabel()
            logo_label.setPixmap(logo_pixmap.scaled(28, 28, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            brand_layout.addWidget(logo_label)
        brand_text = QLabel("VaultWares Studio")
        brand_text.setObjectName("LogoLabel")
        brand_text.setStyleSheet("font-size: 18px; font-weight: 700; letter-spacing: 1px;")
        brand_layout.addWidget(brand_text)
        nav_layout.addLayout(brand_layout)

        nav_layout.addStretch(1)

        # Job selector
        job_label = QLabel("Job:")
        job_label.setStyleSheet("font-size: 11px; color: rgba(22, 19, 32, 0.5);")
        nav_layout.addWidget(job_label)

        self.job_combo = QComboBox(self)
        self.job_combo.setMinimumWidth(220)
        self.job_combo.currentIndexChanged.connect(self._on_job_selected)
        nav_layout.addWidget(self.job_combo)

        nav_layout.addSpacing(16)

        # Action Buttons
        self.btn_open_viewport = QPushButton("LAUNCH 3D VIEWPORT")
        self.btn_settings = QPushButton("SETTINGS")

        nav_layout.addWidget(self.btn_open_viewport)
        nav_layout.addWidget(self.btn_settings)

        self.main_layout.addWidget(self.nav_bar)

        # 2. Main Content Area (Console Mode Container)
        # Pipeline and viewport are stacked pages, not separate windows. The
        # viewport's QWebEngineView loses its GL surface when it lives in its
        # own top-level window, so it stays embedded here and the toolbar
        # button swaps pages instead of spawning a window.
        self.workspace_container = QWidget()
        workspace_layout = QVBoxLayout(self.workspace_container)
        workspace_layout.setContentsMargins(10, 10, 10, 10)

        self.content_stack = QStackedWidget(self)

        self.pipeline_workspace = PipelineWorkspace(self)
        self.content_stack.addWidget(self.pipeline_workspace)

        self.viewport_panel = ViewportPanel(parent=self, translate=_t)
        self.viewport_panel.log.connect(self.pipeline_workspace.append_log)
        self.content_stack.addWidget(self.viewport_panel)

        workspace_layout.addWidget(self.content_stack)
        self.main_layout.addWidget(self.workspace_container, 1)

        # Connections
        self.btn_open_viewport.clicked.connect(self.toggle_viewport)
        self.btn_settings.clicked.connect(self.open_settings)

        # Populate job list after workspace is ready
        self._refresh_job_list()

    def _refresh_job_list(self):
        """Populate the job dropdown with available jobs (newest first)."""
        from vaultwares_studio.pipeline import list_job_manifests, load_job_manifest

        self.job_combo.blockSignals(True)
        self.job_combo.clear()

        manifests = list_job_manifests()
        current_job_id = (
            self.pipeline_workspace.manifest.job_id
            if self.pipeline_workspace and self.pipeline_workspace.manifest
            else None
        )

        self.job_combo.addItem("— New Job —", userData=None)
        for path in manifests:
            try:
                manifest = load_job_manifest(path)
                label = manifest.job_id
                if manifest.source_video:
                    label += f"  ({Path(manifest.source_video).name})"
                self.job_combo.addItem(label, userData=str(path))
            except Exception:
                continue

        # Select the current job if present
        if current_job_id:
            for i in range(self.job_combo.count()):
                data = self.job_combo.itemData(i)
                if data and current_job_id in data:
                    self.job_combo.setCurrentIndex(i)
                    break

        self.job_combo.blockSignals(False)

    def _on_job_selected(self, index: int):
        """Load the selected job manifest into the workspace."""
        from vaultwares_studio.pipeline import load_job_manifest

        data = self.job_combo.currentData()
        if data is None:
            return
        try:
            manifest = load_job_manifest(data)
            self.pipeline_workspace.signals.manifest_changed.emit(manifest)
            self.pipeline_workspace.append_log(f"Loaded job: {manifest.job_id}")
        except Exception as exc:
            self.pipeline_workspace.append_log(f"[ERROR] Failed to load job: {exc}")

    def apply_themes(self):
        warm_qss = self.exporter.generate_revisited_warm_qss()
        console_qss = self.exporter.generate_revisited_console_qss()
        self.setStyleSheet(warm_qss)
        self.pipeline_workspace.setStyleSheet(console_qss)
        self.viewport_panel.setStyleSheet(console_qss)

    def toggle_viewport(self):
        """Swap the content stack between the pipeline and the 3D viewport."""
        showing_viewport = self.content_stack.currentWidget() is self.viewport_panel
        if showing_viewport:
            self.content_stack.setCurrentWidget(self.pipeline_workspace)
            self.btn_open_viewport.setText("LAUNCH 3D VIEWPORT")
            return

        # Point the viewport at the current job before showing it.
        if self.pipeline_workspace and self.pipeline_workspace.manifest:
            self.viewport_panel.set_job(Path(self.pipeline_workspace.manifest.output_dir))

        self.content_stack.setCurrentWidget(self.viewport_panel)
        self.btn_open_viewport.setText("BACK TO PIPELINE")

    def open_settings(self):
        dlg = SettingsDialog(
            parent=self,
            strict_mode=self.pipeline_workspace.strict_mode,
            on_strict_toggled=self._on_strict_toggled,
        )
        dlg.setStyleSheet(self.exporter.generate_revisited_console_qss())
        dlg.exec()

    def _on_strict_toggled(self, checked: bool):
        self.pipeline_workspace.strict_mode = checked
        self.pipeline_workspace.append_log(
            f"Strict mode: {'ON' if checked else 'OFF'}"
        )
