import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.strings import t as _t

# We'll need the VaultWares themes submodule
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "vaultwares-themes" / "theme-manager" / "exports"))
from qt_exporter import QtThemeExporter

from gui.views.pipeline_workspace import PipelineWorkspace
from gui.viewport import ViewportWindow

ICON_PATH = _REPO_ROOT / "vaultwares-studio" / "Brand" / "favicons" / "vaultwares-favicon-gold-filled-256.png"

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

    def setup_ui(self):
        # The main window itself will act as the Warm Mode frame
        central_widget = QWidget(self)
        central_widget.setObjectName("WarmShell")
        self.setCentralWidget(central_widget)
        
        # Main layout
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
        brand_label = QLabel("\U0001f537 VaultWares Studio")
        brand_label.setObjectName("LogoLabel")
        brand_label.setStyleSheet("font-size: 18px; font-weight: 700; letter-spacing: 1px;")
        nav_layout.addWidget(brand_label)
        
        nav_layout.addStretch(1)
        
        # Action Buttons (Warm Mode style)
        self.btn_open_viewport = QPushButton("LAUNCH 3D VIEWPORT")
        self.btn_settings = QPushButton("SETTINGS")
        
        nav_layout.addWidget(self.btn_open_viewport)
        nav_layout.addWidget(self.btn_settings)
        
        self.main_layout.addWidget(self.nav_bar)
        
        # 2. Main Content Area (Console Mode Container)
        # We wrap the PipelineWorkspace inside a layout to give it margins so it looks
        # like it's "sitting" inside the Warm frame.
        self.workspace_container = QWidget()
        workspace_layout = QVBoxLayout(self.workspace_container)
        workspace_layout.setContentsMargins(10, 10, 10, 10)
        
        self.pipeline_workspace = PipelineWorkspace(self)
        workspace_layout.addWidget(self.pipeline_workspace)
        
        self.main_layout.addWidget(self.workspace_container, 1)

        # Connections
        self.btn_open_viewport.clicked.connect(self.open_viewport)
        
        self._viewport_window = None

    def apply_themes(self):
        # Get the global Warm Mode QSS for the main window framing
        warm_qss = self.exporter.generate_revisited_warm_qss()
        
        # Get the global Console Mode QSS for the workspace
        console_qss = self.exporter.generate_revisited_console_qss()
        
        # Apply the Warm Mode globally
        self.setStyleSheet(warm_qss)
        
        # Apply the Console Mode strictly to the Console container
        self.pipeline_workspace.setStyleSheet(console_qss)

    def open_viewport(self):
        if self._viewport_window is None:
            self._viewport_window = ViewportWindow(self, translate=_t)
            # The viewport window is a top-level window (Console mode aesthetics)
            self._viewport_window.setStyleSheet(self.exporter.generate_revisited_console_qss())
            
            # For the central widget of the Viewport, ensure it has the ConsoleShell ID
            if self._viewport_window.centralWidget():
                self._viewport_window.centralWidget().setObjectName("ConsoleShell")
        
        self._viewport_window.showMaximized()
        self._viewport_window.raise_()
        self._viewport_window.activateWindow()
