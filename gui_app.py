from __future__ import annotations
import json
import os
import subprocess
import sys
import shutil
import threading
from pathlib import Path
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
try:
    from qfluentwidgets import BodyLabel
    from qfluentwidgets import FluentIcon as FIF
    from qfluentwidgets import (
        ComboBox,
        FluentWindow,
        NavigationItemPosition,
        PrimaryPushButton,
        SubtitleLabel,
        TextEdit,
        Theme,
        setTheme,
        setThemeColor,
    )
except ImportError:
    from PySide6.QtWidgets import QTextEdit as TextEdit

    class _FallbackIcon:
        ACCEPT = None
        APPLICATION = None
        FOLDER = None
        HOME = None
        LINK = None
        PLAY = None
        PLAY_SOLID = None
        SAVE = None
        SETTING = None
        SYNC = None
        VIDEO = None

    class _FallbackNavigationItemPosition:
        BOTTOM = None

    class _FallbackTheme:
        LIGHT = None

    class BodyLabel(QLabel):
        pass

    class SubtitleLabel(QLabel):
        pass

    class PrimaryPushButton(QPushButton):
        def __init__(self, _icon=None, text: str = "", parent: QWidget | None = None):
            super().__init__(text, parent)

    class FluentWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self._tabs = QTabWidget(self)
            self.setCentralWidget(self._tabs)

        def addSubInterface(self, widget: QWidget, _icon, text: str, _position=None) -> None:
            self._tabs.addTab(widget, text)

    def setTheme(_theme) -> None:
        return None

    def setThemeColor(_color: object) -> None:
        return None

    from PySide6.QtWidgets import QComboBox as ComboBox  # noqa: E402

    FIF = _FallbackIcon()
    NavigationItemPosition = _FallbackNavigationItemPosition()
    Theme = _FallbackTheme()
from studio_core.pipeline import (
    DEFAULT_CAMERA_PROMPT,
    DEFAULT_SOURCE_VIDEO,
    JOBS_DIR,
    DigitalTwinStudioRunner,
    JobManifest,
    StageState,
    build_dependency_health,
    create_job_manifest,
    load_job_manifest,
    load_latest_job_manifest,
    next_incomplete_stage_key,
    stage_dependencies_complete,
)
from studio_core.integration import (
    VaultFlowsConnectionSettings,
    export_vaultflows_workflow,
    push_workflow_to_vaultwares,
    test_vaultwares_api,
)
from studio_core.viewer import open_live_viewer

ROOT = Path(__file__).resolve().parent
ICON_PATH = ROOT / "Brand" / "favicons" / "vaultwares-favicon-gold-filled-256.png"

sys.path.insert(0, str(ROOT / "vault-themes"))
try:
    from theme_manager import VaultTheme, VaultThemeManager  # noqa: E402
except ImportError as _exc:
    raise RuntimeError(
        "vault-themes submodule not found. Run: git submodule update --init vault-themes"
    ) from _exc


def _detect_os_theme() -> str:
    try:
        import winreg  # noqa: PLC0415
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "light" if value == 1 else "dark"
    except Exception:  # noqa: BLE001
        return "dark"


_tm = VaultThemeManager()
_active_theme: VaultTheme = _tm.get_theme_by_name(
    "Solarized Light Revisited" if _detect_os_theme() == "light" else "Golden Slate"
)


def build_stylesheet(theme: VaultTheme) -> str:
    return (
        f"QWidget {{ background: {theme.background}; color: {theme.text_primary}; }}"
        f" QFrame {{ background: {theme.surface}; }}"
    )


def card_style(theme: VaultTheme) -> str:
    return (
        f"QFrame {{ background: {theme.surface}; border: 1px solid {theme.border_subtle};"
        " border-radius: 8px; }}"
    )


def accent_card_style(theme: VaultTheme) -> str:
    return (
        f"QFrame {{ background: {theme.surface_elevated}; border: 1px solid {theme.accent};"
        " border-radius: 8px; }}"
    )


def preview_style(theme: VaultTheme) -> str:
    return (
        f"QLabel {{ background: {theme.surface_elevated}; color: {theme.text_secondary};"
        f" border: 1px dashed {theme.text_muted}; border-radius: 8px;"
        " min-height: 120px; padding: 12px; }}"
    )


def state_card_style(theme: VaultTheme, state: str) -> str:
    if state == StageState.COMPLETE.value:
        left_color = theme.success
    elif state == StageState.FAILED.value:
        left_color = theme.danger
    elif state == StageState.RUNNING.value:
        left_color = theme.accent_hover
    else:
        left_color = theme.border_subtle
    return (
        f"QFrame {{ background: {theme.surface}; border: 1px solid {theme.border_subtle};"
        f" border-left: 4px solid {left_color}; border-radius: 8px; }}"
    )
STATE_LABELS = {
    StageState.QUEUED.value: "Queued",
    StageState.RUNNING.value: "Running",
    StageState.NEEDS_INSTALL.value: "Needs Install",
    StageState.NEEDS_USER_INPUT.value: "Needs User Input",
    StageState.COMPLETE.value: "Complete",
    StageState.FAILED.value: "Failed",
}


class TaskSignals(QObject):
    log = Signal(str)
    manifest_changed = Signal(object)
    running_changed = Signal(bool)


def _open_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    subprocess.Popen(["xdg-open", str(path)])


class SettingsTab(QFrame):
    theme_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.setObjectName("Settings")
        self.setStyleSheet(card_style(_active_theme))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = SubtitleLabel("Settings", self)
        layout.addWidget(title)

        self.mode_label = BodyLabel("Execution mode: fallback-safe for local hardware.", self)
        self.mode_label.setWordWrap(True)
        layout.addWidget(self.mode_label)

        self.toggle_btn = PrimaryPushButton(FIF.SETTING, "Enable Strict Tool Mode", self)
        layout.addWidget(self.toggle_btn)

        self.refresh_btn = PrimaryPushButton(FIF.SYNC, "Refresh Dependency Health", self)
        layout.addWidget(self.refresh_btn)

        self.health_view = TextEdit(self)
        self.health_view.setReadOnly(True)
        layout.addWidget(self.health_view)

        # --- theme picker ---
        theme_row = QHBoxLayout()
        theme_label = BodyLabel("Theme:", self)
        theme_row.addWidget(theme_label)
        self.theme_combo = ComboBox(self)
        for t in _tm.get_themes():
            self.theme_combo.addItem(t.name)
        self.theme_combo.setCurrentText(_active_theme.name)
        theme_row.addWidget(self.theme_combo, 1)
        self.theme_swatch = QFrame(self)
        self.theme_swatch.setFixedSize(24, 24)
        self.theme_swatch.setStyleSheet(
            f"background: {_active_theme.accent}; border-radius: 4px;"
        )
        theme_row.addWidget(self.theme_swatch)
        layout.addLayout(theme_row)
        # --- end theme picker ---

        self.refresh_btn.clicked.connect(self.refresh_dependency_health)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.refresh_dependency_health()

    def set_mode(self, strict_mode: bool) -> None:
        if strict_mode:
            self.mode_label.setText("Execution mode: strict. Missing heavy tools fail the active stage.")
            self.toggle_btn.setText("Disable Strict Tool Mode")
        else:
            self.mode_label.setText("Execution mode: fallback-safe for local hardware.")
            self.toggle_btn.setText("Enable Strict Tool Mode")

    def refresh_dependency_health(self) -> None:
        lines: list[str] = []
        for row in build_dependency_health():
            lines.append(f"[{row['status'].upper()}] {row['kind']}: {row['name']} -> {row['detail']}")
        self.health_view.setPlainText("\n".join(lines))

    def _on_theme_changed(self, index: int) -> None:
        theme = _tm.get_theme(index)
        self.theme_swatch.setStyleSheet(
            f"background: {theme.accent}; border-radius: 4px;"
        )
        self.theme_changed.emit(theme)


class DashboardWidget(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.setObjectName("Dashboard")
        self.signals = TaskSignals()
        self.signals.log.connect(self._append_log)
        self.signals.manifest_changed.connect(self._on_manifest_changed)
        self.signals.running_changed.connect(self._set_running)

        self.strict_mode = False
        self.is_running = False
        self.show_finish_panel = False
        self.manifest = create_job_manifest(DEFAULT_SOURCE_VIDEO)
        self.selected_stage_key = self.manifest.current_stage_key
        self._normal_cards: list[QFrame] = []
        self._accent_cards: list[QFrame] = []

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        self.left_column = self._build_left_column()
        self.right_column = self._build_right_column()
        root_layout.addWidget(self.left_column, 0)
        root_layout.addWidget(self.right_column, 1)

        self._render_manifest()
        self._append_log(f"Studio initialized for {self.manifest.source_video}")

    def _build_left_column(self) -> QWidget:
        container = QWidget(self)
        container.setFixedWidth(320)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        job_card = QFrame(container)
        job_card.setStyleSheet(accent_card_style(_active_theme))
        self._accent_cards.append(job_card)
        job_layout = QVBoxLayout(job_card)
        job_layout.setContentsMargins(18, 18, 18, 18)
        self.job_title = SubtitleLabel("Current Job", job_card)
        self.job_meta = BodyLabel("", job_card)
        self.job_meta.setWordWrap(True)
        self.source_video_label = BodyLabel("", job_card)
        self.source_video_label.setWordWrap(True)
        self.pick_video_btn = PrimaryPushButton(FIF.FOLDER, "Choose Video", job_card)
        self.use_demo_btn = PrimaryPushButton(FIF.VIDEO, "Use Demo Video", job_card)
        self.open_latest_job_btn = PrimaryPushButton(FIF.SYNC, "Open Latest Job", job_card)
        self.open_manifest_btn = PrimaryPushButton(FIF.FOLDER, "Open Job Manifest", job_card)
        job_layout.addWidget(self.job_title)
        job_layout.addWidget(self.job_meta)
        job_layout.addWidget(self.source_video_label)
        job_layout.addWidget(self.pick_video_btn)
        job_layout.addWidget(self.use_demo_btn)
        job_layout.addWidget(self.open_latest_job_btn)
        job_layout.addWidget(self.open_manifest_btn)
        layout.addWidget(job_card)

        stages_card = QFrame(container)
        stages_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(stages_card)
        stages_layout = QVBoxLayout(stages_card)
        stages_layout.setContentsMargins(18, 18, 18, 18)
        stages_layout.addWidget(SubtitleLabel("Job Steps", stages_card))
        self.stage_list = QListWidget(stages_card)
        stages_layout.addWidget(self.stage_list)
        layout.addWidget(stages_card, 1)

        actions_card = QFrame(container)
        actions_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(actions_card)
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(18, 18, 18, 18)
        self.run_full_job_btn = PrimaryPushButton(FIF.PLAY_SOLID, "Run Full Job", actions_card)
        self.run_stage_btn = PrimaryPushButton(FIF.PLAY, "Run Selected Step", actions_card)
        self.open_job_folder_btn = PrimaryPushButton(FIF.FOLDER, "Open Job Folder", actions_card)
        actions_layout.addWidget(self.run_full_job_btn)
        actions_layout.addWidget(self.run_stage_btn)
        actions_layout.addWidget(self.open_job_folder_btn)
        layout.addWidget(actions_card)

        self.pick_video_btn.clicked.connect(self._pick_video)
        self.use_demo_btn.clicked.connect(self._use_demo_video)
        self.open_latest_job_btn.clicked.connect(self._open_latest_job)
        self.open_manifest_btn.clicked.connect(self._open_manifest)
        self.stage_list.currentItemChanged.connect(self._on_stage_selected)
        self.run_stage_btn.clicked.connect(self._run_selected_stage)
        self.run_full_job_btn.clicked.connect(self._run_full_job)
        self.open_job_folder_btn.clicked.connect(lambda: _open_path(Path(self.manifest.output_dir)))
        return container

    def _build_right_column(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self.state_card = QFrame(container)
        self.state_card.setStyleSheet(state_card_style(_active_theme, StageState.QUEUED.value))
        self._accent_cards.append(self.state_card)
        state_layout = QGridLayout(self.state_card)
        state_layout.setContentsMargins(18, 18, 18, 18)
        state_layout.setHorizontalSpacing(18)
        self.state_title = SubtitleLabel("Run State", self.state_card)
        self.state_message = BodyLabel("", self.state_card)
        self.state_message.setWordWrap(True)
        self.progress_label = BodyLabel("", self.state_card)
        self.progress_bar = QProgressBar(self.state_card)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        state_layout.addWidget(self.state_title, 0, 0)
        state_layout.addWidget(self.progress_label, 0, 1, 1, 1, Qt.AlignRight)
        state_layout.addWidget(self.state_message, 1, 0, 1, 2)
        state_layout.addWidget(self.progress_bar, 2, 0, 1, 2)
        layout.addWidget(self.state_card)

        self.viewer_stack = QStackedWidget(container)
        self.step_page = self._build_step_page()
        self.finish_page = self._build_finish_page()
        self.viewer_stack.addWidget(self.step_page)
        self.viewer_stack.addWidget(self.finish_page)
        layout.addWidget(self.viewer_stack, 1)

        return container

    def _build_step_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        summary_card = QFrame(page)
        summary_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(summary_card(_active_theme))
        self._normal_cards.append(summary_card(_active_theme))
        self._normal_cards.append(summary_card)
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(18, 18, 18, 18)
        self.step_title = SubtitleLabel("", summary_card)
        self.step_description = BodyLabel("", summary_card)
        self.step_description.setWordWrap(True)
        self.step_message = BodyLabel("", summary_card)
        self.step_message.setWordWrap(True)
        summary_layout.addWidget(self.step_title)
        summary_layout.addWidget(self.step_description)
        summary_layout.addWidget(self.step_message)
        layout.addWidget(summary_card)

        prompt_card = QFrame(page)
        prompt_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(prompt_card)
        prompt_layout = QVBoxLayout(prompt_card)
        prompt_layout.setContentsMargins(18, 18, 18, 18)
        prompt_layout.addWidget(SubtitleLabel("Prompt Camera Director", prompt_card))
        self.camera_prompt_edit = QLineEdit(prompt_card)
        self.camera_prompt_edit.setText(DEFAULT_CAMERA_PROMPT)
        self.camera_prompt_save_btn = PrimaryPushButton(FIF.SAVE, "Save Prompt", prompt_card)
        prompt_layout.addWidget(self.camera_prompt_edit)
        prompt_layout.addWidget(self.camera_prompt_save_btn)
        layout.addWidget(prompt_card)
        self.camera_prompt_card = prompt_card

        preview_card = QFrame(page)
        preview_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(preview_card)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.addWidget(SubtitleLabel("Stage Previews", preview_card))
        preview_grid = QHBoxLayout()
        self.preview_labels: list[QLabel] = []
        for _ in range(3):
            label = QLabel("Preview pending", preview_card)
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet(preview_style(_active_theme))
            self.preview_labels.append(label)
            preview_grid.addWidget(label)
        preview_layout.addLayout(preview_grid)
        layout.addWidget(preview_card)

        artifact_card = QFrame(page)
        artifact_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(artifact_card)
        artifact_layout = QVBoxLayout(artifact_card)
        artifact_layout.setContentsMargins(18, 18, 18, 18)
        artifact_layout.addWidget(SubtitleLabel("Artifacts", artifact_card))
        self.artifact_buttons_layout = QHBoxLayout()
        artifact_layout.addLayout(self.artifact_buttons_layout)
        layout.addWidget(artifact_card)

        log_card = QFrame(page)
        log_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(log_card)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(18, 18, 18, 18)
        log_layout.addWidget(SubtitleLabel("Run Log", log_card))
        self.log_view = TextEdit(log_card)
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        self.return_to_finish_btn = PrimaryPushButton(FIF.ACCEPT, "Return to Final Review", log_card)
        log_layout.addWidget(self.return_to_finish_btn)
        layout.addWidget(log_card, 1)

        self.camera_prompt_save_btn.clicked.connect(self._save_camera_prompt)
        self.return_to_finish_btn.clicked.connect(self._show_finish_panel)
        return page

    def _build_finish_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        finish_card = QFrame(page)
        finish_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(finish_card)
        finish_layout = QVBoxLayout(finish_card)
        finish_layout.setContentsMargins(18, 18, 18, 18)
        finish_layout.addWidget(SubtitleLabel("Final Review", finish_card))
        self.finish_summary = BodyLabel("", finish_card)
        self.finish_summary.setWordWrap(True)
        finish_layout.addWidget(self.finish_summary)

        buttons_row = QHBoxLayout()
        self.open_video_btn = PrimaryPushButton(FIF.VIDEO, "Open Walkthrough Video", finish_card)
        self.open_viewer_btn = PrimaryPushButton(FIF.APPLICATION, "Open Live 3D Viewer", finish_card)
        self.open_outputs_btn = PrimaryPushButton(FIF.FOLDER, "Open Output Folder", finish_card)
        buttons_row.addWidget(self.open_video_btn)
        buttons_row.addWidget(self.open_viewer_btn)
        buttons_row.addWidget(self.open_outputs_btn)
        finish_layout.addLayout(buttons_row)
        layout.addWidget(finish_card)

        integration_card = QFrame(page)
        integration_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(integration_card)
        integration_layout = QVBoxLayout(integration_card)
        integration_layout.setContentsMargins(18, 18, 18, 18)
        integration_layout.addWidget(SubtitleLabel("VaultWares Integration", integration_card))
        self.api_base_edit = QLineEdit(integration_card)
        self.api_base_edit.setText("https://localhost:8000")
        self.api_base_edit.setPlaceholderText("Vaultwares Pipelines API base URL")
        self.app_url_edit = QLineEdit(integration_card)
        self.app_url_edit.setText("https://localhost:5174")
        self.app_url_edit.setPlaceholderText("Vault Flows app URL")
        self.bearer_token_edit = QLineEdit(integration_card)
        self.bearer_token_edit.setPlaceholderText("Bearer token (optional)")
        self.bearer_token_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit = QLineEdit(integration_card)
        self.api_key_edit.setPlaceholderText("API key (optional)")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        integration_layout.addWidget(self.api_base_edit)
        integration_layout.addWidget(self.app_url_edit)
        integration_layout.addWidget(self.bearer_token_edit)
        integration_layout.addWidget(self.api_key_edit)

        integration_buttons = QHBoxLayout()
        self.test_api_btn = PrimaryPushButton(FIF.SYNC, "Test API", integration_card)
        self.export_workflow_btn = PrimaryPushButton(FIF.SAVE, "Export Workflow JSON", integration_card)
        self.push_workflow_btn = PrimaryPushButton(FIF.SYNC, "Push Workflow", integration_card)
        self.open_vault_flows_btn = PrimaryPushButton(FIF.LINK, "Open Vault Flows", integration_card)
        integration_buttons.addWidget(self.test_api_btn)
        integration_buttons.addWidget(self.export_workflow_btn)
        integration_buttons.addWidget(self.push_workflow_btn)
        integration_buttons.addWidget(self.open_vault_flows_btn)
        integration_layout.addLayout(integration_buttons)
        layout.addWidget(integration_card)

        detail_card = QFrame(page)
        detail_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(detail_card)
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(18, 18, 18, 18)
        detail_layout.addWidget(SubtitleLabel("Inspect Previous Steps", detail_card))
        note = BodyLabel(
            "The step rail remains live. Click any earlier step to reopen its viewer, previews, logs, and artifacts.",
            detail_card,
        )
        note.setWordWrap(True)
        detail_layout.addWidget(note)
        layout.addWidget(detail_card)

        self.open_video_btn.clicked.connect(self._open_walkthrough_video)
        self.open_viewer_btn.clicked.connect(self._open_live_viewer)
        self.open_outputs_btn.clicked.connect(lambda: _open_path(Path(self.manifest.output_dir)))
        self.test_api_btn.clicked.connect(self._test_live_api)
        self.export_workflow_btn.clicked.connect(self._export_workflow_package)
        self.push_workflow_btn.clicked.connect(self._push_workflow_package)
        self.open_vault_flows_btn.clicked.connect(self._open_vault_flows)
        return page

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)

    def _set_running(self, running: bool) -> None:
        self.is_running = running
        self.pick_video_btn.setEnabled(not running)
        self.use_demo_btn.setEnabled(not running)
        self.open_latest_job_btn.setEnabled(not running)
        self.open_manifest_btn.setEnabled(not running)
        self.stage_list.setEnabled(not running)
        self._sync_action_state()

    def _pick_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input video",
            str(ROOT),
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*.*)",
        )
        if file_path:
            self._reset_job(Path(file_path))

    def _use_demo_video(self) -> None:
        self._reset_job(DEFAULT_SOURCE_VIDEO)

    def _open_latest_job(self) -> None:
        manifest = load_latest_job_manifest()
        if manifest is None:
            self._append_log(f"No saved jobs found under {JOBS_DIR}.")
            return

        self._load_existing_job(manifest)

    def _open_manifest(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open job manifest",
            str(JOBS_DIR),
            "Job Manifests (manifest.json);;JSON Files (*.json);;All Files (*.*)",
        )
        if not file_path:
            return

        self._load_existing_job(load_job_manifest(file_path))

    def _load_existing_job(self, manifest: JobManifest) -> None:
        if self.is_running:
            return
        self.manifest = manifest
        self.selected_stage_key = next_incomplete_stage_key(manifest) or manifest.current_stage_key
        self.show_finish_panel = manifest.state == StageState.COMPLETE.value
        self.camera_prompt_edit.setText(str(manifest.metadata.get("cameraPrompt", DEFAULT_CAMERA_PROMPT)))
        self.log_view.clear()
        self._append_log(f"Opened job {manifest.job_id}")
        self._render_manifest()

    def _reset_job(self, source_video: Path) -> None:
        if self.is_running:
            return
        self.manifest = create_job_manifest(source_video=source_video, camera_prompt=self.camera_prompt_edit.text() or DEFAULT_CAMERA_PROMPT)
        self.selected_stage_key = self.manifest.current_stage_key
        self.show_finish_panel = False
        self.log_view.clear()
        self._append_log(f"Created new job {self.manifest.job_id} for {source_video}")
        self._render_manifest()

    def _save_camera_prompt(self) -> None:
        self.manifest.metadata["cameraPrompt"] = self.camera_prompt_edit.text().strip() or DEFAULT_CAMERA_PROMPT
        self._append_log("Updated camera prompt for this job.")
        manifest_path = Path(self.manifest.output_dir) / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest.to_dict(), indent=2), encoding="utf-8")

    def _on_stage_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        self.selected_stage_key = current.data(Qt.UserRole)
        self.show_finish_panel = False
        self._render_manifest()

    def _run_selected_stage(self) -> None:
        self._start_worker(run_full_job=False)

    def _run_full_job(self) -> None:
        self._start_worker(run_full_job=True)

    def _start_worker(self, run_full_job: bool) -> None:
        if self.is_running:
            return
        self._save_camera_prompt()
        self.signals.running_changed.emit(True)

        def worker() -> None:
            try:
                runner = DigitalTwinStudioRunner(self.manifest, self.signals.log.emit, strict_mode=self.strict_mode)
                if run_full_job:
                    result = runner.run_remaining()
                else:
                    result = runner.run_stage(self.selected_stage_key)
            except Exception as exc:  # noqa: BLE001
                self.signals.log.emit(f"[ERROR] {exc}")
                self.signals.manifest_changed.emit(self.manifest)
            else:
                self.signals.manifest_changed.emit(result)
            finally:
                self.signals.running_changed.emit(False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_manifest_changed(self, manifest: JobManifest) -> None:
        self.manifest = manifest
        if manifest.state == StageState.COMPLETE.value:
            self.show_finish_panel = True
        self._render_manifest()

    def _render_manifest(self) -> None:
        self.job_title.setText(f"Current Job: {self.manifest.job_id}")
        self.job_meta.setText(
            f"Profile: {self.manifest.execution_profile}\nMode: {self.manifest.mode}\nArtifacts: USD, cameras, MP4"
        )
        self.source_video_label.setText(f"Source Video: {self.manifest.source_video}")

        self.stage_list.blockSignals(True)
        self.stage_list.clear()
        for index, stage in enumerate(self.manifest.stages, start=1):
            label = f"{index}. {stage.title}  [{STATE_LABELS.get(stage.state, stage.state)}]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, stage.key)
            self.stage_list.addItem(item)
            if stage.key == self.selected_stage_key:
                self.stage_list.setCurrentItem(item)
        self.stage_list.blockSignals(False)

        completed = len([stage for stage in self.manifest.stages if stage.state == StageState.COMPLETE.value])
        total = len(self.manifest.stages)
        progress = int((completed / total) * 100) if total else 0
        current_stage = next(stage for stage in self.manifest.stages if stage.key == self.manifest.current_stage_key)
        self.state_title.setText(f"Run State: {STATE_LABELS.get(self.manifest.state, self.manifest.state)}")
        self.state_message.setText(current_stage.message or "Ready to execute the selected stage.")
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Step {completed if completed < total else total} of {total}")
        self.state_card.setStyleSheet(state_card_style(_active_theme, self.manifest.state))

        self.finish_summary.setText(
            "The digital twin job completed. Open the final walkthrough video, launch the optional live 3D viewer, or inspect any previous step from the rail."
        )

        self._render_selected_stage()
        self._sync_action_state()

    def _render_selected_stage(self) -> None:
        if self.manifest.state == StageState.COMPLETE.value and self.show_finish_panel:
            self.viewer_stack.setCurrentWidget(self.finish_page)
            return

        stage = next(stage for stage in self.manifest.stages if stage.key == self.selected_stage_key)
        self.viewer_stack.setCurrentWidget(self.step_page)
        self.step_title.setText(stage.title)
        self.step_description.setText(stage.description)
        if not stage_dependencies_complete(self.manifest, stage.key):
            self.step_message.setText("Complete earlier stages first, or use Run Full Job.")
        else:
            self.step_message.setText(stage.message or "This stage has not started yet.")

        show_prompt = stage.key == "usd_cameras"
        self.camera_prompt_card.setVisible(show_prompt)
        self.return_to_finish_btn.setVisible(self.manifest.state == StageState.COMPLETE.value)

        image_artifacts = [artifact for artifact in stage.artifacts if artifact.kind == "image"]
        for label, artifact in zip(self.preview_labels, image_artifacts[:3], strict=False):
            pixmap = QPixmap(artifact.path)
            if not pixmap.isNull():
                label.setPixmap(pixmap.scaled(260, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                label.setText(Path(artifact.path).name)
        for label in self.preview_labels[len(image_artifacts[:3]):]:
            label.setPixmap(QPixmap())
            label.setText("Preview pending")

        while self.artifact_buttons_layout.count():
            item = self.artifact_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for artifact in stage.artifacts[:6]:
            button = QPushButton(artifact.label, self)
            button.clicked.connect(lambda _checked=False, artifact_path=artifact.path: _open_path(Path(artifact_path)))
            self.artifact_buttons_layout.addWidget(button)

    def _sync_action_state(self) -> None:
        has_remaining_work = next_incomplete_stage_key(self.manifest) is not None
        selected_stage = next(stage for stage in self.manifest.stages if stage.key == self.selected_stage_key)
        selected_ready = stage_dependencies_complete(self.manifest, selected_stage.key)
        self.run_full_job_btn.setEnabled((not self.is_running) and has_remaining_work)
        self.run_stage_btn.setEnabled((not self.is_running) and selected_ready)

    def _show_finish_panel(self) -> None:
        self.show_finish_panel = True
        self._render_manifest()

    def _integration_settings(self) -> VaultFlowsConnectionSettings:
        return VaultFlowsConnectionSettings(
            api_base=self.api_base_edit.text().strip(),
            app_url=self.app_url_edit.text().strip(),
            bearer_token=self.bearer_token_edit.text(),
            api_key=self.api_key_edit.text(),
        )

    def _open_walkthrough_video(self) -> None:
        if not self.manifest.walkthrough_video:
            self._append_log("No walkthrough video has been generated yet.")
            return
        _open_path(Path(self.manifest.walkthrough_video))

    def _open_live_viewer(self) -> None:
        point_cloud = Path(self.manifest.output_dir) / "reconstruction" / "cloud.ply"

        def worker() -> None:
            success, message = open_live_viewer(point_cloud)
            self.signals.log.emit(message if success else f"[ERROR] {message}")

        threading.Thread(target=worker, daemon=True).start()

    def _test_live_api(self) -> None:
        settings = self._integration_settings()

        def worker() -> None:
            try:
                result = test_vaultwares_api(settings)
                self.signals.log.emit(json.dumps(result, indent=2))
            except Exception as exc:  # noqa: BLE001
                self.signals.log.emit(f"[ERROR] {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _export_workflow_package(self) -> None:
        output_path = Path(self.manifest.output_dir) / "vault_flows_workflow.json"
        export_vaultflows_workflow(self.manifest, output_path)
        self._append_log(f"Exported Vault Flows workflow package: {output_path}")

    def _push_workflow_package(self) -> None:
        settings = self._integration_settings()

        def worker() -> None:
            try:
                result = push_workflow_to_vaultwares(settings, self.manifest)
                self.signals.log.emit(json.dumps(result, indent=2))
            except Exception as exc:  # noqa: BLE001
                self.signals.log.emit(f"[ERROR] {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _open_vault_flows(self) -> None:
        settings = self._integration_settings()
        if not settings.app_url:
            self._append_log("Vault Flows URL is empty.")
            return

        if settings.app_url.startswith(("http://", "https://")):
            if os.name == "nt":
                os.startfile(settings.app_url)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", settings.app_url])
            return

        _open_path(Path(settings.app_url))

    def set_strict_mode(self, strict_mode: bool) -> None:
        self.strict_mode = strict_mode
        self._append_log(f"Strict mode set to {strict_mode}")

    def refresh_cards(self, theme: VaultTheme) -> None:
        for card in self._normal_cards:
            card.setStyleSheet(card_style(theme))
        for card in self._accent_cards:
            card.setStyleSheet(accent_card_style(theme))
        self.state_card.setStyleSheet(state_card_style(theme, self.manifest.state))
        for label in self.preview_labels:
            label.setStyleSheet(preview_style(theme))


class Window(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Twin Studio")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.dashboard = DashboardWidget(self)
        self.settings = SettingsTab(self)
        self.settings.toggle_btn.clicked.connect(self._toggle_strict_mode)
        self.settings.theme_changed.connect(self._apply_theme)

        self.addSubInterface(self.dashboard, FIF.HOME, "Studio")
        self.addSubInterface(self.settings, FIF.SETTING, "Settings", NavigationItemPosition.BOTTOM)
        self.resize(1320, 860)

    def _toggle_strict_mode(self) -> None:
        strict_mode = not self.dashboard.strict_mode
        self.dashboard.set_strict_mode(strict_mode)
        self.settings.set_mode(strict_mode)

    def _apply_theme(self, theme: VaultTheme) -> None:
        global _active_theme
        _active_theme = theme
        qt_theme = Theme.DARK if theme.mode == "dark" else Theme.LIGHT
        setTheme(qt_theme)
        setThemeColor(QColor(theme.accent))
        QApplication.instance().setStyleSheet(build_stylesheet(theme))
        self.dashboard.refresh_cards(theme)
        self.settings.setStyleSheet(card_style(theme))


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")

    app = QApplication(sys.argv)
    font = QFont("Segoe UI Semilight", 10)
    app.setFont(font)
    qt_theme = Theme.DARK if _active_theme.mode == "dark" else Theme.LIGHT
    setTheme(qt_theme)
    setThemeColor(QColor(_active_theme.accent))
    app.setStyleSheet(build_stylesheet(_active_theme))
    window = Window()
    window.show()
    sys.exit(app.exec())
