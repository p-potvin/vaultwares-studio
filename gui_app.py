from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from PySide6.QtCore import QObject, Qt, QSettings, Signal
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
try:
    from qfluentwidgets import BodyLabel  # type: ignore[assignment]
    from qfluentwidgets import FluentIcon as FIF
    from qfluentwidgets import ComboBox
    from qfluentwidgets import FluentWindow  # type: ignore[assignment]
    from qfluentwidgets import NavigationItemPosition
    from qfluentwidgets import PrimaryPushButton  # type: ignore[assignment]
    from qfluentwidgets import SubtitleLabel  # type: ignore[assignment]
    from qfluentwidgets import TextEdit
    from qfluentwidgets import Theme
    from qfluentwidgets import setTheme  # type: ignore[assignment]
    from qfluentwidgets import setThemeColor  # type: ignore[assignment]
    from qfluentwidgets import InfoBar, InfoBarPosition
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
        DARK = None

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
from vaultwares_studio.pipeline import (
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
from vaultwares_studio.integration import (
    VaultFlowsConnectionSettings,
    export_vaultflows_workflow,
    push_workflow_to_vaultwares,
    test_vaultwares_api,
)
from vaultwares_studio.presets import DEFAULT_PRESET_KEY, PRESETS, get_preset
from vaultwares_studio.runners import (
    FLAVOR_RATES_USD_PER_HOUR,
    HfJobsConfig,
    HfJobsStageRunner,
    estimate_cost,
    get_hf_token,
    run_echo_smoke_test,
    set_hf_token,
)
from vaultwares_studio.viewer import open_live_viewer
from gui.viewport import ViewportTab, register_viewer_scheme

ROOT = Path(__file__).resolve().parent
ICON_PATH = ROOT / "Brand" / "favicons" / "vaultwares-favicon-gold-filled-256.png"

sys.path.insert(0, str(ROOT / "vaultwares-themes"))
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


_settings = QSettings("VaultWares", "VaultwaresStudio")
_tm = VaultThemeManager()
_saved_theme_name = _settings.value("theme", None)
if _saved_theme_name:
    _active_theme_result = _tm.get_theme_by_name(str(_saved_theme_name))
    _active_theme = _active_theme_result if _active_theme_result else _tm.get_theme("Golden Slate")
else:
    _active_theme_result = _tm.get_theme_by_name(
        "Solarized Light Revisited" if _detect_os_theme() == "light" else "Golden Slate"
    )
    _active_theme = _active_theme_result if _active_theme_result else _tm.get_theme("Golden Slate")


def build_stylesheet(theme: VaultTheme) -> str:  # noqa: PLR0914
    bg = theme.background
    surface = theme.surface
    surface_alt = theme.surface_alt
    surface_el = theme.surface_elevated
    text = theme.text_primary
    text_sec = theme.text_secondary
    text_mut = theme.text_muted
    text_inv = theme.text_inverse
    accent = theme.accent
    accent_m = theme.accent_muted
    border = theme.border
    muted = theme.muted
    return f"""
QWidget {{
    background: {bg};
    color: {text};
    font-family: "Segoe UI Semilight", "Segoe UI", "Inter", system-ui, sans-serif;
    font-size: 10pt;
}}
QMainWindow, QDialog {{ background: {bg}; }}
QFrame {{
    background: {surface};
    border: none;
    border-radius: 0px;
}}
/* ── Scroll areas ──────────────────────────── */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget, QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
/* ── Scroll bars ───────────────────────────── */
QScrollBar:vertical {{
    background: {surface_alt};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {muted};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {accent}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none; height: 0; width: 0;
}}
QScrollBar:horizontal {{
    background: {surface_alt};
    height: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {muted};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {accent}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none; height: 0; width: 0;
}}
/* ── Push buttons ──────────────────────────── */
QPushButton {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 10pt;
    min-height: 32px;
}}
QPushButton:hover {{
    background: {accent};
    color: {text_inv};
    border-color: {accent};
}}
QPushButton:pressed {{
    background: {accent_m};
    color: {text_inv};
}}
QPushButton:disabled {{
    background: {surface_alt};
    color: {muted};
    border-color: {border};
}}
/* ── Line edits ────────────────────────────── */
QLineEdit {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 10pt;
    min-height: 36px;
    selection-background-color: {accent};
    selection-color: {text_inv};
}}
QLineEdit:focus {{ border: 1px solid {accent}; }}
/* ── Text edits (log view) ─────────────────── */
QTextEdit {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px;
    font-family: "Consolas", "Cascadia Code", "Courier New", monospace;
    font-size: 9pt;
    selection-background-color: {accent};
    selection-color: {text_inv};
}}
QTextEdit:focus {{ border: 1px solid {accent}; }}
/* ── List widget ───────────────────────────── */
QListWidget {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-radius: 4px;
    color: {text};
}}
QListWidget::item:hover {{ background: {surface_alt}; }}
QListWidget::item:selected {{ background: {accent}; color: {text_inv}; }}
/* ── Combo box ─────────────────────────────── */
QComboBox {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 32px;
    font-size: 10pt;
}}
QComboBox:hover, QComboBox:focus {{ border: 1px solid {accent}; }}
QComboBox::drop-down {{
    border: none;
    width: 24px;
    border-left: 1px solid {border};
}}
QComboBox QAbstractItemView {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    selection-background-color: {accent};
    selection-color: {text_inv};
    padding: 4px;
    outline: none;
}}
/* ── Progress bar ──────────────────────────── */
QProgressBar {{
    background: {surface_alt};
    border: none;
    border-radius: 4px;
    max-height: 6px;
    min-height: 6px;
}}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
/* ── Labels ────────────────────────────────── */
QLabel {{ background: transparent; color: {text}; }}
/* ── Splitter ──────────────────────────────── */
QSplitter::handle {{ background: {border}; }}
QSplitter::handle:horizontal {{ width: 4px; margin: 2px 0; }}
QSplitter::handle:vertical   {{ height: 4px; margin: 0 2px; }}
QSplitter::handle:hover      {{ background: {accent_m}; }}
/* ── Tab widget (fallback) ─────────────────── */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 8px;
    background: {surface};
}}
QTabBar::tab {{
    background: {surface_alt};
    color: {text_sec};
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid {border};
    margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {surface}; color: {text}; border-bottom-color: {surface}; }}
QTabBar::tab:hover    {{ background: {surface_el}; color: {text}; }}
"""


def card_style(theme: VaultTheme) -> str:
    return (
        f"QFrame {{ background: {theme.surface}; border: 1px solid {theme.border};"
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
        left_color = theme.error
    elif state == StageState.RUNNING.value:
        left_color = theme.accent_muted
    else:
        left_color = theme.border
    return (
        f"QFrame {{ background: {theme.surface}; border: 1px solid {theme.border};"
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


# ── Localisation (EN / QC French) ────────────────────────────────────────────
_active_lang: str = str(_settings.value("general/lang", "EN"))
_STRINGS: dict[str, dict[str, str]] = {
    "app_title":             {"EN": "Digital Twin Studio",                    "QC": "Studio de jumeau numérique"},
    "tab_studio":            {"EN": "Studio",                                  "QC": "Studio"},
    "tab_settings":          {"EN": "Settings",                                "QC": "Paramètres"},
    "current_job":           {"EN": "Current Job",                             "QC": "Travail actuel"},
    "source_video":          {"EN": "Source Video",                            "QC": "Vidéo source"},
    "choose_video":          {"EN": "Choose Video",                            "QC": "Choisir une vidéo"},
    "use_demo_video":        {"EN": "Use Demo Video",                          "QC": "Vidéo de démonstration"},
    "open_latest_job":       {"EN": "Open Latest Job",                         "QC": "Dernier travail"},
    "open_manifest":         {"EN": "Open Job Manifest",                       "QC": "Ouvrir le manifeste"},
    "job_steps":             {"EN": "Job Steps",                               "QC": "Étapes du travail"},
    "run_full_job":          {"EN": "Run Full Job",                            "QC": "Exécuter tout le travail"},
    "run_selected_step":     {"EN": "Run Selected Step",                       "QC": "Exécuter l'étape"},
    "open_job_folder":       {"EN": "Open Job Folder",                         "QC": "Ouvrir le dossier"},
    "run_state":             {"EN": "Run State",                               "QC": "État d'exécution"},
    "prompt_camera":         {"EN": "Prompt Camera Director",                  "QC": "Inviter le directeur de caméra"},
    "save_prompt":           {"EN": "Save Prompt",                             "QC": "Enregistrer l'invite"},
    "stage_previews":        {"EN": "Stage Previews",                          "QC": "Aperçus de l'étape"},
    "artifacts":             {"EN": "Artifacts",                               "QC": "Artéfacts"},
    "run_log":               {"EN": "Run Log",                                 "QC": "Journal d'exécution"},
    "clear_log":             {"EN": "Clear",                                   "QC": "Effacer"},
    "return_to_finish":      {"EN": "Return to Final Review",                  "QC": "Retour à la révision finale"},
    "final_review":          {"EN": "Final Review",                            "QC": "Révision finale"},
    "open_walkthrough":      {"EN": "Open Walkthrough Video",                  "QC": "Vidéo de visite guidée"},
    "open_3d_viewer":        {"EN": "Open Live 3D Viewer",                     "QC": "Visualiseur 3D en direct"},
    "open_output_folder":    {"EN": "Open Output Folder",                      "QC": "Dossier de sortie"},
    "integration":           {"EN": "VaultWares Integration",                  "QC": "Intégration VaultWares"},
    "api_url_label":         {"EN": "API Base URL",                            "QC": "URL de base de l'API"},
    "app_url_label":         {"EN": "App URL (Vault Flows)",                   "QC": "URL de l'application"},
    "bearer_label":          {"EN": "Bearer Token (optional)",                 "QC": "Jeton porteur (optionnel)"},
    "api_key_label":         {"EN": "API Key (optional)",                      "QC": "Clé API (optionnelle)"},
    "test_api":              {"EN": "Test API",                                "QC": "Tester l'API"},
    "export_workflow":       {"EN": "Export Workflow JSON",                    "QC": "Exporter le flux JSON"},
    "push_workflow":         {"EN": "Push Workflow",                           "QC": "Pousser le flux de travail"},
    "open_vault_flows":      {"EN": "Open Vault Flows",                        "QC": "Ouvrir Vault Flows"},
    "inspect_steps":         {"EN": "Inspect Previous Steps",                  "QC": "Inspecter les étapes précédentes"},
    "inspect_note":          {"EN": "The step rail remains live. Click any earlier step to reopen its viewer, previews, logs, and artifacts.",
                              "QC": "Le rail d'étapes reste actif. Cliquez sur une étape précédente pour rouvrir son visualiseur, ses aperçus, ses journaux et ses artéfacts."},
    "execution_mode_safe":   {"EN": "Execution mode: fallback-safe for local hardware.",
                              "QC": "Mode d'exécution : sécurisé pour le matériel local."},
    "execution_mode_strict": {"EN": "Execution mode: strict. Missing heavy tools fail the active stage.",
                              "QC": "Mode strict : les outils manquants font échouer l'étape active."},
    "enable_strict":         {"EN": "Enable Strict Tool Mode",                 "QC": "Activer le mode strict"},
    "disable_strict":        {"EN": "Disable Strict Tool Mode",                "QC": "Désactiver le mode strict"},
    "refresh_health":        {"EN": "Refresh Dependency Health",               "QC": "Vérifier les dépendances"},
    "health_title":          {"EN": "Dependency Health",                       "QC": "État des dépendances"},
    "theme_label":           {"EN": "Theme",                                   "QC": "Thème"},
    "preview_pending":       {"EN": "Preview pending",                         "QC": "Aperçu en attente"},
    "finish_summary":        {"EN": "The digital twin job completed. Open the final walkthrough video, launch the optional live 3D viewer, or inspect any previous step from the rail.",
                              "QC": "Le travail de jumeau numérique est terminé. Ouvrez la vidéo de visite finale, lancez le visualiseur 3D optionnel, ou inspectez une étape précédente."},
    "ready_to_execute":      {"EN": "Ready to execute the selected stage.",
                              "QC": "Prêt à exécuter l'étape sélectionnée."},
    "complete_earlier":      {"EN": "Complete earlier stages first, or use Run Full Job.",
                              "QC": "Terminez d'abord les étapes précédentes, ou utilisez Exécuter tout le travail."},
    "lang_switch_label":     {"EN": "FR",                                      "QC": "EN"},
    "no_video_yet":          {"EN": "No walkthrough video has been generated yet.",
                              "QC": "Aucune vidéo de visite n'a encore été générée."},
    "strict_off":            {"EN": "Strict mode: OFF",                        "QC": "Mode strict : OFF"},
    "strict_on":             {"EN": "Strict mode: ON",                         "QC": "Mode strict : ON"},
    "remote_title":          {"EN": "Remote Compute (Hugging Face Jobs)",      "QC": "Calcul à distance (Hugging Face Jobs)"},
    "hf_token_label":        {"EN": "HF Token (stored in OS keyring)",         "QC": "Jeton HF (stocké dans le trousseau)"},
    "hf_repo_label":         {"EN": "Artifact dataset repo (blank = auto)",    "QC": "Dépôt d'artéfacts (vide = auto)"},
    "hf_flavor_label":       {"EN": "Default GPU flavor",                      "QC": "Type de GPU par défaut"},
    "save_remote":           {"EN": "Save Remote Settings",                    "QC": "Enregistrer les paramètres"},
    "test_remote":           {"EN": "Run Echo Test Job (cpu-basic)",           "QC": "Tester avec un travail écho (cpu-basic)"},
    "remote_cost_title":     {"EN": "Confirm paid remote job",                 "QC": "Confirmer le travail payant"},
    "remote_cost_body":      {"EN": "This will launch a paid Hugging Face Job:\n{estimate}\n\nProceed?",
                              "QC": "Ceci lancera un travail Hugging Face payant :\n{estimate}\n\nContinuer ?"},
    "remote_saved":          {"EN": "Remote settings saved.",                  "QC": "Paramètres enregistrés."},
    "remote_no_token":       {"EN": "No HF token configured. Paste your token and save first.",
                              "QC": "Aucun jeton HF configuré. Collez votre jeton et enregistrez d'abord."},
    "preset_label":          {"EN": "Quality preset",                            "QC": "Préréglage de qualité"},
    "tab_viewport":          {"EN": "Viewport",                                  "QC": "Vue 3D"},
    "viewport_reload":       {"EN": "Reload Scene",                              "QC": "Recharger la scène"},
    "viewport_capture":      {"EN": "Capture Camera",                            "QC": "Capturer la caméra"},
    "viewport_loading":      {"EN": "Loading reconstruction…",                   "QC": "Chargement de la reconstruction…"},
    "viewport_no_scene":     {"EN": "No reconstruction yet — run a job, then reload.",
                              "QC": "Aucune reconstruction — exécutez un travail, puis rechargez."},
    "viewport_no_webengine": {"EN": "QtWebEngine is unavailable on this system. Use the Open Live 3D Viewer button instead.",
                              "QC": "QtWebEngine n'est pas disponible. Utilisez plutôt le visualiseur 3D en direct."},
    "viewport_captured":     {"EN": "Camera captured ({count} total) — saved to captured_cameras.json.",
                              "QC": "Caméra capturée ({count} au total) — enregistrée dans captured_cameras.json."},
    "remote_declined":       {"EN": "Remote reconstruction declined; using the local quick path.",
                              "QC": "Reconstruction à distance refusée; utilisation du chemin local rapide."},
}


def _t(key: str) -> str:
    """Return the localised UI string for the currently active language."""
    entry = _STRINGS.get(key, {})
    return entry.get(_active_lang, entry.get("EN", key))



class TaskSignals(QObject):
    log = Signal(str)
    manifest_changed = Signal(object)
    running_changed = Signal(bool)
    api_test_finished = Signal(bool, str)


def _open_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    subprocess.Popen(["xdg-open", str(path)])


class SettingsTab(QFrame):
    theme_changed = Signal(object)
    remote_log = Signal(str)
    remote_test_done = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.setObjectName("Settings")
        self.setStyleSheet(card_style(_active_theme))
        self._strict_mode: bool = str(_settings.value("general/strict_mode", "false")).lower() == "true"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        self.settings_title = SubtitleLabel(_t("tab_settings"), inner)
        layout.addWidget(self.settings_title)

        self.mode_label = BodyLabel("Execution mode: fallback-safe for local hardware.", inner)
        self.mode_label.setWordWrap(True)
        layout.addWidget(self.mode_label)

        self.toggle_btn = PrimaryPushButton(FIF.SETTING, _t("disable_strict") if self._strict_mode else _t("enable_strict"), inner)
        layout.addWidget(self.toggle_btn)

        self.refresh_btn = PrimaryPushButton(FIF.SYNC, _t("refresh_health"), inner)
        layout.addWidget(self.refresh_btn)

        self.health_title_label = SubtitleLabel(_t("health_title"), inner)
        layout.addWidget(self.health_title_label)

        self.health_view = TextEdit(inner)
        self.health_view.setReadOnly(True)
        self.health_view.setMinimumHeight(180)
        layout.addWidget(self.health_view)

        # --- theme picker ---
        theme_row = QHBoxLayout()
        self.theme_label_widget = BodyLabel(_t("theme_label"), inner)
        theme_row.addWidget(self.theme_label_widget)
        self.theme_combo = ComboBox(inner)
        for t in _tm.get_themes():
            self.theme_combo.addItem(t.name)
        self.theme_combo.setCurrentText(_active_theme.name)
        theme_row.addWidget(self.theme_combo, 1)
        self.theme_swatch = QFrame(inner)
        self.theme_swatch.setFixedSize(24, 24)
        self.theme_swatch.setStyleSheet(
            f"background: {_active_theme.accent}; border-radius: 4px;"
        )
        theme_row.addWidget(self.theme_swatch)
        layout.addLayout(theme_row)
        # --- end theme picker ---

        # --- remote compute (Hugging Face Jobs) ---
        self._remote_config = HfJobsConfig.load()
        self.remote_title_label = SubtitleLabel(_t("remote_title"), inner)
        layout.addWidget(self.remote_title_label)

        self.hf_token_label = BodyLabel(_t("hf_token_label"), inner)
        layout.addWidget(self.hf_token_label)
        self.hf_token_edit = QLineEdit(inner)
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        if get_hf_token():
            self.hf_token_edit.setPlaceholderText("•••••••• (token already stored)")
        layout.addWidget(self.hf_token_edit)

        self.hf_repo_label = BodyLabel(_t("hf_repo_label"), inner)
        layout.addWidget(self.hf_repo_label)
        self.hf_repo_edit = QLineEdit(inner)
        self.hf_repo_edit.setText(self._remote_config.artifact_repo)
        layout.addWidget(self.hf_repo_edit)

        flavor_row = QHBoxLayout()
        self.hf_flavor_label = BodyLabel(_t("hf_flavor_label"), inner)
        flavor_row.addWidget(self.hf_flavor_label)
        self.hf_flavor_combo = ComboBox(inner)
        for flavor in FLAVOR_RATES_USD_PER_HOUR:
            self.hf_flavor_combo.addItem(flavor)
        self.hf_flavor_combo.setCurrentText(self._remote_config.default_flavor)
        flavor_row.addWidget(self.hf_flavor_combo, 1)
        layout.addLayout(flavor_row)

        self.save_remote_btn = PrimaryPushButton(FIF.SAVE, _t("save_remote"), inner)
        layout.addWidget(self.save_remote_btn)
        self.test_remote_btn = PrimaryPushButton(FIF.SYNC, _t("test_remote"), inner)
        layout.addWidget(self.test_remote_btn)

        self.remote_log_view = TextEdit(inner)
        self.remote_log_view.setReadOnly(True)
        self.remote_log_view.setMinimumHeight(120)
        layout.addWidget(self.remote_log_view)
        # --- end remote compute ---

        layout.addStretch(1)
        self.refresh_btn.clicked.connect(self.refresh_dependency_health)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.save_remote_btn.clicked.connect(self._save_remote_settings)
        self.test_remote_btn.clicked.connect(self._start_echo_test)
        self.remote_log.connect(self._append_remote_log)
        self.remote_test_done.connect(lambda: self.test_remote_btn.setEnabled(True))
        self.refresh_dependency_health()

    def _collect_remote_config(self) -> HfJobsConfig:
        self._remote_config.artifact_repo = self.hf_repo_edit.text().strip()
        self._remote_config.default_flavor = self.hf_flavor_combo.currentText()
        return self._remote_config

    def _save_remote_settings(self) -> None:
        token = self.hf_token_edit.text().strip()
        if token:
            set_hf_token(token)
            self.hf_token_edit.clear()
            self.hf_token_edit.setPlaceholderText("•••••••• (token already stored)")
        config = self._collect_remote_config()
        config.enabled = True
        config.save()
        self._append_remote_log(_t("remote_saved"))

    def _start_echo_test(self) -> None:
        if not get_hf_token():
            self._append_remote_log(_t("remote_no_token"))
            return
        config = self._collect_remote_config()
        estimate = estimate_cost("cpu-basic", 5)
        answer = QMessageBox.question(
            self,
            _t("remote_cost_title"),
            _t("remote_cost_body").format(estimate=estimate.summary()),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.test_remote_btn.setEnabled(False)
        # Consent was given via the dialog above (GUI thread); the runner's
        # confirm hook is satisfied with a pre-confirmed callback so no
        # dialog has to cross threads.
        threading.Thread(target=self._echo_worker, args=(config,), daemon=True).start()

    def _echo_worker(self, config: HfJobsConfig) -> None:
        try:
            summary = run_echo_smoke_test(config, confirm_cost=lambda _e: True, log=self.remote_log.emit)
            self.remote_log.emit(summary)
        except Exception as exc:  # noqa: BLE001 - surfaced to the log panel
            self.remote_log.emit(f"Echo test failed: {exc}")
        finally:
            self.remote_test_done.emit()

    def _append_remote_log(self, message: str) -> None:
        self.remote_log_view.append(message)

    def set_mode(self, strict_mode: bool) -> None:
        self._strict_mode = strict_mode
        if strict_mode:
            self.mode_label.setText("Execution mode: strict. Missing heavy tools fail the active stage.")
            self.toggle_btn.setText(_t("disable_strict"))
        else:
            self.mode_label.setText("Execution mode: fallback-safe for local hardware.")
            self.toggle_btn.setText(_t("enable_strict"))

    def refresh_dependency_health(self) -> None:
        lines: list[str] = []
        for row in build_dependency_health():
            lines.append(f"[{row['status'].upper()}] {row['kind']}: {row['name']} -> {row['detail']}")
        self.health_view.setPlainText("\n".join(lines))

    def _on_theme_changed(self, index: int) -> None:
        theme = _tm.get_theme(index=index)
        self.theme_swatch.setStyleSheet(
            f"background: {theme.accent}; border-radius: 4px;"
        )
        self.theme_changed.emit(theme)

    def retranslate(self) -> None:
        """Re-apply localised strings after a language switch."""
        self.settings_title.setText(_t("tab_settings"))
        self.toggle_btn.setText(_t("disable_strict") if self._strict_mode else _t("enable_strict"))
        self.refresh_btn.setText(_t("refresh_health"))
        self.health_title_label.setText(_t("health_title"))
        self.theme_label_widget.setText(_t("theme_label"))
        self.remote_title_label.setText(_t("remote_title"))
        self.hf_token_label.setText(_t("hf_token_label"))
        self.hf_repo_label.setText(_t("hf_repo_label"))
        self.hf_flavor_label.setText(_t("hf_flavor_label"))
        self.save_remote_btn.setText(_t("save_remote"))
        self.test_remote_btn.setText(_t("test_remote"))


class DashboardWidget(QFrame):
    theme_changed = Signal(object)
    lang_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.setObjectName("Dashboard")
        self.signals = TaskSignals()
        self.signals.log.connect(self._append_log)
        self.signals.manifest_changed.connect(self._on_manifest_changed)
        self.signals.running_changed.connect(self._set_running)
        self.signals.api_test_finished.connect(self._on_api_test_finished)

        self.strict_mode = str(_settings.value("general/strict_mode", "false")).lower() == "true"
        self.is_running = False
        self.show_finish_panel = False
        self.manifest = create_job_manifest(DEFAULT_SOURCE_VIDEO)
        self.selected_stage_key = self.manifest.current_stage_key
        self._normal_cards: list[QFrame] = []
        self._accent_cards: list[QFrame] = []

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._main_splitter.setChildrenCollapsible(False)
        self.left_column = self._build_left_column()
        self.right_column = self._build_right_column()
        self._main_splitter.addWidget(self.left_column)
        self._main_splitter.addWidget(self.right_column)
        self._main_splitter.setSizes([300, 900])
        root_layout.addWidget(self._main_splitter, 1)

        self._render_manifest()
        self._append_log(f"Studio initialized for {self.manifest.source_video}")

    def _build_left_column(self) -> QWidget:
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        job_card = QFrame(inner)
        job_card.setStyleSheet(accent_card_style(_active_theme))
        self._accent_cards.append(job_card)
        job_layout = QVBoxLayout(job_card)
        job_layout.setContentsMargins(18, 18, 18, 18)
        job_layout.setSpacing(8)
        self.job_title = SubtitleLabel(_t("current_job"), job_card)
        self.job_meta = BodyLabel("", job_card)
        self.job_meta.setWordWrap(True)
        self.source_video_label = BodyLabel("", job_card)
        self.source_video_label.setWordWrap(True)
        self.pick_video_btn = PrimaryPushButton(FIF.FOLDER, _t("choose_video"), job_card)
        self.use_demo_btn = PrimaryPushButton(FIF.VIDEO, _t("use_demo_video"), job_card)
        self.open_latest_job_btn = PrimaryPushButton(FIF.SYNC, _t("open_latest_job"), job_card)
        self.open_manifest_btn = PrimaryPushButton(FIF.FOLDER, _t("open_manifest"), job_card)
        job_layout.addWidget(self.job_title)
        job_layout.addWidget(self.job_meta)
        job_layout.addWidget(self.source_video_label)
        job_layout.addWidget(self.pick_video_btn)
        job_layout.addWidget(self.use_demo_btn)
        job_layout.addWidget(self.open_latest_job_btn)
        job_layout.addWidget(self.open_manifest_btn)
        layout.addWidget(job_card)

        stages_card = QFrame(inner)
        stages_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(stages_card)
        stages_layout = QVBoxLayout(stages_card)
        stages_layout.setContentsMargins(18, 18, 18, 18)
        self.stages_label = SubtitleLabel(_t("job_steps"), stages_card)
        stages_layout.addWidget(self.stages_label)
        self.stage_list = QListWidget(stages_card)
        self.stage_list.setMinimumHeight(180)
        stages_layout.addWidget(self.stage_list)
        layout.addWidget(stages_card)

        actions_card = QFrame(inner)
        actions_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(actions_card)
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(18, 18, 18, 18)
        actions_layout.setSpacing(8)
        preset_row = QHBoxLayout()
        self.preset_label = BodyLabel(_t("preset_label"), actions_card)
        preset_row.addWidget(self.preset_label)
        self.preset_combo = ComboBox(actions_card)
        for preset in PRESETS.values():
            self.preset_combo.addItem(preset.label, userData=preset.key)
        default_index = list(PRESETS).index(DEFAULT_PRESET_KEY)
        self.preset_combo.setCurrentIndex(default_index)
        preset_row.addWidget(self.preset_combo, 1)
        actions_layout.addLayout(preset_row)
        self.run_full_job_btn = PrimaryPushButton(FIF.PLAY_SOLID, _t("run_full_job"), actions_card)
        self.run_stage_btn = PrimaryPushButton(FIF.PLAY, _t("run_selected_step"), actions_card)
        self.open_job_folder_btn = PrimaryPushButton(FIF.FOLDER, _t("open_job_folder"), actions_card)
        actions_layout.addWidget(self.run_full_job_btn)
        actions_layout.addWidget(self.run_stage_btn)
        actions_layout.addWidget(self.open_job_folder_btn)
        layout.addWidget(actions_card)

        layout.addStretch(1)

        self.pick_video_btn.clicked.connect(self._pick_video)
        self.use_demo_btn.clicked.connect(self._use_demo_video)
        self.open_latest_job_btn.clicked.connect(self._open_latest_job)
        self.open_manifest_btn.clicked.connect(self._open_manifest)
        self.stage_list.currentItemChanged.connect(self._on_stage_selected)
        self.run_stage_btn.clicked.connect(self._run_selected_stage)
        self.run_full_job_btn.clicked.connect(self._run_full_job)
        self.open_job_folder_btn.clicked.connect(lambda: _open_path(Path(self.manifest.output_dir)))

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(240)
        scroll.setMaximumWidth(420)
        scroll.setWidget(inner)
        return scroll

    def _build_right_column(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        self.state_card = QFrame(container)
        self.state_card.setStyleSheet(state_card_style(_active_theme, StageState.QUEUED.value))
        self._accent_cards.append(self.state_card)
        state_layout = QGridLayout(self.state_card)
        state_layout.setContentsMargins(18, 18, 18, 18)
        state_layout.setHorizontalSpacing(18)
        self.state_title = SubtitleLabel(_t("run_state"), self.state_card)
        self.state_message = BodyLabel("", self.state_card)
        self.state_message.setWordWrap(True)
        self.progress_label = BodyLabel("", self.state_card)
        self.progress_bar = QProgressBar(self.state_card)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        state_layout.addWidget(self.state_title, 0, 0)
        state_layout.addWidget(self.progress_label, 0, 1, 1, 1, Qt.AlignmentFlag.AlignRight)
        state_layout.addWidget(self.state_message, 1, 0, 1, 2)
        state_layout.addWidget(self.progress_bar, 2, 0, 1, 2)
        layout.addWidget(self.state_card)

        self.right_tabs = QTabWidget(container)
        
        self.viewer_stack = QStackedWidget()
        self.step_page = self._build_step_page()
        self.finish_page = self._build_finish_page()
        self.viewer_stack.addWidget(self.step_page)
        self.viewer_stack.addWidget(self.finish_page)
        
        self.right_tabs.addTab(self.viewer_stack, _t("tab_studio"))
        
        self.integration_page = self._build_integration_page()
        self.right_tabs.addTab(self.integration_page, _t("integration"))
        
        log_panel = self._build_log_panel()
        self.right_tabs.addTab(log_panel, _t("run_log"))

        layout.addWidget(self.right_tabs, 1)
        return container

    def _build_step_page(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 8, 4)
        layout.setSpacing(14)

        summary_card = QFrame(page)
        summary_card.setStyleSheet(card_style(_active_theme))
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
        prompt_layout.setSpacing(10)
        self.prompt_label = SubtitleLabel(_t("prompt_camera"), prompt_card)
        prompt_layout.addWidget(self.prompt_label)
        self.camera_prompt_edit = QLineEdit(prompt_card)
        self.camera_prompt_edit.setText(str(_settings.value("general/camera_prompt", DEFAULT_CAMERA_PROMPT)))
        self.camera_prompt_edit.textChanged.connect(lambda txt: _settings.setValue("general/camera_prompt", txt))
        self.camera_prompt_save_btn = PrimaryPushButton(FIF.SAVE, _t("save_prompt"), prompt_card)
        prompt_layout.addWidget(self.camera_prompt_edit)
        prompt_layout.addWidget(self.camera_prompt_save_btn)
        layout.addWidget(prompt_card)
        self.camera_prompt_card = prompt_card

        preview_card = QFrame(page)
        preview_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(preview_card)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        self.previews_label = SubtitleLabel(_t("stage_previews"), preview_card)
        preview_layout.addWidget(self.previews_label)
        preview_grid = QHBoxLayout()
        self.preview_labels: list[QLabel] = []
        for _ in range(3):
            label = QLabel(_t("preview_pending"), preview_card)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self.artifacts_label = SubtitleLabel(_t("artifacts"), artifact_card)
        artifact_layout.addWidget(self.artifacts_label)
        self.artifact_buttons_layout = QHBoxLayout()
        artifact_layout.addLayout(self.artifact_buttons_layout)
        layout.addWidget(artifact_card)

        layout.addStretch(1)
        self.camera_prompt_save_btn.clicked.connect(self._save_camera_prompt)
        scroll.setWidget(page)
        return scroll

    def _build_finish_page(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 8, 4)
        layout.setSpacing(14)

        finish_card = QFrame(page)
        finish_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(finish_card)
        finish_layout = QVBoxLayout(finish_card)
        finish_layout.setContentsMargins(18, 18, 18, 18)
        finish_layout.setSpacing(12)
        self.finish_label = SubtitleLabel(_t("final_review"), finish_card)
        finish_layout.addWidget(self.finish_label)
        self.finish_summary = BodyLabel("", finish_card)
        self.finish_summary.setWordWrap(True)
        finish_layout.addWidget(self.finish_summary)

        buttons_row = QHBoxLayout()
        self.open_video_btn = PrimaryPushButton(FIF.VIDEO, _t("open_walkthrough"), finish_card)
        self.open_viewer_btn = PrimaryPushButton(FIF.APPLICATION, _t("open_3d_viewer"), finish_card)
        self.open_outputs_btn = PrimaryPushButton(FIF.FOLDER, _t("open_output_folder"), finish_card)
        buttons_row.addWidget(self.open_video_btn)
        buttons_row.addWidget(self.open_viewer_btn)
        buttons_row.addWidget(self.open_outputs_btn)
        finish_layout.addLayout(buttons_row)
        layout.addWidget(finish_card)

        detail_card = QFrame(page)
        detail_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(detail_card)
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(18, 18, 18, 18)
        self.inspect_label = SubtitleLabel(_t("inspect_steps"), detail_card)
        detail_layout.addWidget(self.inspect_label)
        note = BodyLabel(_t("inspect_note"), detail_card)
        note.setWordWrap(True)
        detail_layout.addWidget(note)
        layout.addWidget(detail_card)

        layout.addStretch(1)
        self.open_video_btn.clicked.connect(self._open_walkthrough_video)
        self.open_viewer_btn.clicked.connect(self._open_live_viewer)
        self.open_outputs_btn.clicked.connect(lambda: _open_path(Path(self.manifest.output_dir)))
        scroll.setWidget(page)
        return scroll

    def _build_integration_page(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 8, 4)
        layout.setSpacing(14)

        integration_card = QFrame(page)
        integration_card.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(integration_card)
        integration_layout = QVBoxLayout(integration_card)
        integration_layout.setContentsMargins(18, 18, 18, 18)
        integration_layout.setSpacing(8)
        self.integration_label = SubtitleLabel(_t("integration"), integration_card)
        integration_layout.addWidget(self.integration_label)
        self.api_url_label_widget = QLabel(_t("api_url_label"), integration_card)
        integration_layout.addWidget(self.api_url_label_widget)
        self.api_base_edit = QLineEdit(integration_card)
        self.api_base_edit.setText("https://localhost:8000")
        self.api_base_edit.setPlaceholderText("Vaultwares Pipelines API base URL")
        self.api_base_edit.setMinimumWidth(200)
        self.api_base_edit.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        integration_layout.addWidget(self.api_base_edit)
        self.app_url_label_widget = QLabel(_t("app_url_label"), integration_card)
        integration_layout.addWidget(self.app_url_label_widget)
        self.app_url_edit = QLineEdit(integration_card)
        self.app_url_edit.setText("https://localhost:5174")
        self.app_url_edit.setPlaceholderText("Vault Flows app URL")
        self.app_url_edit.setMinimumWidth(200)
        self.app_url_edit.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        integration_layout.addWidget(self.app_url_edit)
        self.bearer_label_widget = QLabel(_t("bearer_label"), integration_card)
        integration_layout.addWidget(self.bearer_label_widget)
        self.bearer_token_edit = QLineEdit(integration_card)
        self.bearer_token_edit.setPlaceholderText("Bearer token (optional)")
        self.bearer_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.bearer_token_edit.setMinimumWidth(200)
        self.bearer_token_edit.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        integration_layout.addWidget(self.bearer_token_edit)
        self.api_key_label_widget = QLabel(_t("api_key_label"), integration_card)
        integration_layout.addWidget(self.api_key_label_widget)
        self.api_key_edit = QLineEdit(integration_card)
        self.api_key_edit.setPlaceholderText("API key (optional)")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setMinimumWidth(200)
        self.api_key_edit.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        integration_layout.addWidget(self.api_key_edit)

        integration_buttons = QHBoxLayout()
        self.test_api_btn = PrimaryPushButton(FIF.SYNC, _t("test_api"), integration_card)
        self.export_workflow_btn = PrimaryPushButton(FIF.SAVE, _t("export_workflow"), integration_card)
        self.push_workflow_btn = PrimaryPushButton(FIF.SYNC, _t("push_workflow"), integration_card)
        self.open_vault_flows_btn = PrimaryPushButton(FIF.LINK, _t("open_vault_flows"), integration_card)
        integration_buttons.addWidget(self.test_api_btn)
        integration_buttons.addWidget(self.export_workflow_btn)
        integration_buttons.addWidget(self.push_workflow_btn)
        integration_buttons.addWidget(self.open_vault_flows_btn)
        integration_layout.addLayout(integration_buttons)
        layout.addWidget(integration_card)
        
        layout.addStretch(1)
        self.test_api_btn.clicked.connect(self._test_live_api)
        self.export_workflow_btn.clicked.connect(self._export_workflow_package)
        self.push_workflow_btn.clicked.connect(self._push_workflow_package)
        self.open_vault_flows_btn.clicked.connect(self._open_vault_flows)
        
        scroll.setWidget(page)
        return scroll

    def _build_header(self) -> QWidget:
        self._header_frame = QFrame(self)
        self._header_frame.setFixedHeight(52)
        self._header_frame.setStyleSheet(
            f"QFrame {{ background: {_active_theme.surface_elevated};"
            f" border-bottom: 1px solid {_active_theme.border}; border-radius: 0px; }}"
        )
        h_layout = QHBoxLayout(self._header_frame)
        h_layout.setContentsMargins(18, 0, 18, 0)
        h_layout.setSpacing(12)
        brand = QLabel("\U0001f537 VaultWares Studio", self._header_frame)
        brand.setStyleSheet(
            f"color: {_active_theme.text_primary}; font-size: 15px; font-weight: 600;"
        )
        h_layout.addWidget(brand)
        h_layout.addStretch(1)
        theme_label = QLabel("Theme:", self._header_frame)
        theme_label.setStyleSheet(
            f"color: {_active_theme.text_secondary}; font-size: 12px;"
        )
        h_layout.addWidget(theme_label)
        self.header_theme_combo = ComboBox(self._header_frame)
        for t in _tm.get_themes():
            self.header_theme_combo.addItem(t.name)
        self.header_theme_combo.setCurrentText(_active_theme.name)
        self.header_theme_combo.setFixedWidth(180)
        h_layout.addWidget(self.header_theme_combo)
        self.lang_btn = QPushButton("EN / QC" if _active_lang == "EN" else "QC / EN", self._header_frame)
        self.lang_btn.setFixedWidth(80)
        h_layout.addWidget(self.lang_btn)
        self.header_theme_combo.currentIndexChanged.connect(self._on_header_theme_changed)
        self.lang_btn.clicked.connect(self._toggle_language)
        return self._header_frame

    def _build_log_panel(self) -> QWidget:
        from PySide6.QtWidgets import QTextEdit
        panel = QFrame(self)
        panel.setStyleSheet(card_style(_active_theme))
        self._normal_cards.append(panel)
        panel.setMinimumHeight(140)
        log_layout = QVBoxLayout(panel)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_layout.setSpacing(6)
        top_row = QHBoxLayout()
        self.log_title_label = SubtitleLabel(_t("run_log"), panel)
        top_row.addWidget(self.log_title_label)
        top_row.addStretch(1)
        self._clear_log_btn = QPushButton(_t("clear_log"), panel)
        top_row.addWidget(self._clear_log_btn)
        self.return_to_finish_btn = PrimaryPushButton(FIF.ACCEPT, _t("return_to_finish"), panel)
        top_row.addWidget(self.return_to_finish_btn)
        self.return_to_finish_btn.hide() # Not needed anymore
        log_layout.addLayout(top_row)
        self.log_view = QTextEdit(panel)
        self.log_view.setReadOnly(True)
        self.log_view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard | Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.log_view.setMinimumHeight(80)
        log_layout.addWidget(self.log_view, 1)
        self._clear_log_btn.clicked.connect(self.log_view.clear)
        self.return_to_finish_btn.clicked.connect(self._show_finish_panel)
        return panel

    def _on_header_theme_changed(self, index: int) -> None:
        theme = _tm.get_theme(index=index)
        if theme is not None:
            self.theme_changed.emit(theme)

    def _toggle_language(self) -> None:
        global _active_lang
        _active_lang = "QC" if _active_lang == "EN" else "EN"
        _settings.setValue("general/lang", _active_lang)
        self.lang_btn.setText("EN / QC" if _active_lang == "EN" else "QC / EN")
        self.retranslate()
        self.lang_changed.emit(_active_lang)

    def retranslate(self) -> None:
        # left column
        self.job_title.setText(_t("current_job"))
        self.stages_label.setText(_t("job_steps"))
        self.pick_video_btn.setText(_t("choose_video"))
        self.use_demo_btn.setText(_t("use_demo_video"))
        self.open_latest_job_btn.setText(_t("open_latest_job"))
        self.open_manifest_btn.setText(_t("open_manifest"))
        self.run_full_job_btn.setText(_t("run_full_job"))
        self.run_stage_btn.setText(_t("run_selected_step"))
        self.preset_label.setText(_t("preset_label"))
        self.open_job_folder_btn.setText(_t("open_job_folder"))
        # right column state card
        self.state_title.setText(_t("run_state"))
        # step page
        self.prompt_label.setText(_t("prompt_camera"))
        self.camera_prompt_save_btn.setText(_t("save_prompt"))
        self.previews_label.setText(_t("stage_previews"))
        self.artifacts_label.setText(_t("artifacts"))
        for lbl in self.preview_labels:
            if not lbl.pixmap() or lbl.pixmap().isNull():
                lbl.setText(_t("preview_pending"))
        # finish page
        self.finish_label.setText(_t("final_review"))
        self.open_video_btn.setText(_t("open_walkthrough"))
        self.open_viewer_btn.setText(_t("open_3d_viewer"))
        self.open_outputs_btn.setText(_t("open_output_folder"))
        self.integration_label.setText(_t("integration"))
        self.api_url_label_widget.setText(_t("api_url_label"))
        self.app_url_label_widget.setText(_t("app_url_label"))
        self.bearer_label_widget.setText(_t("bearer_label"))
        self.api_key_label_widget.setText(_t("api_key_label"))
        self.test_api_btn.setText(_t("test_api"))
        self.export_workflow_btn.setText(_t("export_workflow"))
        self.push_workflow_btn.setText(_t("push_workflow"))
        self.open_vault_flows_btn.setText(_t("open_vault_flows"))
        self.inspect_label.setText(_t("inspect_steps"))
        # log panel
        self.log_title_label.setText(_t("run_log"))
        self._clear_log_btn.setText(_t("clear_log"))
        self.return_to_finish_btn.setText(_t("return_to_finish"))

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
        self.selected_stage_key = current.data(Qt.ItemDataRole.UserRole)
        self.show_finish_panel = False
        self._render_manifest()

    def _run_selected_stage(self) -> None:
        self._start_worker(run_full_job=False)

    def _run_full_job(self) -> None:
        self._start_worker(run_full_job=True)

    def _maybe_build_remote_runner(self, run_full_job: bool) -> HfJobsStageRunner | None:
        """Pre-confirm the remote reconstruction cost on the GUI thread.

        Returns a pre-confirmed runner, or None to use the local quick path
        (not configured, stage complete, not part of this run, or declined).
        """
        config = HfJobsConfig.load()
        if not (config.enabled and get_hf_token()):
            return None
        recon_stage = next(
            (stage for stage in self.manifest.stages if stage.key == "reconstruction"), None
        )
        if recon_stage is None or recon_stage.state == StageState.COMPLETE.value:
            return None
        if recon_stage.placement != "remote":
            return None
        if not (run_full_job or self.selected_stage_key == "reconstruction"):
            return None
        preset = get_preset(self.manifest.metadata.get("preset"))
        estimate = preset.cost()
        answer = QMessageBox.question(
            self,
            _t("remote_cost_title"),
            _t("remote_cost_body").format(estimate=estimate.summary()),
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._append_log(_t("remote_declined"))
            return None
        return HfJobsStageRunner(config=config, confirm_cost=lambda _estimate: True)

    def _start_worker(self, run_full_job: bool) -> None:
        if self.is_running:
            return
        self._save_camera_prompt()
        self.manifest.metadata["preset"] = self.preset_combo.currentData() or DEFAULT_PRESET_KEY
        remote_runner = self._maybe_build_remote_runner(run_full_job)
        self.signals.running_changed.emit(True)

        def worker() -> None:
            try:
                runner = DigitalTwinStudioRunner(
                    self.manifest,
                    self.signals.log.emit,
                    strict_mode=self.strict_mode,
                    remote_runner=remote_runner,
                )
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
            item.setData(Qt.ItemDataRole.UserRole, stage.key)
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
            self.right_tabs.setCurrentIndex(0)
            if hasattr(self, 'return_to_finish_btn'):
                self.return_to_finish_btn.setVisible(False)
            return

        stage = next(stage for stage in self.manifest.stages if stage.key == self.selected_stage_key)
        self.right_tabs.setCurrentIndex(0)
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
                label.setPixmap(pixmap.scaled(260, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
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
        recon_dir = Path(self.manifest.output_dir) / "reconstruction"
        # cloud.ply now carries full gaussian attributes; the Open3D viewer
        # renders the decimated preview when one exists.
        preview = recon_dir / "cloud_preview.ply"
        point_cloud = preview if preview.exists() else recon_dir / "cloud.ply"

        def worker() -> None:
            success, message = open_live_viewer(point_cloud)
            self.signals.log.emit(message if success else f"[ERROR] {message}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_api_test_finished(self, success: bool, message: str) -> None:
        try:
            if success:
                InfoBar.success("API Test", message, duration=3000, position=InfoBarPosition.TOP, parent=self)  # type: ignore
            else:
                InfoBar.error("API Test", message, duration=5000, position=InfoBarPosition.TOP, parent=self)  # type: ignore
        except NameError:
            self._append_log(f"API Test result: {message}")

    def _test_live_api(self) -> None:
        settings = self._integration_settings()

        def worker() -> None:
            try:
                result = test_vaultwares_api(settings)
                self.signals.log.emit(json.dumps(result, indent=2))
                self.signals.api_test_finished.emit(True, "Connection to Vaultwares API successful")
            except Exception as exc:  # noqa: BLE001
                self.signals.log.emit(f"[ERROR] {exc}")
                self.signals.api_test_finished.emit(False, str(exc))

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
        self._header_frame.setStyleSheet(
            f"QFrame {{ background: {theme.surface_elevated};"
            f" border-bottom: 1px solid {theme.border}; border-radius: 0px; }}"
        )


class Window(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Twin Studio")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.dashboard = DashboardWidget(self)
        self.settings = SettingsTab(self)
        self.viewport = ViewportTab(self, translate=_t)
        self.settings.toggle_btn.clicked.connect(self._toggle_strict_mode)
        self.settings.theme_changed.connect(self._apply_theme)

        self.dashboard.theme_changed.connect(self._apply_theme)
        self.dashboard.lang_changed.connect(self._apply_lang)
        self.viewport.log.connect(self.dashboard.signals.log.emit)
        self.dashboard.signals.manifest_changed.connect(
            lambda manifest: self.viewport.set_job(Path(manifest.output_dir))
        )
        self.addSubInterface(self.dashboard, FIF.HOME, "Studio")
        self.addSubInterface(self.viewport, FIF.VIDEO, _t("tab_viewport"))
        self.addSubInterface(self.settings, FIF.SETTING, "Settings", NavigationItemPosition.BOTTOM)
        self.resize(1440, 900)
        self.viewport.set_job(Path(self.dashboard.manifest.output_dir))

    def _toggle_strict_mode(self) -> None:
        strict_mode = not self.dashboard.strict_mode
        _settings.setValue("general/strict_mode", strict_mode)
        self.dashboard.set_strict_mode(strict_mode)
        self.settings.set_mode(strict_mode)

    def _apply_theme(self, theme: VaultTheme) -> None:
        global _active_theme
        _active_theme = theme
        _settings.setValue("theme", theme.name)
        qt_theme = Theme.DARK if theme.mode == "dark" else Theme.LIGHT
        setTheme(qt_theme)
        setThemeColor(QColor(theme.accent))
        _qapp = QApplication.instance()
        if isinstance(_qapp, QApplication):
            _qapp.setStyleSheet(build_stylesheet(theme))
        self.dashboard.refresh_cards(theme)
        self.settings.setStyleSheet(card_style(theme))
        self.settings.theme_swatch.setStyleSheet(
            f"background: {theme.accent}; border-radius: 4px;"
        )
        for combo in (self.settings.theme_combo, self.dashboard.header_theme_combo):
            combo.blockSignals(True)
            combo.setCurrentText(theme.name)
            combo.blockSignals(False)

    def _apply_lang(self, lang: str) -> None:
        self.settings.retranslate()


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    register_viewer_scheme()  # must precede QApplication construction
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
