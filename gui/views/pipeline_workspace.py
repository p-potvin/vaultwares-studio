import os
import json
import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QComboBox, QProgressBar, QTextEdit, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QObject, Signal, Slot

from vaultwares_studio.pipeline import (
    DEFAULT_CAMERA_PROMPT,
    DEFAULT_SOURCE_VIDEO,
    JOBS_DIR,
    DigitalTwinStudioRunner,
    JobManifest,
    StageState,
    create_job_manifest,
    load_latest_job_manifest
)
from vaultwares_studio.presets import DEFAULT_PRESET_KEY, PRESETS, get_preset
from vaultwares_studio.runners import (
    HfJobsConfig,
    HfJobsStageRunner,
    get_hf_token
)
from gui.strings import t as _t, STATE_LABELS

class WorkspaceSignals(QObject):
    log = Signal(str)
    manifest_changed = Signal(object)
    running_changed = Signal(bool)

class PipelineWorkspace(QFrame):
    """
    The central operational core of the application (Console Mode).
    Houses the video picker, preset selector, run logs, and active pipeline state.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ConsoleShell")
        
        self.signals = WorkspaceSignals()
        self.signals.log.connect(self.append_log)
        self.signals.manifest_changed.connect(self._on_manifest_changed)
        self.signals.running_changed.connect(self._set_running)

        self.manifest = None
        self.is_running = False
        self.strict_mode = False

        self.setup_ui()
        
        # Load the latest job or initialize a new one with the demo video
        latest = load_latest_job_manifest()
        if latest:
            self.manifest = latest
            self._render_manifest()
            self.append_log(f"Loaded previous job: {latest.job_id}")
        else:
            self._reset_job(DEFAULT_SOURCE_VIDEO)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 50, 40, 40)
        main_layout.setSpacing(24)

        # Header Title
        self.title_label = QLabel("Digital Twin Pipeline")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(self.title_label)

        # Central configuration card (Matte Hardware Panel)
        config_card = QFrame()
        config_card.setProperty("class", "vw-card")
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(24, 24, 24, 24)
        config_layout.setSpacing(16)

        # File selection row
        file_row = QHBoxLayout()
        self.file_label = QLabel(f"Source: {DEFAULT_SOURCE_VIDEO.name}")
        self.btn_pick_file = QPushButton("Select Video...")
        self.btn_pick_file.clicked.connect(self._pick_video)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.btn_pick_file)
        config_layout.addLayout(file_row)

        # Preset selection row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quality Preset:"))
        self.preset_combo = QComboBox()
        for preset in PRESETS.values():
            self.preset_combo.addItem(preset.label, userData=preset.key)
        default_index = list(PRESETS).index(DEFAULT_PRESET_KEY)
        self.preset_combo.setCurrentIndex(default_index)
        preset_row.addWidget(self.preset_combo, 1)
        config_layout.addLayout(preset_row)

        # Actions row
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.btn_run = QPushButton("INITIALIZE PIPELINE")
        self.btn_run.setProperty("class", "primary")
        self.btn_run.setMinimumWidth(200)
        self.btn_run.clicked.connect(self._run_full_job)
        action_row.addWidget(self.btn_run)
        config_layout.addLayout(action_row)

        main_layout.addWidget(config_card)

        # Run Logs / Execution space
        log_label = QLabel("Execution Logs")
        log_label.setStyleSheet("font-weight: bold; letter-spacing: 1px;")
        main_layout.addWidget(log_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("No active jobs. Awaiting initialization...")
        main_layout.addWidget(self.log_area, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        main_layout.addWidget(self.progress_bar)

    @Slot(str)
    def append_log(self, text: str):
        self.log_area.append(text)
        
    @Slot(bool)
    def _set_running(self, running: bool):
        self.is_running = running
        self.btn_pick_file.setEnabled(not running)
        self.btn_run.setEnabled(not running)
        if running:
            self.btn_run.setText("RUNNING...")
        else:
            self.btn_run.setText("INITIALIZE PIPELINE")
            
    @Slot(object)
    def _on_manifest_changed(self, manifest: JobManifest):
        self.manifest = manifest
        self._render_manifest()

    def _render_manifest(self):
        if not self.manifest:
            return
        
        self.title_label.setText(f"Digital Twin Pipeline — {self.manifest.job_id}")
        self.file_label.setText(f"Source: {Path(self.manifest.source_video).name}")
        
        completed = len([stage for stage in self.manifest.stages if stage.state == StageState.COMPLETE.value])
        total = len(self.manifest.stages)
        progress = int((completed / total) * 100) if total else 0
        self.progress_bar.setValue(progress)

    def _pick_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input video",
            str(Path(__file__).resolve().parent.parent.parent),
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*.*)"
        )
        if file_path:
            self._reset_job(Path(file_path))

    def _reset_job(self, source_video: Path):
        if self.is_running:
            return
        self.manifest = create_job_manifest(source_video=source_video, camera_prompt=DEFAULT_CAMERA_PROMPT)
        self.log_area.clear()
        self.append_log(f"Created new job {self.manifest.job_id} for {source_video.name}")
        self._render_manifest()
        
    def _maybe_build_remote_runner(self):
        config = HfJobsConfig.load()
        if not (config.enabled and get_hf_token()):
            return None
            
        recon_stage = next((stage for stage in self.manifest.stages if stage.key == "reconstruction"), None)
        if recon_stage is None or recon_stage.state == StageState.COMPLETE.value:
            return None
        if recon_stage.placement != "remote":
            return None
            
        preset = get_preset(self.manifest.metadata.get("preset"))
        estimate = preset.cost()
        answer = QMessageBox.question(
            self,
            _t("remote_cost_title"),
            _t("remote_cost_body").format(estimate=estimate.summary()),
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.append_log(_t("remote_declined"))
            return None
        return HfJobsStageRunner(config=config, confirm_cost=lambda _estimate: True)

    def _run_full_job(self):
        if self.is_running or not self.manifest:
            return
            
        # Ensure we capture preset changes
        self.manifest.metadata["preset"] = self.preset_combo.currentData() or DEFAULT_PRESET_KEY
        
        # Save manifest
        manifest_path = Path(self.manifest.output_dir) / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(self.manifest.to_dict(), indent=2), encoding="utf-8")
        
        remote_runner = self._maybe_build_remote_runner()
        self.signals.running_changed.emit(True)

        def worker():
            try:
                runner = DigitalTwinStudioRunner(
                    self.manifest,
                    self.signals.log.emit,
                    strict_mode=self.strict_mode,
                    remote_runner=remote_runner,
                )
                result = runner.run_remaining()
            except Exception as exc:
                self.signals.log.emit(f"[ERROR] {exc}")
                self.signals.manifest_changed.emit(self.manifest)
            else:
                self.signals.manifest_changed.emit(result)
            finally:
                self.signals.running_changed.emit(False)

        threading.Thread(target=worker, daemon=True).start()
