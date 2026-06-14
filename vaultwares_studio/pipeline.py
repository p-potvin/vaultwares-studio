from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from .camera_director import build_camera_bundle
from .camera_paths import (
    CameraEntity,
    CameraKeyframe,
    author_usd_camera,
    build_visit_path,
    load_captured_entities,
    to_nerfstudio_camera_path,
)
from .presets import get_preset
from .runners import (
    CancelToken,
    CostDeniedError,
    LocalStageRunner,
    StageCancelledError,
    StageContext,
    StageRunner,
)
from .splat_io import convert_splat_outputs, is_gaussian_ply

MANIFEST_SCHEMA_VERSION = 2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
DEFAULT_SOURCE_VIDEO = ROOT / "my-room.mp4"
DEFAULT_CAMERA_PROMPT = "show me the desk from the doorway, then orbit left and rise"

DEPENDENCY_INSTALL_HINTS = {
    "ffmpeg": "winget install ffmpeg",
    "ffprobe": "Install ffmpeg and make sure ffprobe is on PATH",
    "colmap": "Install COLMAP binary manually and add it to PATH",
    "ns-process-data": ".venv\\Scripts\\python.exe -m pip install nerfstudio",
    "ns-train": ".venv\\Scripts\\python.exe -m pip install nerfstudio",
    "PySide6": ".venv\\Scripts\\python.exe -m pip install PySide6",
    "qfluentwidgets": ".venv\\Scripts\\python.exe -m pip install PySide6-Fluent-Widgets",
    "redis": ".venv\\Scripts\\python.exe -m pip install redis",
    "pxr": ".venv\\Scripts\\python.exe -m pip install usd-core",
    "open3d": ".venv\\Scripts\\python.exe -m pip install open3d",
    "PIL": ".venv\\Scripts\\python.exe -m pip install Pillow",
}

COLMAP_CANDIDATE_PATHS = [
    Path(os.environ["COLMAP_EXE"]).expanduser()
    for _ in [0]
    if os.environ.get("COLMAP_EXE")
]
COLMAP_CANDIDATE_PATHS.append(Path(r"C:\Users\Administrator\Desktop\COLMAP\COLMAP.bat"))
COLMAP_CANDIDATE_PATHS.append(ROOT / "tools" / "colmap" / "COLMAP.bat")
COLMAP_CANDIDATE_PATHS.append(ROOT / "tools" / "colmap" / "bin" / "colmap.exe")


class StageState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_INSTALL = "needs-install"
    NEEDS_USER_INPUT = "needs-user-input"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ArtifactRecord:
    label: str
    kind: str
    path: str
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StageRecord:
    key: str
    title: str
    description: str
    state: str = StageState.QUEUED.value
    message: str = ""
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Schema v2: where this stage prefers to execute ("local" | "remote"),
    # which runner executed it, runner parameters, and cost records for
    # paid remote runs. Remote placement is honored once a remote runner is
    # configured (M1+); until then execution falls back to local handlers.
    placement: str = "local"
    runner: str = "local"
    params: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return payload


@dataclass
class JobManifest:
    job_id: str
    source_video: str
    output_dir: str
    execution_profile: str
    mode: str
    state: str
    current_stage_key: str
    walkthrough_video: str | None
    live_viewer_supported: bool
    metadata: dict
    stages: list[StageRecord]
    created_at: str
    updated_at: str
    schema_version: int = MANIFEST_SCHEMA_VERSION
    spend_ledger: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["stages"] = [stage.to_dict() for stage in self.stages]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "JobManifest":
        # v1 manifests (no schema_version) migrate with additive defaults:
        # stages gain placement/runner/params/cost from the stage definitions.
        is_v1 = "schema_version" not in payload
        default_placements = {
            definition.key: definition.default_placement for definition in STAGE_DEFINITIONS
        }
        # Stage rename usd_cameras -> camera_staging (June 2026). Migrate
        # in-place so existing job manifests open without manual editing.
        _LEGACY_KEY_REMAP = {"usd_cameras": "camera_staging"}
        for stage in payload.get("stages", []):
            stage["key"] = _LEGACY_KEY_REMAP.get(stage.get("key"), stage.get("key"))
        if "current_stage_key" in payload:
            payload["current_stage_key"] = _LEGACY_KEY_REMAP.get(
                payload["current_stage_key"], payload["current_stage_key"]
            )
        stages = [
            StageRecord(
                key=stage["key"],
                title=stage["title"],
                description=stage["description"],
                state=stage.get("state", StageState.QUEUED.value),
                message=stage.get("message", ""),
                artifacts=[
                    ArtifactRecord(**artifact) for artifact in stage.get("artifacts", [])
                ],
                metadata=stage.get("metadata", {}),
                placement=stage.get(
                    "placement",
                    default_placements.get(stage["key"], "local") if is_v1 else "local",
                ),
                runner=stage.get("runner", "local"),
                params=stage.get("params", {}),
                cost=stage.get("cost", {}),
            )
            for stage in payload["stages"]
        ]
        return cls(
            job_id=payload["job_id"],
            source_video=payload["source_video"],
            output_dir=payload["output_dir"],
            execution_profile=payload["execution_profile"],
            mode=payload["mode"],
            state=payload["state"],
            current_stage_key=payload["current_stage_key"],
            walkthrough_video=payload.get("walkthrough_video"),
            live_viewer_supported=payload.get("live_viewer_supported", False),
            metadata=payload.get("metadata", {}),
            stages=stages,
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            schema_version=MANIFEST_SCHEMA_VERSION,
            spend_ledger=payload.get("spend_ledger", []),
        )


@dataclass(frozen=True)
class StageDefinition:
    key: str
    title: str
    description: str
    # "remote" stages run on rented GPU compute (HF Jobs) once a remote
    # runner is configured; everything else stays on the local machine.
    default_placement: str = "local"


STAGE_DEFINITIONS = [
    StageDefinition(
        "video_intake",
        "Video Intake",
        "Inspect the input video and initialize a local-safe job profile.",
    ),
    StageDefinition(
        "frame_extraction",
        "Frame Extraction",
        "Extract frames, keep previews, and capture sampling metadata.",
    ),
    StageDefinition(
        "reconstruction",
        "Reconstruction",
        "Run COLMAP / Nerfstudio / gsplat or fall back to placeholder-safe outputs.",
        default_placement="remote",
    ),
    StageDefinition(
        "camera_staging",
        "Camera Staging",
        "Compose the USD stage and stage preset, prompt-directed, and captured cameras.",
    ),
    StageDefinition(
        "cosmos_output",
        "Cosmos + Output",
        "Write optional Cosmos outputs and produce the final walkthrough video.",
    ),
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "camera_shot"


def build_dependency_health() -> list[dict[str, str]]:
    binary_deps = ["ffmpeg", "ffprobe", "colmap", "ns-process-data", "ns-train"]
    python_deps = ["PySide6", "qfluentwidgets", "redis", "pxr", "open3d", "PIL"]
    rows: list[dict[str, str]] = []

    for dep in binary_deps:
        resolved = resolve_binary(dep)
        rows.append(
            {
                "name": dep,
                "kind": "binary",
                "status": "ok" if resolved else "missing",
                "detail": resolved or DEPENDENCY_INSTALL_HINTS.get(dep, "No install hint available"),
            }
        )

    for dep in python_deps:
        available = bool(importlib.util.find_spec(dep))
        rows.append(
            {
                "name": dep,
                "kind": "python",
                "status": "ok" if available else "missing",
                "detail": "Available" if available else DEPENDENCY_INSTALL_HINTS.get(dep, "No install hint available"),
            }
        )

    return rows


def resolve_binary(name: str) -> str | None:
    resolved = shutil.which(name)
    if resolved:
        return resolved

    if name == "colmap":
        for candidate in COLMAP_CANDIDATE_PATHS:
            if candidate.exists():
                return str(candidate)

    return None


def create_job_manifest(
    source_video: Path | str = DEFAULT_SOURCE_VIDEO,
    camera_prompt: str = DEFAULT_CAMERA_PROMPT,
    mode: str = "guided",
) -> JobManifest:
    source_path = Path(source_video).resolve()
    job_id = f"local-run-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = (JOBS_DIR / job_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = [
        StageRecord(
            key=stage.key,
            title=stage.title,
            description=stage.description,
            placement=stage.default_placement,
        )
        for stage in STAGE_DEFINITIONS
    ]
    manifest = JobManifest(
        job_id=job_id,
        source_video=str(source_path),
        output_dir=str(output_dir),
        execution_profile="RTX 3060 12GB Safe",
        mode=mode,
        state=StageState.QUEUED.value,
        current_stage_key=stages[0].key,
        walkthrough_video=None,
        live_viewer_supported=bool(importlib.util.find_spec("open3d")),
        metadata={"cameraPrompt": camera_prompt},
        stages=stages,
        created_at=_now(),
        updated_at=_now(),
    )
    save_job_manifest(manifest)
    return manifest


def save_job_manifest(manifest: JobManifest) -> Path:
    manifest.updated_at = _now()
    manifest_path = Path(manifest.output_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest_path


def load_job_manifest(path: Path | str) -> JobManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return JobManifest.from_dict(payload)


def list_job_manifests(jobs_dir: Path | str = JOBS_DIR) -> list[Path]:
    root = Path(jobs_dir)
    if not root.exists():
        return []

    manifests = [path for path in root.glob("*/manifest.json") if path.is_file()]
    return sorted(manifests, key=lambda path: path.stat().st_mtime, reverse=True)


def load_latest_job_manifest(jobs_dir: Path | str = JOBS_DIR) -> JobManifest | None:
    manifests = list_job_manifests(jobs_dir)
    if not manifests:
        return None

    return load_job_manifest(manifests[0])


def completed_stage_count(manifest: JobManifest) -> int:
    return len([stage for stage in manifest.stages if stage.state == StageState.COMPLETE.value])


def next_incomplete_stage_key(manifest: JobManifest) -> str | None:
    for stage in manifest.stages:
        if stage.state != StageState.COMPLETE.value:
            return stage.key

    return None


def stage_dependencies_complete(manifest: JobManifest, stage_key: str) -> bool:
    for stage in manifest.stages:
        if stage.key == stage_key:
            return True
        if stage.state != StageState.COMPLETE.value:
            return False

    raise KeyError(stage_key)


# Walkthrough footage wants ~300 frames for robust SfM (nerfstudio's own
# video target); we extract at ~2x that and keep the sharpest per bucket.
FRAME_KEEP_TARGET = 280
FRAME_EXTRACT_TARGET = 560
FRAME_PATTERNS = ("*.jpg", "*.png")


def list_frames(frames_dir: Path) -> list[Path]:
    return sorted(path for pattern in FRAME_PATTERNS for path in frames_dir.glob(pattern))


def compute_extraction_fps(duration_seconds: float | None, target_frames: int = 100) -> int:
    """Sampling rate that yields ~target_frames regardless of clip length.

    Short clips need denser sampling or COLMAP sequential matching starves
    (a 12 s clip at the old fixed 2 fps gave only 24 frames and registered 3).
    """
    if not duration_seconds or duration_seconds <= 0:
        return 2
    return max(2, min(10, round(target_frames / duration_seconds)))


def record_spend(manifest: JobManifest, stage_key: str, cost_metadata: dict) -> None:
    """Append a paid-run record to the manifest's spend ledger and persist it."""
    entry = {"stage": stage_key, "recorded_at": _now(), **cost_metadata}
    manifest.spend_ledger.append(entry)
    for stage in manifest.stages:
        if stage.key == stage_key:
            stage.cost = cost_metadata
            break
    save_job_manifest(manifest)


class DigitalTwinStudioRunner:
    def __init__(
        self,
        manifest: JobManifest,
        log: Callable[[str], None],
        strict_mode: bool = False,
        local_runner: LocalStageRunner | None = None,
        remote_runner: StageRunner | None = None,
    ):
        self.manifest = manifest
        self.log = log
        self.strict_mode = strict_mode
        self.local_runner = local_runner or LocalStageRunner()
        # Remote runner (HF Jobs / SSH GPU). Stages with placement == "remote"
        # delegate to it starting in M1; until then it is carried but unused.
        self.remote_runner = remote_runner
        self.cancel_token = CancelToken()
        self.root = Path(manifest.output_dir)
        self.frames_dir = self.root / "frames"
        self.recon_dir = self.root / "reconstruction"
        self.usd_dir = self.root / "usd"
        self.cameras_dir = self.root / "camera_previews"
        self.cosmos_dir = self.root / "cosmos"
        self.deliverables_dir = self.root / "deliverables"
        self.recon_stage_path = self.recon_dir / "cloud.usda"
        self.recon_ply_path = self.recon_dir / "cloud.ply"
        self.recon_preview_ply_path = self.recon_dir / "cloud_preview.ply"
        self.recon_splat_path = self.recon_dir / "cloud.splat"
        self.usd_stage_path = self.usd_dir / "digital_twin_scene.usda"
        self.camera_plan_path = self.usd_dir / "camera_plan.json"
        self.camera_render_path = self.usd_dir / "camera_path.json"
        self.walkthrough_path = self.deliverables_dir / "digital_twin_walkthrough.mp4"
        self.splat_walkthrough_path = self.deliverables_dir / "walkthrough.mp4"

    def stage_for(self, stage_key: str) -> StageRecord:
        for stage in self.manifest.stages:
            if stage.key == stage_key:
                return stage
        raise KeyError(stage_key)

    def run_stage(self, stage_key: str) -> JobManifest:
        handlers = {
            "video_intake": self._run_video_intake,
            "frame_extraction": self._run_frame_extraction,
            "reconstruction": self._run_reconstruction,
            "camera_staging": self._run_camera_staging,
            "cosmos_output": self._run_cosmos_output,
        }
        if not stage_dependencies_complete(self.manifest, stage_key):
            raise RuntimeError(
                "Complete earlier stages before running this step, or use Run Full Job."
            )
        stage = self.stage_for(stage_key)
        self.manifest.current_stage_key = stage_key
        self.manifest.state = StageState.RUNNING.value
        self._transition(stage, StageState.RUNNING, f"Running {stage.title}")
        try:
            handlers[stage_key](stage)
            # A handler that explicitly set its own terminal state (e.g.
            # NEEDS_USER_INPUT) keeps it; only auto-complete when the handler
            # finished without taking a side path.
            if stage.state == StageState.RUNNING.value:
                self._transition(stage, StageState.COMPLETE, stage.message or f"{stage.title} complete.")
            else:
                save_job_manifest(self.manifest)
        except Exception as exc:  # noqa: BLE001
            self.manifest.state = StageState.FAILED.value
            self._transition(stage, StageState.FAILED, str(exc))
            raise
        else:
            if all(s.state == StageState.COMPLETE.value for s in self.manifest.stages):
                self.manifest.state = StageState.COMPLETE.value
            save_job_manifest(self.manifest)
            return self.manifest

    def run_remaining(self, start_stage_key: str | None = None) -> JobManifest:
        start_found = start_stage_key is None
        for definition in STAGE_DEFINITIONS:
            if definition.key == start_stage_key:
                start_found = True
            if not start_found:
                continue
            stage = self.stage_for(definition.key)
            if stage.state == StageState.COMPLETE.value:
                continue
            self.run_stage(definition.key)
        return self.manifest

    def _transition(self, stage: StageRecord, state: StageState, message: str) -> None:
        stage.state = state.value
        stage.message = message
        self.log(message)
        save_job_manifest(self.manifest)

    def _add_artifact(self, stage: StageRecord, label: str, kind: str, path: Path, description: str = "") -> None:
        record = ArtifactRecord(label=label, kind=kind, path=str(path), description=description)
        stage.artifacts = [artifact for artifact in stage.artifacts if artifact.path != record.path]
        stage.artifacts.append(record)
        save_job_manifest(self.manifest)

    def _run_video_intake(self, stage: StageRecord) -> None:
        source_video = Path(self.manifest.source_video)
        if not source_video.exists():
            raise FileNotFoundError(f"Missing source video: {source_video}")
        metadata = {
            "sourceVideo": str(source_video),
            "fileSizeBytes": source_video.stat().st_size,
            "executionProfile": self.manifest.execution_profile,
            "mode": self.manifest.mode,
        }
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            cmd = [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(source_video),
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if completed.returncode == 0 and completed.stdout:
                metadata["probe"] = json.loads(completed.stdout)
        metadata_path = self.root / "input_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        stage.metadata = metadata
        stage.message = "Video metadata captured and job initialized."
        self._add_artifact(stage, "Input Metadata", "json", metadata_path, "Video intake metadata.")

    def _run_frame_extraction(self, stage: StageRecord) -> None:
        ffmpeg = resolve_binary("ffmpeg")
        if ffmpeg is None:
            stage.state = StageState.NEEDS_INSTALL.value
            stage.message = DEPENDENCY_INSTALL_HINTS["ffmpeg"]
            save_job_manifest(self.manifest)
            raise RuntimeError("ffmpeg is required for frame extraction.")
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for stale in list_frames(self.frames_dir):
            stale.unlink(missing_ok=True)
        duration = None
        intake = self.stage_for("video_intake")
        try:
            duration = float(intake.metadata["probe"]["format"]["duration"])
        except (KeyError, TypeError, ValueError):
            pass
        fps = compute_extraction_fps(duration, target_frames=FRAME_EXTRACT_TARGET)
        self.log(
            f"Sampling at {fps} fps (duration: {duration or 'unknown'}s); "
            f"keeping the sharpest ~{FRAME_KEEP_TARGET} frames"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            self.manifest.source_video,
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(self.frames_dir / "frame_%05d.jpg"),
        ]
        self._run_command(cmd, "Frame extraction failed.", timeout_seconds=1800)
        extracted_count = len(list_frames(self.frames_dir))
        kept_count = extracted_count
        if extracted_count > FRAME_KEEP_TARGET:
            try:
                from .frame_selection import prune_to_sharpest

                extracted_count, kept_count = prune_to_sharpest(self.frames_dir, FRAME_KEEP_TARGET)
                self.log(f"Kept {kept_count}/{extracted_count} sharpest frames (motion-blur filter).")
            except ImportError:
                self.log("OpenCV unavailable; skipping blur-aware frame selection.")
        frame_paths = list_frames(self.frames_dir)
        if not frame_paths:
            raise RuntimeError("No frames were extracted.")
        preview_manifest = self.frames_dir / "frames_manifest.json"
        preview_manifest.write_text(
            json.dumps(
                {
                    "frameCount": len(frame_paths),
                    "sampleFrames": [str(path) for path in frame_paths[:6]],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        stage.metadata = {"frameCount": len(frame_paths), "fps": fps, "extracted": extracted_count}
        stage.message = f"Extracted {len(frame_paths)} frames."
        self._add_artifact(stage, "Frames Manifest", "json", preview_manifest, "Frame extraction summary.")
        for index, frame in enumerate(frame_paths[:3], start=1):
            self._add_artifact(stage, f"Frame Preview {index}", "image", frame, "Sample extracted frame.")

    def _run_reconstruction(self, stage: StageRecord) -> None:
        self.recon_dir.mkdir(parents=True, exist_ok=True)
        if not list_frames(self.frames_dir):
            raise RuntimeError("Frame extraction must complete before reconstruction.")
        preset = get_preset(self.manifest.metadata.get("preset"))
        exported_ply: Path | None = None

        if stage.placement == "remote" and self.remote_runner is not None:
            try:
                exported_ply = self._run_remote_reconstruction(stage, preset)
            except StageCancelledError:
                raise
            except CostDeniedError as exc:
                self.log(f"{exc} Using the local quick path instead.")
            except Exception as exc:  # noqa: BLE001 - incl. hub HTTP errors, which are not RuntimeErrors
                if self.strict_mode:
                    raise
                self.log(f"Remote reconstruction failed, falling back to local path: {exc}")
        elif stage.placement == "remote":
            self.log(
                "Reconstruction prefers remote execution but no remote runner is "
                "configured (Settings > Remote Compute). Using the local quick path."
            )

        if exported_ply is None:
            exported_ply = self._run_local_reconstruction(stage)

        degraded = True
        if exported_ply is not None and is_gaussian_ply(exported_ply):
            try:
                info = convert_splat_outputs(
                    exported_ply,
                    self.recon_ply_path,
                    self.recon_preview_ply_path,
                    self.recon_stage_path,
                    self.log,
                )
                stage.metadata.update(info)
                degraded = False
            except Exception as exc:  # noqa: BLE001
                if self.strict_mode:
                    raise
                self.log(f"Splat conversion failed: {exc}")
        elif exported_ply is not None:
            # Unexpected non-gaussian PLY: keep the legacy point-cloud conversion.
            degraded = not self._convert_ply_to_cloud_files(exported_ply)

        if not self.recon_stage_path.exists():
            self._write_placeholder_reconstruction()
            degraded = True
        if not self.recon_ply_path.exists():
            self._write_placeholder_ply()
            degraded = True

        if not degraded and self.recon_preview_ply_path.exists():
            self._gravity_align(stage)
        if not degraded and self.recon_ply_path.exists():
            self._write_packed_splat(stage)

        stage.metadata["degraded"] = degraded
        stage.metadata["preset"] = preset.key
        stage.message = (
            "Reconstruction completed with placeholder-safe outputs."
            if degraded
            else f"Reconstruction completed ({stage.metadata.get('gaussians', '?')} gaussians, preset: {preset.key})."
        )
        self._add_artifact(stage, "Reconstruction Stage", "usd", self.recon_stage_path, "Reconstruction stage.")
        self._add_artifact(stage, "Reconstruction PLY", "ply", self.recon_ply_path, "Gaussian splat output.")
        if self.recon_preview_ply_path.exists():
            self._add_artifact(
                stage, "Preview Point Cloud", "ply", self.recon_preview_ply_path,
                "Decimated point cloud for the live viewer.",
            )

    def _run_remote_reconstruction(self, stage: StageRecord, preset) -> Path | None:
        """Train the splat on rented GPU compute (HF Jobs). Returns the splat PLY."""
        import json as _json
        import zipfile

        runner_config = getattr(self.remote_runner, "config", None)
        image_name = getattr(runner_config, "worker_image", "") if runner_config else ""
        if not image_name or image_name.startswith("python:"):
            raise RuntimeError(
                "Remote worker image not configured. Build it with "
                "tools/build_worker_image.ps1 and set worker_image in Settings."
            )

        frames_zip = self.recon_dir / "frames.zip"
        frame_paths = list_frames(self.frames_dir)
        with zipfile.ZipFile(frames_zip, "w", zipfile.ZIP_STORED) as archive:
            for frame in frame_paths:
                archive.write(frame, frame.name)
        self.log(f"Packed {len(frame_paths)} frames for remote reconstruction ({frames_zip.stat().st_size // 1_000_000} MB)")

        export_dir = self.recon_dir / "gsplat_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        splat_path = export_dir / "splat.ply"
        summary_path = self.recon_dir / "summary.json"
        # Force the worker bundle into expected_outputs so a missing render
        # bundle fails the reconstruction stage NOW instead of silently
        # falling back to a slideshow at cosmos_output time. _run_remote_render
        # checks both files at reconstruction/remote_out/.
        remote_out_dir = self.recon_dir / "remote_out"
        bundle_model = remote_out_dir / "model.zip"
        bundle_processed = remote_out_dir / "processed_min.zip"
        ctx = StageContext(
            job_dir=self.root,
            job_id=self.manifest.job_id,
            stage_key="reconstruction",
            params={
                "image": image_name,
                "image_has_hub": True,
                "flavor": preset.flavor,
                "est_minutes": preset.est_minutes,
                # Headroom for COLMAP non-determinism: when sequential matching
                # draws a degenerate init pair, retry_mapper either recovers in
                # ~2 min OR falls through to exhaustive matching, which adds
                # 25-40 min on top of the baseline. 6x est gives a job that
                # finishes in ~10-15 min on the happy path and still has room
                # for the exhaustive fallback.
                "timeout_seconds": int(max(3600, preset.est_minutes * 60 * 6)),
                "command": [
                    "python", "/opt/vw/recon_entrypoint.py",
                    "--downscale", str(preset.downscale_factor),
                    "--train-args", _json.dumps(preset.train_args()),
                    "--keep-checkpoint",
                ],
            },
            inputs=[frames_zip],
            expected_outputs=[splat_path, summary_path, bundle_model, bundle_processed],
            log=self.log,
            cancel=self.cancel_token,
        )
        result = self.remote_runner.run(ctx)
        stage.runner = self.remote_runner.name
        stage.params = {"preset": preset.key, "flavor": preset.flavor}
        if result.metadata:
            record_spend(self.manifest, "reconstruction", result.metadata)
        return splat_path if splat_path.exists() else None

    def _run_local_reconstruction(self, stage: StageRecord) -> Path | None:
        """Local quick path (250-iteration smoke training). Heavy local runs stay opt-in."""
        ns_process_data = resolve_binary("ns-process-data")
        colmap_bin = resolve_binary("colmap")
        ns_train = resolve_binary("ns-train")

        run_env = os.environ.copy()
        run_env["PYTHONUTF8"] = "1"
        if colmap_bin:
            colmap_dir = str(Path(colmap_bin).parent.resolve())
            run_env["PATH"] = f"{colmap_dir}{os.pathsep}{run_env.get('PATH', '')}"

        if not (ns_process_data and colmap_bin):
            return None
        cmd = [
            ns_process_data,
            "images",
            "--data",
            str(self.frames_dir),
            "--output-dir",
            str(self.recon_dir),
        ]
        if str(colmap_bin).upper().endswith(".BAT"):
            cmd.extend(["--colmap-cmd", "COLMAP.bat"])
        try:
            self._run_command(cmd, "Reconstruction failed.", timeout_seconds=3600, env=run_env)
        except RuntimeError:
            if self.strict_mode:
                raise
            return None

        transforms_path = self.recon_dir / "transforms.json"
        if not (ns_train and transforms_path.exists()):
            return None
        local_preset = get_preset("local-debug")
        train_cmd = [
            ns_train,
            "splatfacto",
            "--data",
            str(self.recon_dir),
            "--output-dir",
            str(self.recon_dir / "gsplat_outputs"),
            *local_preset.train_args(),
        ]
        try:
            self._run_command(train_cmd, "gsplat training failed.", timeout_seconds=3600, env=run_env)
            return self._export_gsplat_ply()
        except RuntimeError:
            if self.strict_mode:
                raise
            return None

    def _run_camera_staging(self, stage: StageRecord) -> None:
        # Read the pause flag BEFORE any handler logic touches stage.metadata;
        # the metadata dict is replaced wholesale further down, which would
        # wipe the flag and trap re-runs in a permanent NEEDS_USER_INPUT loop.
        already_paused = bool(stage.metadata.get("pausedForUserInput"))
        self.usd_dir.mkdir(parents=True, exist_ok=True)
        self.cameras_dir.mkdir(parents=True, exist_ok=True)
        stage_path = Usd.Stage.CreateNew(str(self.usd_stage_path))
        UsdGeom.SetStageUpAxis(stage_path, UsdGeom.Tokens.y)
        world = UsdGeom.Xform.Define(stage_path, "/World")
        stage_path.SetDefaultPrim(world.GetPrim())
        ground = UsdGeom.Cube.Define(stage_path, "/World/Environment/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddScaleOp().Set(Gf.Vec3f(20.0, 0.1, 20.0))
        ground.AddTranslateOp().Set(Gf.Vec3f(0.0, -0.05, 0.0))
        light = UsdLux.DistantLight.Define(stage_path, "/World/Environment/Sun")
        light.CreateIntensityAttr(900.0)
        twin = UsdGeom.Xform.Define(stage_path, "/World/DigitalTwin")
        twin.GetPrim().CreateAttribute("sourceVideo", Sdf.ValueTypeNames.String, custom=True).Set(self.manifest.source_video)
        if self.recon_stage_path.exists():
            twin.GetPrim().GetReferences().AddReference(str(self.recon_stage_path))

        bundle = build_camera_bundle(str(self.manifest.metadata.get("cameraPrompt", DEFAULT_CAMERA_PROMPT)))
        UsdGeom.Xform.Define(stage_path, "/World/Navigation")

        center = self._scene_center()
        entities: list[CameraEntity] = []
        for shot in bundle["allShots"]:
            entities.append(
                CameraEntity(
                    name=shot["name"],
                    source=shot["source"],
                    keyframes=[
                        CameraKeyframe(t=0.0, position=list(shot["position"]), look_at=list(center))
                    ],
                )
            )
        captured = load_captured_entities(Path(self.manifest.output_dir) / "usd" / "captured_cameras.json")
        entities.extend(captured)
        visit_path = build_visit_path(captured)
        if visit_path is not None:
            entities.append(visit_path)

        for entity in entities:
            author_usd_camera(stage_path, f"/World/Navigation/{_slugify(entity.name)}", entity)
        stage_path.GetRootLayer().Save()

        # The render path: user-authored walkthrough when captures exist,
        # otherwise a gentle orbit around the scene.
        path_entity = visit_path or self._default_orbit_path(center)
        self.camera_render_path.write_text(
            json.dumps(to_nerfstudio_camera_path(path_entity), indent=2), encoding="utf-8"
        )
        self.manifest.metadata["cameras"] = [entity.to_dict() for entity in entities]
        self.camera_plan_path.write_text(
            json.dumps({**bundle, "entities": self.manifest.metadata["cameras"]}, indent=2),
            encoding="utf-8",
        )
        preview_paths = self._render_camera_previews(bundle["allShots"])
        stage.metadata = {
            "cameraCount": len(entities),
            "capturedCount": len(captured),
            "renderPath": path_entity.name,
        }
        stage.message = (
            f"USD stage composed with {len(entities)} cameras "
            f"({len(captured)} captured); render path: {path_entity.name}."
        )

        # Guided mode pauses once when no user captures exist so the user can
        # author poses in the viewport before the path renders downstream.
        # Re-running with the metadata flag set finalises (whether captures
        # were added or the user chose to accept the presets as-is).
        if self.manifest.mode == "guided" and len(captured) == 0 and not already_paused:
            stage.metadata["pausedForUserInput"] = True
            stage.message = (
                f"Generated {len(entities)} default cameras. Capture poses in "
                "the viewport to add more, then re-run this step. Re-run as-is "
                "to accept the defaults."
            )
            stage.state = StageState.NEEDS_USER_INPUT.value
            self.log(stage.message)
            # Skip artifact registration: the USD stage is provisional until
            # the user finalises on the next invocation.
            return
        self._add_artifact(stage, "USD Stage", "usd", self.usd_stage_path, "Composed digital twin stage.")
        self._add_artifact(stage, "Camera Plan", "json", self.camera_plan_path, "Cameras and paths.")
        self._add_artifact(stage, "Render Camera Path", "json", self.camera_render_path, "ns-render camera path.")
        for index, preview in enumerate(preview_paths[:3], start=1):
            self._add_artifact(stage, f"Camera Preview {index}", "image", preview, "Generated camera preview.")

    def _scene_center(self) -> list[float]:
        """Robust centroid of the reconstruction (preview cloud percentiles)."""
        try:
            import numpy as np
            from plyfile import PlyData

            vertex = PlyData.read(str(self.recon_preview_ply_path))["vertex"]
            points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
            low, high = np.percentile(points, [5, 95], axis=0)
            return [float(v) for v in (low + high) / 2]
        except Exception:  # noqa: BLE001 - placeholder scenes have no preview
            return [0.0, 0.0, 0.0]

    def _gravity_align(self, stage: StageRecord) -> None:
        """Rotate cloud.ply + cloud_preview.ply so world +Y is up.

        Skips silently if the strict-mode build flag is off and the rotation
        fails (alignment is a polish step, not a correctness gate).
        """
        from .gravity_align import align_cloud

        summary_path = self.recon_dir / "summary.json"
        try:
            result = align_cloud(
                self.recon_ply_path,
                self.recon_preview_ply_path,
                summary_path=summary_path,
                captured_cameras_path=self.usd_dir / "captured_cameras.json",
            )
        except Exception as exc:  # noqa: BLE001 - alignment is best-effort
            if self.strict_mode:
                raise
            self.log(f"Gravity alignment skipped: {exc}")
            return
        if result is None:
            self.log("Gravity alignment skipped (cloud already aligned).")
            stage.metadata["gravity_aligned"] = True
            return
        stage.metadata["gravity_aligned"] = True
        stage.metadata["alignment_tilt_degrees"] = round(result.angle_from_y_degrees, 2)
        self.log(
            "Gravity-aligned cloud: rotated "
            f"{result.angle_from_y_degrees:.1f}° to bring scene up to +Y "
            f"(skewness {result.skewness:+.2f}, flipped={result.flipped})."
        )

    def _write_packed_splat(self, stage: StageRecord) -> None:
        """Encode the splat PLY into the compact .splat format for fast loads.

        ~7x smaller than the equivalent PLY (32 bytes per gaussian vs the full
        ply row), and the viewer skips the PLY header parse entirely. Best
        effort: skips silently outside strict mode if the input isn't a 3DGS
        PLY (placeholder paths) or anything else goes sideways.
        """
        from .splat_io import is_gaussian_ply
        from .splat_packed import ply_to_splat

        if not is_gaussian_ply(self.recon_ply_path):
            return
        try:
            size = ply_to_splat(self.recon_ply_path, self.recon_splat_path)
        except Exception as exc:  # noqa: BLE001 - packing is opportunistic
            if self.strict_mode:
                raise
            self.log(f"Packed .splat skipped: {exc}")
            return
        stage.metadata["packedSplatBytes"] = size
        self.log(f"Packed .splat written: {size // 1_000_000} MB.")

    def _default_orbit_path(self, center: list[float], seconds: float = 12.0) -> CameraEntity:
        from .walk_patterns import SceneBounds, bounds_from_preview_ply, orbit

        try:
            bounds = bounds_from_preview_ply(self.recon_preview_ply_path)
            # Honour the caller's centroid: it already does the same percentile
            # math but may incorporate other heuristics.
            bounds = SceneBounds(center=tuple(float(value) for value in center), radius=bounds.radius)
        except Exception:  # noqa: BLE001 - placeholder scenes have no preview cloud
            bounds = SceneBounds(center=tuple(float(value) for value in center), radius=2.0)
        return orbit(bounds, seconds=seconds, name="Scene Orbit")

    def _run_cosmos_output(self, stage: StageRecord) -> None:
        self.cosmos_dir.mkdir(parents=True, exist_ok=True)
        self.deliverables_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = self.cosmos_dir / "cosmos_annotations.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "sourceStage": str(self.usd_stage_path),
                    "model": "cosmos-reason2 (placeholder-safe)",
                    "annotations": [
                        {"label": "environment", "path": "/World/Environment"},
                        {"label": "digital-twin", "path": "/World/DigitalTwin"},
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        transfer_path = self.cosmos_dir / "cosmos_transfer_notes.txt"
        transfer_path.write_text(
            "Placeholder-safe Cosmos Transfer notes.\n"
            f"Source stage: {self.usd_stage_path}\n",
            encoding="utf-8",
        )
        rendered_remotely = False
        if self.remote_runner is not None:
            try:
                rendered_remotely = self._run_remote_render(stage)
            except StageCancelledError:
                raise
            except CostDeniedError as exc:
                self.log(f"{exc} Falling back to the preview slideshow.")
            except Exception as exc:  # noqa: BLE001
                if self.strict_mode:
                    raise
                self.log(f"Remote walkthrough render failed, using preview slideshow: {exc}")

        if rendered_remotely:
            self.manifest.walkthrough_video = str(self.splat_walkthrough_path)
            self._add_artifact(
                stage, "Walkthrough Video", "video", self.splat_walkthrough_path,
                "Splat-rendered camera-path walkthrough.",
            )
            stage.message = "Cosmos artifacts written; splat walkthrough rendered along the camera path."
        else:
            self._build_walkthrough_video()
            self.manifest.walkthrough_video = str(self.walkthrough_path)
            self._add_artifact(stage, "Walkthrough Video", "video", self.walkthrough_path, "Final MP4 walkthrough.")
            stage.message = "Cosmos artifacts written and walkthrough video rendered."
        self._add_artifact(stage, "Cosmos Annotation", "json", annotation_path, "Reason model output.")
        self._add_artifact(stage, "Cosmos Transfer Notes", "text", transfer_path, "Transfer model notes.")

    def _run_remote_render(self, stage: StageRecord) -> bool:
        """Render the authored camera path with the trained splat (HF Job).

        Needs the recon stage's checkpoint bundle in the artifact dataset
        (model.zip + processed_min.zip, produced by remote reconstructions
        from M2 onward) and the camera_path.json authored by camera staging.
        """
        remote_out = self.root / "reconstruction" / "remote_out"
        bundle_ok = (remote_out / "model.zip").exists() and (remote_out / "processed_min.zip").exists()
        if not bundle_ok or not self.camera_render_path.exists():
            self.log(
                "No render bundle for this job (model.zip + processed_min.zip + camera_path.json) — "
                "re-run reconstruction to bank one. Using the preview slideshow."
            )
            return False
        runner_config = getattr(self.remote_runner, "config", None)
        image_name = getattr(runner_config, "worker_image", "") if runner_config else ""
        if not image_name or image_name.startswith("python:"):
            raise RuntimeError("Remote worker image not configured.")

        dataset_prefix = f"jobs/{self.manifest.job_id}/reconstruction/out"
        ctx = StageContext(
            job_dir=self.root,
            job_id=self.manifest.job_id,
            stage_key="walkthrough_render",
            params={
                "image": image_name,
                "image_has_hub": True,
                "flavor": "l4x1",
                "est_minutes": 8,
                "timeout_seconds": 2400,
                "command": ["python", "/opt/vw/render_entrypoint.py"],
                "extra_repo_inputs": [
                    f"{dataset_prefix}/model.zip",
                    f"{dataset_prefix}/processed_min.zip",
                ],
            },
            inputs=[self.camera_render_path],
            expected_outputs=[self.splat_walkthrough_path],
            log=self.log,
            cancel=self.cancel_token,
        )
        result = self.remote_runner.run(ctx)
        if result.metadata:
            record_spend(self.manifest, "cosmos_output", result.metadata)
        return self.splat_walkthrough_path.exists()

    def _render_camera_previews(self, shots: list[dict]) -> list[Path]:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return []
        paths: list[Path] = []
        palette = {
            "background": "#F5F5DC",
            "surface": "#FDFDFD",
            "accent": "#006994",
            "accent_alt": "#D4AF37",
            "text": "#222222",
        }
        for index, shot in enumerate(shots, start=1):
            image = Image.new("RGB", (1280, 720), palette["background"])
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((48, 48, 1232, 672), radius=28, fill=palette["surface"], outline=palette["accent"], width=4)
            draw.rectangle((88, 116, 1188, 520), fill=palette["accent"])
            draw.rectangle((124, 152, 1152, 484), fill=palette["accent_alt"])
            draw.text((88, 72), f"{index:02d}. {shot['name']}", fill=palette["text"])
            draw.text((88, 540), shot["description"], fill=palette["text"])
            draw.text((88, 600), f"Source: {shot['source']} | Position: {tuple(shot['position'])}", fill=palette["text"])
            path = self.cameras_dir / f"shot_{index:02d}.png"
            image.save(path)
            paths.append(path)
        return paths

    def _build_walkthrough_video(self) -> None:
        ffmpeg = resolve_binary("ffmpeg")
        preview_paths = sorted(self.cameras_dir.glob("shot_*.png"))
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to render the final walkthrough video.")
        if not preview_paths:
            raise RuntimeError("Camera previews must exist before rendering the walkthrough video.")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            "1",
            "-i",
            str(self.cameras_dir / "shot_%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(self.walkthrough_path),
        ]
        self._run_command(cmd, "Walkthrough render failed.", timeout_seconds=1800)

    def _find_gsplat_config(self) -> "Path | None":
        """Return the most recently modified config.yml under gsplat_outputs, or None."""
        gsplat_out = self.recon_dir / "gsplat_outputs"
        if not gsplat_out.exists():
            return None
        configs = sorted(gsplat_out.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
        return configs[0] if configs else None

    def _export_gsplat_ply(self) -> "Path | None":
        """Run ns-export gaussian-splat on the trained model and return the PLY path, or None."""
        ns_export = resolve_binary("ns-export")
        if ns_export is None:
            self.log("ns-export not found; skipping gsplat PLY export.")
            return None
        config = self._find_gsplat_config()
        if config is None:
            self.log("No splatfacto config.yml found in gsplat_outputs; skipping PLY export.")
            return None
        export_dir = self.recon_dir / "gsplat_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            ns_export,
            "gaussian-splat",
            "--load-config", str(config),
            "--output-dir", str(export_dir),
        ]
        try:
            self._run_command(cmd, "ns-export gaussian-splat failed.", timeout_seconds=600)
        except RuntimeError as exc:
            self.log(f"gsplat PLY export failed: {exc}")
            return None
        candidates = sorted(export_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _convert_ply_to_cloud_files(self, ply_source: Path) -> bool:
        """Read a gsplat PLY with open3d and write cloud.ply + cloud.usda. Returns True on success."""
        try:
            import open3d as o3d
        except ImportError:
            self.log("open3d not available; cannot convert PLY.")
            return False
        try:
            pcd = o3d.io.read_point_cloud(str(ply_source))
        except Exception as exc:
            self.log(f"open3d failed to read {ply_source}: {exc}")
            return False
        if len(pcd.points) == 0:
            self.log(f"Loaded PLY has no points: {ply_source}")
            return False
        o3d.io.write_point_cloud(str(self.recon_ply_path), pcd)
        self.log(f"Wrote {len(pcd.points)} points to {self.recon_ply_path}")
        self._write_usd_from_point_cloud(pcd)
        return True

    def _write_usd_from_point_cloud(self, pcd) -> None:
        """Write cloud.usda populated with real geometry from an open3d PointCloud."""
        stage = Usd.Stage.CreateNew(str(self.recon_stage_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        pts_prim = UsdGeom.Points.Define(stage, "/World/Reconstruction")
        pts_vec = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pcd.points]
        pts_prim.GetPointsAttr().Set(pts_vec)
        pts_prim.GetWidthsAttr().Set([0.02] * len(pts_vec))
        if pcd.has_colors():
            pts_prim.GetDisplayColorAttr().Set(
                [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in pcd.colors]
            )
        stage.GetRootLayer().Save()
        self.log(f"Wrote USD stage with {len(pts_vec)} real points to {self.recon_stage_path}")

    def _write_placeholder_reconstruction(self) -> None:
        stage = Usd.Stage.CreateNew(str(self.recon_stage_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        points = UsdGeom.Points.Define(stage, "/World/Reconstruction")
        points.GetPointsAttr().Set(
            [Gf.Vec3f(-0.5, 0.0, 0.0), Gf.Vec3f(0.0, 0.5, 0.0), Gf.Vec3f(0.5, 0.0, 0.0)]
        )
        points.GetWidthsAttr().Set([0.05, 0.05, 0.05])
        stage.GetRootLayer().Save()

    def _write_placeholder_ply(self) -> None:
        self.recon_ply_path.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 4",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0.0 0.0 0.0",
                    "1.0 0.0 0.0",
                    "0.0 1.0 0.0",
                    "0.0 0.0 1.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def cancel(self) -> None:
        """Request cancellation of the currently running stage command."""
        self.cancel_token.cancel()

    def _run_command(self, cmd: list[str], error_message: str, timeout_seconds: int, env: dict[str, str] | None = None) -> None:
        self.local_runner.run_command(
            cmd,
            error_message=error_message,
            timeout_seconds=timeout_seconds,
            log=self.log,
            env=env,
            cancel=self.cancel_token,
        )
