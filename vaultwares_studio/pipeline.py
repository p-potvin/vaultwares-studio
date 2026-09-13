from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .runners import (
    CancelToken,
    LocalStageRunner,
    StageRunner,
)

MANIFEST_SCHEMA_VERSION = 2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
EXTERNAL_JOBS_DIR = Path("D:/vaultwares-studio-jobs/data/jobs")
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
    python_deps = ["PySide6", "qfluentwidgets", "redis", "pxr", "open3d", "PIL", "torch", "depth_anything_3"]
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
        execution_profile=os.environ.get("VW_EXECUTION_PROFILE", "Auto-detect"),
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
    roots = [Path(jobs_dir)]
    if Path(jobs_dir).resolve() == JOBS_DIR.resolve() and EXTERNAL_JOBS_DIR.exists():
        roots.append(EXTERNAL_JOBS_DIR)
    manifests = [path for root in roots if root.exists() for path in root.glob("*/manifest.json") if path.is_file()]
    unique: list[Path] = []
    seen_ids: set[str] = set()
    for path in sorted(manifests, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            job_id = json.loads(path.read_text(encoding="utf-8"))["job_id"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            job_id = str(path.resolve())
        if len(roots) > 1 and job_id in seen_ids:
            continue
        seen_ids.add(job_id)
        unique.append(path)
    return unique


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
FRAME_KEEP_TARGET = 500
FRAME_EXTRACT_TARGET = 1000
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
        from .stages import video_intake
        video_intake.run(self, stage)

    def _run_frame_extraction(self, stage: StageRecord) -> None:
        from .stages import frame_extraction
        frame_extraction.run(self, stage)

    def _extract_extra_video(self, video_path: str, prefix: str, ffmpeg: str) -> None:
        from .stages import frame_extraction
        frame_extraction.extract_extra_video(self, video_path, prefix, ffmpeg)

    def _merge_refine_base_frames(self, base_zip: Path) -> None:
        from .stages import frame_extraction
        frame_extraction.merge_refine_base_frames(self, base_zip)

    def _run_reconstruction(self, stage: StageRecord) -> None:
        from .stages import reconstruction
        reconstruction.run(self, stage)

    def _run_split_remote_reconstruction(
        self,
        stage: StageRecord,
        preset,
        resume_job_id: str | None = None,
    ) -> Path | None:
        from .stages import reconstruction
        return reconstruction._run_split_remote(self, stage, preset, resume_job_id=resume_job_id)

    def _run_remote_reconstruction(
        self,
        stage: StageRecord,
        preset,
        resume_job_id: str | None = None,
    ) -> Path | None:
        from .stages import reconstruction
        return reconstruction._run_remote(self, stage, preset, resume_job_id=resume_job_id)

    def _run_local_reconstruction(self, stage: StageRecord) -> Path | None:
        from .stages import reconstruction
        return reconstruction._run_local(self, stage)

    def _run_camera_staging(self, stage: StageRecord) -> None:
        from .stages import camera_staging
        camera_staging.run(self, stage)

    def _scene_center(self) -> list[float]:
        from .stages import camera_staging
        return camera_staging._scene_center(self)

    def _gravity_align(self, stage: StageRecord) -> None:
        from .stages import reconstruction
        reconstruction._gravity_align(self, stage)

    def _write_packed_splat(self, stage: StageRecord) -> None:
        from .stages import reconstruction
        reconstruction._write_packed_splat(self, stage)

    def _default_orbit_path(self, center: list[float], seconds: float = 12.0) -> CameraEntity:
        from .stages import camera_staging
        return camera_staging._default_orbit_path(self, center, seconds=seconds)

    def _run_cosmos_output(self, stage: StageRecord) -> None:
        from .stages import cosmos_output
        cosmos_output.run(self, stage)

    def _run_remote_render(self, stage: StageRecord) -> bool:
        from .stages import cosmos_output
        return cosmos_output._run_remote_render(self, stage)

    def _render_camera_previews(self, shots: list[dict]) -> list[Path]:
        from .stages import camera_staging
        return camera_staging._render_camera_previews(self, shots)

    def _build_walkthrough_video(self) -> None:
        from .stages import cosmos_output
        cosmos_output._build_walkthrough_video(self)

    def _find_gsplat_config(self) -> "Path | None":
        from .stages import reconstruction
        return reconstruction._find_gsplat_config(self)

    def _export_gsplat_ply(self) -> "Path | None":
        from .stages import reconstruction
        return reconstruction._export_gsplat_ply(self)

    def _convert_ply_to_cloud_files(self, ply_source: Path) -> bool:
        from .stages import reconstruction
        return reconstruction._convert_ply_to_cloud_files(self, ply_source)

    def _write_usd_from_point_cloud(self, pcd) -> None:
        from .stages import reconstruction
        reconstruction._write_usd_from_point_cloud(self, pcd)

    def _write_placeholder_reconstruction(self) -> None:
        from .stages import reconstruction
        reconstruction._write_placeholder_reconstruction(self)

    def _write_placeholder_ply(self) -> None:
        from .stages import reconstruction
        reconstruction._write_placeholder_ply(self)

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
