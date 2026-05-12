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
COLMAP_CANDIDATE_PATHS.append(Path(r"C:\Users\Administrator\Desktop\COLMAP\bin\colmap.exe"))
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

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["stages"] = [stage.to_dict() for stage in self.stages]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "JobManifest":
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
        )


@dataclass(frozen=True)
class StageDefinition:
    key: str
    title: str
    description: str


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
    ),
    StageDefinition(
        "usd_cameras",
        "USD + Cameras",
        "Compose the USD stage and generate preset plus prompt-directed cameras.",
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
        StageRecord(key=stage.key, title=stage.title, description=stage.description)
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


class DigitalTwinStudioRunner:
    def __init__(self, manifest: JobManifest, log: Callable[[str], None], strict_mode: bool = False):
        self.manifest = manifest
        self.log = log
        self.strict_mode = strict_mode
        self.root = Path(manifest.output_dir)
        self.frames_dir = self.root / "frames"
        self.recon_dir = self.root / "reconstruction"
        self.usd_dir = self.root / "usd"
        self.cameras_dir = self.root / "camera_previews"
        self.cosmos_dir = self.root / "cosmos"
        self.deliverables_dir = self.root / "deliverables"
        self.recon_stage_path = self.recon_dir / "cloud.usda"
        self.recon_ply_path = self.recon_dir / "cloud.ply"
        self.usd_stage_path = self.usd_dir / "digital_twin_scene.usda"
        self.camera_plan_path = self.usd_dir / "camera_plan.json"
        self.walkthrough_path = self.deliverables_dir / "digital_twin_walkthrough.mp4"

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
            "usd_cameras": self._run_usd_cameras,
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
            self._transition(stage, StageState.COMPLETE, stage.message or f"{stage.title} complete.")
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
        for stale in self.frames_dir.glob("*.png"):
            stale.unlink(missing_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            self.manifest.source_video,
            "-vf",
            "fps=2",
            "-q:v",
            "2",
            str(self.frames_dir / "frame_%04d.png"),
        ]
        self._run_command(cmd, "Frame extraction failed.", timeout_seconds=1800)
        frame_paths = sorted(self.frames_dir.glob("*.png"))
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
        stage.metadata = {"frameCount": len(frame_paths)}
        stage.message = f"Extracted {len(frame_paths)} frames."
        self._add_artifact(stage, "Frames Manifest", "json", preview_manifest, "Frame extraction summary.")
        for index, frame in enumerate(frame_paths[:3], start=1):
            self._add_artifact(stage, f"Frame Preview {index}", "image", frame, "Sample extracted frame.")

    def _run_reconstruction(self, stage: StageRecord) -> None:
        self.recon_dir.mkdir(parents=True, exist_ok=True)
        ns_process_data = resolve_binary("ns-process-data")
        colmap_bin = resolve_binary("colmap")
        ns_train = resolve_binary("ns-train")
        if not any(self.frames_dir.glob("*.png")):
            raise RuntimeError("Frame extraction must complete before reconstruction.")
        
        run_env = os.environ.copy()
        run_env["PYTHONUTF8"] = "1"
        if colmap_bin:
            colmap_dir = str(Path(colmap_bin).parent.resolve())
            run_env["PATH"] = f"{colmap_dir}{os.pathsep}{run_env.get('PATH', '')}"

        degraded = False
        if ns_process_data and colmap_bin:
            cmd = [
                ns_process_data,
                "images",
                "--data",
                str(self.frames_dir),
                "--output-dir",
                str(self.recon_dir),
                "--no-gpu",
            ]
            try:
                self._run_command(cmd, "Reconstruction failed.", timeout_seconds=3600, env=run_env)
            except RuntimeError:
                if self.strict_mode:
                    raise
                degraded = True
                self._write_placeholder_reconstruction()
        else:
            degraded = True
            self._write_placeholder_reconstruction()

        transforms_path = self.recon_dir / "transforms.json"
        if ns_train and transforms_path.exists():
            train_cmd = [
                ns_train,
                "splatfacto",
                "--data",
                str(self.recon_dir),
                "--output-dir",
                str(self.recon_dir / "gsplat_outputs"),
                "--max-num-iterations",
                "250",
                "--vis",
                "none",
            ]
            try:
                self._run_command(train_cmd, "gsplat training failed.", timeout_seconds=3600, env=run_env)
                exported_ply = self._export_gsplat_ply()
                if exported_ply is None or not self._convert_ply_to_cloud_files(exported_ply):
                    degraded = True
            except RuntimeError:
                if self.strict_mode:
                    raise
                degraded = True

        if not self.recon_stage_path.exists():
            self._write_placeholder_reconstruction()
            degraded = True
        if not self.recon_ply_path.exists():
            self._write_placeholder_ply()
            degraded = True

        stage.metadata = {"degraded": degraded}
        stage.message = (
            "Reconstruction completed with placeholder-safe outputs."
            if degraded
            else "Reconstruction completed with tool-backed outputs."
        )
        self._add_artifact(stage, "Reconstruction Stage", "usd", self.recon_stage_path, "Reconstruction stage.")
        self._add_artifact(stage, "Reconstruction PLY", "ply", self.recon_ply_path, "Point cloud output.")

    def _run_usd_cameras(self, stage: StageRecord) -> None:
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
        navigation = UsdGeom.Xform.Define(stage_path, "/World/Navigation")
        for shot in bundle["allShots"]:
            shot_path = f"/World/Navigation/{_slugify(shot['name'])}"
            camera = UsdGeom.Camera.Define(stage_path, shot_path)
            camera.AddTranslateOp().Set(Gf.Vec3f(*shot["position"]))
            camera.GetFocalLengthAttr().Set(24.0)
            camera.GetHorizontalApertureAttr().Set(36.0)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))
            camera.GetPrim().CreateAttribute("shotDescription", Sdf.ValueTypeNames.String, custom=True).Set(shot["description"])
            camera.GetPrim().CreateAttribute("shotSource", Sdf.ValueTypeNames.String, custom=True).Set(shot["source"])
        stage_path.GetRootLayer().Save()

        self.camera_plan_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        preview_paths = self._render_camera_previews(bundle["allShots"])
        stage.metadata = {"cameraCount": len(bundle["allShots"])}
        stage.message = f"USD stage composed with {len(bundle['allShots'])} camera shots."
        self._add_artifact(stage, "USD Stage", "usd", self.usd_stage_path, "Composed digital twin stage.")
        self._add_artifact(stage, "Camera Plan", "json", self.camera_plan_path, "Preset and prompt camera plan.")
        for index, preview in enumerate(preview_paths[:3], start=1):
            self._add_artifact(stage, f"Camera Preview {index}", "image", preview, "Generated camera preview.")

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
        self._build_walkthrough_video()
        self.manifest.walkthrough_video = str(self.walkthrough_path)
        stage.message = "Cosmos artifacts written and walkthrough video rendered."
        self._add_artifact(stage, "Cosmos Annotation", "json", annotation_path, "Reason model output.")
        self._add_artifact(stage, "Cosmos Transfer Notes", "text", transfer_path, "Transfer model notes.")
        self._add_artifact(stage, "Walkthrough Video", "video", self.walkthrough_path, "Final MP4 walkthrough.")

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

    def _run_command(self, cmd: list[str], error_message: str, timeout_seconds: int, env: dict[str, str] | None = None) -> None:
        self.log(f"Running: {' '.join(cmd)}")
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
        )
        if completed.stdout:
            self.log(completed.stdout.strip())
        if completed.stderr:
            self.log(completed.stderr.strip())
        if completed.returncode != 0:
            raise RuntimeError(f"{error_message} (exit code={completed.returncode})")
