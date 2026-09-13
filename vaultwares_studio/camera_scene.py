"""Persist the same camera path for the viewport, USD and remote rendering.

Scene/viewer coordinates include the optional post-training gravity rotation.
Nerfstudio rendering uses the unrotated model coordinates. Raw input camera
poses also require the trainer's dataparser transform and scale before replay.
"""
from __future__ import annotations

import json
import os
import zipfile
import uuid
from pathlib import Path

import numpy as np

from .camera_paths import CameraEntity, author_usd_camera, to_nerfstudio_camera_path


def gravity_rotation(job_dir: Path) -> np.ndarray:
    summary = job_dir / "reconstruction" / "summary.json"
    if not summary.exists():
        return np.eye(3)
    data = json.loads(summary.read_text(encoding="utf-8-sig"))
    if not data.get("gravity_aligned"):
        return np.eye(3)
    rotation = np.asarray(data["alignment"]["rotation"], dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Invalid recorded scene gravity rotation.")
    return rotation


def load_active_camera(job_dir: Path) -> CameraEntity | None:
    path = job_dir / "usd" / "active_camera.json"
    return CameraEntity.from_dict(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None


def write_render_path(job_dir: Path, entity: CameraEntity) -> Path:
    """Convert display-world poses back to the checkpoint's frame for ns-render."""
    document = to_nerfstudio_camera_path(entity)
    world_to_model = np.eye(4)
    world_to_model[:3, :3] = gravity_rotation(job_dir).T
    for frame in document["camera_path"]:
        matrix = np.asarray(frame["camera_to_world"]).reshape(4, 4)
        frame["camera_to_world"] = (world_to_model @ matrix).flatten().tolist()
    path = job_dir / "usd" / "camera_path.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def compose_scene(
    path: Path,
    reconstruction: Path,
    cameras: list[CameraEntity],
    *,
    capture_frames: list | None = None,
    mesh: Path | None = None,
    annotations: list[dict] | None = None,
) -> None:
    """Create a portable USD root with real reconstruction and animated cameras.

    ``capture_frames`` (CaptureFrame list) authors the reconstruction's own
    cameras under /World/Capture; ``mesh`` references the fused surface layer
    under /World/DigitalTwin/Surface beside the splat; ``annotations`` (Cosmos
    Reason output) become named places under /World/Annotations.
    """
    from pxr import Sdf, Usd, UsdGeom

    path.parent.mkdir(parents=True, exist_ok=True)
    # A real layer anchors ../ references while composing. Write beside the
    # destination and replace after saving, keeping the previous scene intact
    # if authoring fails and allowing repeated saves in one process.
    temporary = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.usda")
    try:
        stage = Usd.Stage.CreateNew(str(temporary))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        # One numerical world unit; metric scale remains unknown until calibrated.
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        world.GetPrim().CreateAttribute("vw:metricScaleKnown", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        UsdGeom.Xform.Define(stage, "/World/Environment")
        twin = UsdGeom.Xform.Define(stage, "/World/DigitalTwin")
        relative = Path(os.path.relpath(reconstruction.resolve(), path.parent.resolve())).as_posix()
        if reconstruction.exists():
            twin.GetPrim().GetReferences().AddReference(relative)
        if mesh is not None and mesh.exists():
            surface = UsdGeom.Xform.Define(stage, "/World/DigitalTwin/Surface")
            surface.GetPrim().GetReferences().AddReference(
                Path(os.path.relpath(mesh.resolve(), path.parent.resolve())).as_posix()
            )
        for index, entity in enumerate(cameras):
            author_usd_camera(stage, f"/World/Navigation/Camera_{index + 1}", entity)
        if capture_frames:
            from .capture_cameras import author_capture_cameras

            author_capture_cameras(stage, capture_frames)
        if annotations:
            from .cosmos_reason import author_annotation_prims

            author_annotation_prims(stage, annotations)
        stage.GetRootLayer().Save()
        del stage
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_active_camera(job_dir: Path, entity: CameraEntity) -> None:
    """Persist a selected preset or the ordered captured path without a GPU job."""
    from .capture_cameras import load_capture_cameras_json

    usd_dir = job_dir / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    write_render_path(job_dir, entity)
    (usd_dir / "active_camera.json").write_text(json.dumps(entity.to_dict(), indent=2), encoding="utf-8")
    cloud = job_dir / "reconstruction" / "cloud.usda"
    # Re-composing for a new render path must not drop what staging authored.
    compose_scene(
        usd_dir / "digital_twin_scene.usda", cloud, [entity],
        capture_frames=load_capture_cameras_json(job_dir),
        mesh=job_dir / "reconstruction" / "mesh.usda",
        annotations=load_annotations(job_dir),
    )


def load_annotations(job_dir: Path) -> list[dict]:
    """Anchored Cosmos Reason annotations for this job, if the pass has run."""
    path = job_dir / "cosmos" / "cosmos_annotations.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("annotations", [])
    except (OSError, json.JSONDecodeError):
        return []


def _read_archived_json(archive: Path, filename: str) -> dict | None:
    if not archive.exists():
        return None
    with zipfile.ZipFile(archive) as source:
        matches = [n for n in source.namelist() if Path(n).name == filename]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {filename} in {archive.name}, found {len(matches)}.")
        return json.loads(source.read(matches[0]))


def scene_frame_transform(job_dir: Path) -> np.ndarray:
    """The 4x4 that maps reconstruction (SfM/DA3) world into the scene world.

    Same chain ``prepare_retrace_transforms`` applies to cameras, as one
    matrix — trainer normalisation (rotation, translation, uniform scale) then
    the gravity rotation — so a mesh fused in DA3 coordinates lands on the
    splat. Identity when the job carries no normalisation.
    """
    recon = job_dir / "reconstruction"
    normalization = None
    for path in [recon / "dataparser_transforms.json", recon / "remote_out" / "dataparser_transforms.json"]:
        if path.exists():
            normalization = json.loads(path.read_text(encoding="utf-8-sig"))
            break
    model = recon / "remote_out" / "model.zip"
    if normalization is None and model.exists():
        normalization = _read_archived_json(model, "dataparser_transforms.json")
    matrix = np.eye(4)
    if normalization:
        transform = np.eye(4)
        transform[:3, :] = np.asarray(normalization["transform"], dtype=float)[:3, :]
        scale = np.eye(4) * float(normalization["scale"])
        scale[3, 3] = 1.0
        matrix = scale @ transform
    gravity = np.eye(4)
    gravity[:3, :3] = gravity_rotation(job_dir)
    return gravity @ matrix


def prepare_retrace_transforms(job_dir: Path) -> Path | None:
    """Recover archived poses and map them into the splat viewer's world frame.

    Never modify the original transforms or model archives. A dataparser
    transform by itself is a normalization matrix, not a camera trajectory.
    """
    recon = job_dir / "reconstruction"
    data = None
    for path in [recon / "transforms.json", recon / "remote_out" / "transforms.json"]:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            break
    if data is None:
        data = _read_archived_json(recon / "remote_out" / "processed_min.zip", "transforms.json")
    if data is None or not data.get("frames"):
        return None
    normalization = None
    for path in [recon / "dataparser_transforms.json", recon / "remote_out" / "dataparser_transforms.json"]:
        if path.exists():
            normalization = json.loads(path.read_text(encoding="utf-8-sig"))
            break
    model = recon / "remote_out" / "model.zip"
    if normalization is None and model.exists():
        normalization = _read_archived_json(model, "dataparser_transforms.json")
    # Direct DA3 output can be in the original frame. Trained jobs must provide
    # normalization rather than silently replaying raw SfM coordinates.
    transform = np.eye(4)
    scale = 1.0
    if normalization:
        transform[:3, :] = np.asarray(normalization["transform"], dtype=float)[:3, :]
        scale = float(normalization["scale"])
    rotation = gravity_rotation(job_dir)
    for frame in data["frames"]:
        matrix = np.eye(4)
        raw = np.asarray(frame["transform_matrix"], dtype=float)
        matrix[:3, :] = raw[:3, :]
        matrix = transform @ matrix
        matrix[:3, 3] *= scale
        matrix[:3, :] = rotation @ matrix[:3, :]
        frame["transform_matrix"] = matrix.tolist()
    target = job_dir / "usd" / "retrace_transforms.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target
