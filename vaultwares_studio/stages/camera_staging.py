"""Camera staging stage: the reconstruction's cameras, the render path, the USD stage.

Second-to-last stage, the one Cosmos Reason reads from. It composes the scene
from three sources of cameras:

- **capture**: every registered source frame with its pose and intrinsics,
  mapped into the viewer world (``capture_cameras``). Authored under
  /World/Capture and written to ``usd/capture_cameras.json`` for the hand-off.
- **authored**: presets and prompt-derived shots (scaled to the scene bounds
  rather than a fixed 5 m room), plus whatever the user captured or chose in
  the viewport.
- **render path**: the active viewport path if one was saved, else the
  captured walkthrough, else a *retrace* of the real trajectory when poses
  exist, else the orbit.

If a fused surface (``reconstruction/mesh.usda``) exists it is referenced
beside the splat so the stage carries structure as well as appearance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import DigitalTwinStudioRunner, StageRecord

# The preset shots were authored for a ~5 m room centred near (0, 1.1, 0);
# this is the radius that room implied, so shots scale with the real scene.
_PRESET_ROOM_RADIUS = 4.0
_PRESET_ROOM_CENTER = (0.0, 1.1, 0.0)


def run(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    from ..camera_director import build_camera_bundle
    from ..camera_scene import compose_scene, load_active_camera, write_render_path
    from ..camera_paths import (
        CameraEntity,
        CameraKeyframe,
        build_visit_path,
        load_captured_entities,
    )
    from ..capture_cameras import load_capture_frames, trajectory_stats, write_capture_cameras_json
    from ..pipeline import DEFAULT_CAMERA_PROMPT
    from ..pipeline import StageState

    already_paused = bool(stage.metadata.get("pausedForUserInput"))
    ctx.usd_dir.mkdir(parents=True, exist_ok=True)
    ctx.cameras_dir.mkdir(parents=True, exist_ok=True)
    job_dir = Path(ctx.manifest.output_dir)
    bundle = build_camera_bundle(str(ctx.manifest.metadata.get("cameraPrompt", DEFAULT_CAMERA_PROMPT)))

    bounds = _scene_bounds(ctx)
    center = list(bounds.center)
    entities: list[CameraEntity] = []
    for shot in bundle["allShots"]:
        entities.append(
            CameraEntity(
                name=shot["name"],
                source=shot["source"],
                keyframes=[
                    CameraKeyframe(
                        t=0.0,
                        position=_scale_preset_position(shot["position"], bounds),
                        look_at=list(center),
                    )
                ],
            )
        )
    captured = load_captured_entities(job_dir / "usd" / "captured_cameras.json")
    entities.extend(captured)
    visit_path = build_visit_path(captured)
    if visit_path is not None:
        entities.append(visit_path)

    # The real cameras.
    duration = _source_duration(ctx)
    capture = load_capture_frames(job_dir, duration=duration)
    capture_json: Path | None = None
    if capture:
        capture_json = write_capture_cameras_json(
            job_dir, capture,
            extra={"job_id": ctx.manifest.job_id, "source_video": ctx.manifest.source_video,
                   "loop_pairs": ctx.manifest.metadata.get("loop_pairs")},
        )
        ctx.log(f"Capture cameras: {len(capture)} registered frames -> {capture_json.name}")

    active = load_active_camera(job_dir)
    retrace = _retrace_path(ctx, bounds, capture, duration) if len(capture) >= 2 else None
    path_entity = active or visit_path or retrace or _default_orbit_path(ctx, center)
    if active and visit_path:
        entities.remove(visit_path)
    if path_entity not in entities:
        entities.append(path_entity)

    mesh = ctx.recon_dir / "mesh.usda"
    compose_scene(
        ctx.usd_stage_path, ctx.recon_stage_path, entities,
        capture_frames=capture, mesh=mesh if mesh.exists() else None,
    )
    write_render_path(job_dir, path_entity)
    ctx.manifest.metadata["cameras"] = [entity.to_dict() for entity in entities]
    ctx.camera_plan_path.write_text(
        json.dumps({**bundle, "entities": ctx.manifest.metadata["cameras"]}, indent=2),
        encoding="utf-8",
    )
    preview_paths = _render_camera_previews(ctx, bundle["allShots"])
    stage.metadata = {
        "cameraCount": len(entities),
        "capturedCount": len(captured),
        "captureFrames": len(capture),
        "renderPath": path_entity.name,
        "meshReferenced": mesh.exists(),
        "sceneRadius": round(bounds.radius, 4),
    }
    if capture:
        stage.metadata["trajectory"] = trajectory_stats(capture)
    stage.message = (
        f"USD stage composed with {len(entities)} cameras "
        f"({len(captured)} captured, {len(capture)} capture frames); render path: {path_entity.name}."
    )

    if ctx.manifest.mode == "guided" and len(captured) == 0 and active is None and not already_paused:
        stage.metadata["pausedForUserInput"] = True
        stage.message = (
            f"Generated {len(entities)} default cameras. Capture poses in "
            "the viewport to add more, then re-run this step. Re-run as-is "
            "to accept the defaults."
        )
        stage.state = StageState.NEEDS_USER_INPUT.value
        ctx.log(stage.message)
        return
    ctx._add_artifact(stage, "USD Stage", "usd", ctx.usd_stage_path, "Composed digital twin stage.")
    ctx._add_artifact(stage, "Camera Plan", "json", ctx.camera_plan_path, "Cameras and paths.")
    ctx._add_artifact(stage, "Render Camera Path", "json", ctx.camera_render_path, "ns-render camera path.")
    if capture_json is not None:
        ctx._add_artifact(
            stage, "Capture Cameras", "json", capture_json,
            "Every registered frame: pose, intrinsics, time. The Cosmos hand-off.",
        )
    if mesh.exists():
        ctx._add_artifact(stage, "Fused Surface", "usd", mesh, "TSDF mesh from the DA3 depth fields.")
    for index, preview in enumerate(preview_paths[:3], start=1):
        ctx._add_artifact(stage, f"Camera Preview {index}", "image", preview, "Generated camera preview.")


def _source_duration(ctx: "DigitalTwinStudioRunner") -> float | None:
    """Seconds of source video, from intake or an imported console manifest."""
    try:
        return float(ctx.stage_for("video_intake").metadata["probe"]["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        pass
    for candidate in (
        Path(ctx.manifest.output_dir) / "input_metadata.json",
        Path(ctx.manifest.output_dir) / "reconstruction" / "remote_out" / "input_manifest.json",
    ):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return float(payload["probe"]["format"]["duration"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _scene_bounds(ctx: "DigitalTwinStudioRunner"):
    from ..walk_patterns import SceneBounds, bounds_from_preview_ply

    center = _scene_center(ctx)
    try:
        radius = bounds_from_preview_ply(ctx.recon_preview_ply_path).radius
    except Exception:  # noqa: BLE001 - placeholder scenes have no preview cloud
        radius = 2.0
    return SceneBounds(center=tuple(float(v) for v in center), radius=float(radius))


def _scale_preset_position(position, bounds) -> list[float]:
    """Room-sized preset coordinates -> this scene's centre and radius."""
    factor = bounds.radius / _PRESET_ROOM_RADIUS
    return [
        float(bounds.center[axis] + (float(position[axis]) - _PRESET_ROOM_CENTER[axis]) * factor)
        for axis in range(3)
    ]


def _retrace_path(ctx: "DigitalTwinStudioRunner", bounds, capture, duration):
    """Replay the capture trajectory as the default walkthrough.

    Keyframes are thinned to ~60 so the path stays a path and not 500 stops,
    and the replay is capped at 30 s of render — the source can be minutes.
    """
    from ..walk_patterns import retrace_steps

    transforms = Path(ctx.manifest.output_dir) / "usd" / "retrace_transforms.json"
    if not transforms.exists():
        return None
    try:
        seconds = min(30.0, float(duration)) if duration else 20.0
        return retrace_steps(
            bounds, transforms_json=transforms, stride=max(1, len(capture) // 60),
            seconds=seconds, name="Retrace Steps",
        )
    except Exception as exc:  # noqa: BLE001 - fall back to the orbit
        ctx.log(f"Retrace path unavailable: {exc}")
        return None


def _scene_center(ctx: "DigitalTwinStudioRunner") -> list[float]:
    """Robust centroid of the reconstruction (preview cloud percentiles)."""
    try:
        import numpy as np
        from plyfile import PlyData

        vertex = PlyData.read(str(ctx.recon_preview_ply_path))["vertex"]
        points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
        low, high = np.percentile(points, [5, 95], axis=0)
        return [float(v) for v in (low + high) / 2]
    except Exception:  # noqa: BLE001 - placeholder scenes have no preview
        return [0.0, 0.0, 0.0]


def _default_orbit_path(
    ctx: "DigitalTwinStudioRunner", center: list[float], seconds: float = 12.0
) -> "CameraEntity":
    from ..camera_paths import CameraEntity
    from ..walk_patterns import SceneBounds, bounds_from_preview_ply, orbit

    try:
        bounds = bounds_from_preview_ply(ctx.recon_preview_ply_path)
        bounds = SceneBounds(center=tuple(float(value) for value in center), radius=bounds.radius)
    except Exception:  # noqa: BLE001 - placeholder scenes have no preview cloud
        bounds = SceneBounds(center=tuple(float(value) for value in center), radius=2.0)
    return orbit(bounds, seconds=seconds, name="Scene Orbit")


def _render_camera_previews(ctx: "DigitalTwinStudioRunner", shots: list[dict]) -> list[Path]:
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
        path = ctx.cameras_dir / f"shot_{index:02d}.png"
        image.save(path)
        paths.append(path)
    return paths
