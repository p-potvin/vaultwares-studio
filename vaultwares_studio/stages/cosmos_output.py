"""Cosmos stage: annotate the reconstruction, then render the walkthrough.

The last stage. A vision-language model (Cosmos Reason, or whichever backend
is reachable — see ``cosmos_reason``) looks at the frames the capture cameras
actually saw, names what is in the scene, and each observation is anchored in
3D through DA3's depth. The results land in ``cosmos/cosmos_annotations.json``
and as ``/World/Annotations/<slug>`` prims in the USD stage, which is what the
robot lab needs to turn a label into a navigation goal.

Provider and view count come from the job manifest:

    metadata["cosmos"] = {"provider": "nvidia"|"ollama"|"none",
                          "model": "...", "views": 12}

The walkthrough render is unchanged: the trained checkpoint renders the
authored camera path remotely when a bundle exists, otherwise the preview
slideshow stands in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import DigitalTwinStudioRunner, StageRecord

DEFAULT_COSMOS = {"provider": "nvidia", "model": "", "views": 12, "per_view_limit": 8}


def run(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    from ..runners import CostDeniedError, StageCancelledError

    ctx.cosmos_dir.mkdir(parents=True, exist_ok=True)
    ctx.deliverables_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = _run_reason(ctx, stage)

    rendered_remotely = False
    if ctx.remote_runner is not None:
        try:
            rendered_remotely = _run_remote_render(ctx, stage)
        except StageCancelledError:
            raise
        except CostDeniedError as exc:
            ctx.log(f"{exc} Falling back to the preview slideshow.")
        except Exception as exc:  # noqa: BLE001
            if ctx.strict_mode:
                raise
            ctx.log(f"Remote walkthrough render failed, using preview slideshow: {exc}")

    if rendered_remotely:
        ctx.manifest.walkthrough_video = str(ctx.splat_walkthrough_path)
        ctx._add_artifact(
            stage, "Walkthrough Video", "video", ctx.splat_walkthrough_path,
            "Splat-rendered camera-path walkthrough.",
        )
        stage.message = f"{stage.message} Splat walkthrough rendered along the camera path."
    else:
        _build_walkthrough_video(ctx)
        ctx.manifest.walkthrough_video = str(ctx.walkthrough_path)
        ctx._add_artifact(stage, "Walkthrough Video", "video", ctx.walkthrough_path, "Final MP4 walkthrough.")
    if annotation_path is not None:
        ctx._add_artifact(
            stage, "Cosmos Annotation", "json", annotation_path,
            "Scene objects with 3D anchors; ObjectNav goals.",
        )


def _run_reason(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> Path | None:
    """The annotation pass. Never fails the stage on a provider problem."""
    from ..camera_scene import compose_scene
    from ..capture_cameras import load_capture_cameras_json
    from ..cosmos_reason import annotate_job, build_provider, write_annotations

    settings = {**DEFAULT_COSMOS, **(ctx.manifest.metadata.get("cosmos") or {})}
    job_dir = Path(ctx.manifest.output_dir)
    try:
        provider = build_provider(settings["provider"], settings.get("model", ""))
    except Exception as exc:  # noqa: BLE001 - a missing key is not a pipeline failure
        if ctx.strict_mode:
            raise
        ctx.log(f"Cosmos provider unavailable ({exc}); annotating with no model.")
        provider = build_provider("none")

    try:
        payload = annotate_job(
            job_dir, provider,
            views=int(settings.get("views", 12)),
            per_view_limit=int(settings.get("per_view_limit", 8)),
            log=ctx.log,
        )
    except Exception as exc:  # noqa: BLE001
        if ctx.strict_mode:
            raise
        ctx.log(f"Cosmos Reason pass failed: {exc}")
        stage.metadata["cosmosError"] = str(exc)[:300]
        stage.message = "Cosmos annotation skipped; walkthrough only."
        return None

    path = write_annotations(job_dir, payload)
    annotations = payload["annotations"]
    stage.metadata.update({
        "cosmosProvider": payload.get("provider"),
        "cosmosModel": payload.get("model"),
        "annotations": len(annotations),
        "anchored": sum(1 for a in annotations if a.get("position")),
        "observations": payload["stats"]["observations"],
        "cosmosSeconds": payload["stats"]["seconds"],
        "anchoring": payload["source"]["anchoring"],
    })
    labels = ", ".join(a["label"] for a in annotations[:6])
    stage.message = (
        f"Cosmos Reason ({payload.get('model') or 'no model'}) found {len(annotations)} objects, "
        f"{stage.metadata['anchored']} anchored in 3D"
        + (f": {labels}." if labels else ".")
    )
    ctx.log(stage.message)

    # Re-compose so the annotations land in the stage beside the cameras.
    mesh = ctx.recon_dir / "mesh.usda"
    if ctx.usd_stage_path.exists():
        from ..camera_paths import CameraEntity

        entities = [CameraEntity.from_dict(entity) for entity in ctx.manifest.metadata.get("cameras", [])]
        compose_scene(
            ctx.usd_stage_path, ctx.recon_stage_path, entities,
            capture_frames=load_capture_cameras_json(job_dir),
            mesh=mesh if mesh.exists() else None,
            annotations=annotations,
        )
    return path


def _run_remote_render(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> bool:
    """Render the authored camera path with the trained splat (HF Job).

    Needs the recon stage's checkpoint bundle in the artifact dataset
    (model.zip + processed_min.zip, produced by remote reconstructions
    from M2 onward) and the camera_path.json authored by camera staging.
    """
    from ..runners import StageContext, record_spend

    remote_out = ctx.root / "reconstruction" / "remote_out"
    bundle_ok = (remote_out / "model.zip").exists() and (remote_out / "processed_min.zip").exists()
    if not bundle_ok or not ctx.camera_render_path.exists():
        ctx.log(
            "No render bundle for this job (model.zip + processed_min.zip + camera_path.json) — "
            "re-run reconstruction to bank one. Using the preview slideshow."
        )
        return False
    runner_config = getattr(ctx.remote_runner, "config", None)
    image_name = getattr(runner_config, "worker_image", "") if runner_config else ""
    if not image_name or image_name.startswith("python:"):
        raise RuntimeError("Remote worker image not configured.")

    dataset_prefix = f"jobs/{ctx.manifest.job_id}/reconstruction/out"
    ctx_obj = StageContext(
        job_dir=ctx.root,
        job_id=ctx.manifest.job_id,
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
        inputs=[ctx.camera_render_path],
        expected_outputs=[ctx.splat_walkthrough_path],
        log=ctx.log,
        cancel=ctx.cancel_token,
    )
    result = ctx.remote_runner.run(ctx_obj)
    if result.metadata:
        record_spend(ctx.manifest, "cosmos_output", result.metadata)
    return ctx.splat_walkthrough_path.exists()


def _build_walkthrough_video(ctx: "DigitalTwinStudioRunner") -> None:
    from ..pipeline import resolve_binary

    ffmpeg = resolve_binary("ffmpeg")
    preview_paths = sorted(ctx.cameras_dir.glob("shot_*.png"))
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
        str(ctx.cameras_dir / "shot_%02d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(ctx.walkthrough_path),
    ]
    ctx._run_command(cmd, "Walkthrough render failed.", timeout_seconds=1800)
