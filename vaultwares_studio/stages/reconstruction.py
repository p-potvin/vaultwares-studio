"""Reconstruction stage: SfM + Gaussian Splatting training.

Handles local, remote (single-job), and remote (split-job) reconstruction.
The two remote paths share setup via ``_prepare_remote`` to avoid the
~60% code duplication that existed when they were inline methods.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..presets import _flavor_label

if TYPE_CHECKING:
    from ..pipeline import DigitalTwinStudioRunner, StageRecord


@dataclass
class _RemotePrep:
    """Shared state prepared by ``_prepare_remote`` for both remote paths."""

    image_name: str
    frames_zip: Path
    frame_paths: list[Path]
    export_dir: Path
    splat_path: Path
    summary_path: Path
    remote_out_dir: Path
    bundle_model: Path
    bundle_processed: Path
    hf_job_id: str


def run(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    from ..pipeline import list_frames, record_spend, resolve_binary
    from ..presets import get_preset
    from ..runners import CostDeniedError, StageCancelledError
    from ..splat_io import convert_splat_outputs, is_gaussian_ply

    ctx.recon_dir.mkdir(parents=True, exist_ok=True)
    if not list_frames(ctx.frames_dir):
        raise RuntimeError("Frame extraction must complete before reconstruction.")
    preset = get_preset(ctx.manifest.metadata.get("preset"))
    exported_ply: Path | None = None
    resume_job_id: str | None = ctx.manifest.metadata.get("resume_job_id")

    if stage.placement == "remote" and ctx.remote_runner is not None:
        try:
            if preset.key == "da3-incremental":
                exported_ply = _run_da3_incremental(ctx, stage, preset, resume_job_id=resume_job_id)
            elif preset.da3_direct_gs:
                exported_ply = _run_da3_direct_gs(ctx, stage, preset, resume_job_id=resume_job_id)
            elif preset.split_jobs:
                exported_ply = _run_split_remote(ctx, stage, preset, resume_job_id=resume_job_id)
            else:
                exported_ply = _run_remote(ctx, stage, preset, resume_job_id=resume_job_id)
        except StageCancelledError:
            raise
        except CostDeniedError as exc:
            ctx.log(f"{exc} Using the local quick path instead.")
        except Exception as exc:  # noqa: BLE001 - incl. hub HTTP errors
            if ctx.strict_mode:
                raise
            ctx.log(f"Remote reconstruction failed, falling back to local path: {exc}")
    elif stage.placement == "remote":
        # No remote runner configured — try local DA3 if the preset uses it
        from ..presets import SfmMethod
        if preset.sfm_method == SfmMethod.DA3 and preset.da3_direct_gs:
            ctx.log(
                "Reconstruction prefers remote execution but no remote runner is "
                "configured. Running DA3 draft locally."
            )
            exported_ply = _run_local_da3(ctx, stage, preset)
        else:
            ctx.log(
                "Reconstruction prefers remote execution but no remote runner is "
                "configured (Settings > Remote Compute). Using the local quick path."
            )

    if exported_ply is None:
        exported_ply = _run_local(ctx, stage)

    degraded = True
    if exported_ply is not None and is_gaussian_ply(exported_ply):
        try:
            info = convert_splat_outputs(
                exported_ply,
                ctx.recon_ply_path,
                ctx.recon_preview_ply_path,
                ctx.recon_stage_path,
                ctx.log,
                keep_quantile=preset.splat_keep_quantile,
            )
            stage.metadata.update(info)
            degraded = False
        except Exception as exc:  # noqa: BLE001
            if ctx.strict_mode:
                raise
            ctx.log(f"Splat conversion failed: {exc}")
    elif exported_ply is not None:
        degraded = not _convert_ply_to_cloud_files(ctx, exported_ply)

    if not ctx.recon_stage_path.exists():
        _write_placeholder_reconstruction(ctx)
        degraded = True
    if not ctx.recon_ply_path.exists():
        _write_placeholder_ply(ctx)
        degraded = True

    if not degraded and ctx.recon_preview_ply_path.exists():
        _gravity_align(ctx, stage)
        if is_gaussian_ply(ctx.recon_ply_path):
            from ..splat_io import read_gaussian_ply, splat_to_usd
            # The viewer uses the rotated PLY; USD must use that same world.
            stage.metadata["usd_mode"] = splat_to_usd(
                read_gaussian_ply(ctx.recon_ply_path), ctx.recon_stage_path,
                source=ctx.recon_ply_path.name,
            )
    if not degraded and ctx.recon_ply_path.exists():
        _write_packed_splat(ctx, stage)

    stage.metadata["degraded"] = degraded
    stage.metadata["preset"] = preset.key
    stage.message = (
        "Reconstruction completed with placeholder-safe outputs."
        if degraded
        else f"Reconstruction completed ({stage.metadata.get('gaussians', '?')} gaussians, preset: {preset.key})."
    )
    ctx._add_artifact(stage, "Reconstruction Stage", "usd", ctx.recon_stage_path, "Reconstruction stage.")
    ctx._add_artifact(stage, "Reconstruction PLY", "ply", ctx.recon_ply_path, "Gaussian splat output.")
    if ctx.recon_preview_ply_path.exists():
        ctx._add_artifact(
            stage, "Preview Point Cloud", "ply", ctx.recon_preview_ply_path,
            "Decimated point cloud for the live viewer.",
        )


# ---------------------------------------------------------------------------
# DA3 direct 3DGS (single-job, no splatfacto training)
# ---------------------------------------------------------------------------

def _scheduling_timeout(ctx) -> float:
    """Seconds a flavor candidate may sit in SCHEDULING before the next is tried.

    120s suits an interactive run; unattended retries can afford to wait out a
    busy GPU pool, and SCHEDULING is not billed. Read from manifest metadata so
    every remote path honours the same override — the direct-GS path had it
    first and the split path silently ignored it, which cost a wasted launch.
    """
    return float(ctx.manifest.metadata.get("flavor_scheduling_timeout_seconds", 120.0))


def _resolve_da3_image(ctx, preset, prep) -> str:
    """Resolve the DA3 Space image URL, handling the case where the default
    worker_image is a placeholder (python:3.12)."""
    if not preset.sfm_image_override:
        return prep.image_name
    runner_config = getattr(ctx.remote_runner, "config", None)
    owner = ""
    if runner_config:
        owner = getattr(runner_config, "namespace", "") or ""
        if not owner and getattr(runner_config, "artifact_repo", ""):
            owner = runner_config.artifact_repo.split("/")[0]
    if not owner:
        try:
            from ..runners.hf_jobs import get_hf_token
            from huggingface_hub import HfApi
            api = HfApi(token=get_hf_token())
            owner = api.whoami()["name"]
        except Exception:
            owner = ""
    return preset.sfm_image_override.format(owner=owner)


def _run_da3_direct_gs(
    ctx: "DigitalTwinStudioRunner",
    stage: "StageRecord",
    preset,
    resume_job_id: str | None = None,
) -> Path | None:
    """Run DA3-GIANT with infer_gs=True for direct 3DGS output.

    Single remote job: DA3 predicts camera poses + 3D Gaussians in one pass.
    No COLMAP, no splatfacto training. The output is a 3DGS PLY ready for
    viewing. Quality is lower than trained splatfacto but the job completes
    in ~5 min vs ~30 min for a full training run.
    """
    from ..pipeline import record_spend
    from ..runners import StageContext

    prep = _prepare_remote(ctx, preset, resume_job_id=resume_job_id)

    gs_command = [
        "python", "/opt/vw/da3_entrypoint.py",
        "--gs-only",
        "--da3-model", preset.da3_model,
        "--max-sfm-frames", str(preset.da3_max_sfm_frames),
    ]

    da3_image = _resolve_da3_image(ctx, preset, prep)
    gs_ctx = StageContext(
        job_dir=ctx.root,
        job_id=prep.hf_job_id,
        stage_key="reconstruction",
        params={
            "image": da3_image,
            "image_has_hub": True,
            "flavor": preset.sfm_flavor,
            "est_minutes": preset.sfm_est_minutes,
            "timeout_seconds": (
                preset.sfm_timeout_seconds
                if preset.sfm_timeout_seconds > 0
                else int(max(1800, preset.sfm_est_minutes * 60 * 4))
            ),
            "flavor_scheduling_timeout_seconds": _scheduling_timeout(ctx),
            "command": gs_command,
            "extra_repo_inputs": [],
        },
        inputs=[prep.frames_zip],
        expected_outputs=[prep.splat_path, prep.summary_path],
        log=ctx.log,
        cancel=ctx.cancel_token,
        skip_inputs_upload=bool(resume_job_id),
    )
    ctx.log(
        f"[da3] Direct 3DGS job ({_flavor_label(preset.sfm_flavor)}, model={preset.da3_model}) — "
        f"est {preset.sfm_est_minutes:.0f} min ~${preset.sfm_cost().est_usd:.2f}"
    )
    result = ctx.remote_runner.run(gs_ctx)
    stage.runner = ctx.remote_runner.name
    stage.params = {"preset": preset.key, "sfm_method": "da3", "direct_gs": True}
    if result.metadata:
        record_spend(ctx.manifest, "reconstruction", result.metadata)
    return prep.splat_path if prep.splat_path.exists() else None


# ---------------------------------------------------------------------------
# DA3 incremental merge (add new footage to existing splat)
# ---------------------------------------------------------------------------

def _run_da3_incremental(
    ctx: "DigitalTwinStudioRunner",
    stage: "StageRecord",
    preset,
    resume_job_id: str | None = None,
) -> Path | None:
    """Run DA3 direct 3DGS on new frames, then merge with base splat.ply.

    Pulls the base job's splat.ply from the HF artifact dataset via
    extra_repo_inputs. The DA3 entrypoint runs --gs-only --merge-splat:
    DA3-GIANT predicts 3D Gaussians for the new frames, then the merge
    module aligns (ICP), culls stale base gaussians near new ones, and
    voxel-deduplicates the combined set.

    No splatfacto training, no checkpoint needed. ~5 min on L4.
    """
    from ..pipeline import record_spend
    from ..runners import StageContext

    prep = _prepare_remote(ctx, preset, resume_job_id=resume_job_id)

    merge_from = ctx.manifest.metadata.get("merge_from_job_id")
    if not merge_from:
        raise RuntimeError(
            "da3-incremental requires 'merge_from_job_id' in manifest metadata. "
            "Set it to the base job whose splat.ply should be merged with new footage."
        )

    gs_command = [
        "python", "/opt/vw/da3_entrypoint.py",
        "--gs-only",
        "--merge-splat",
        "--da3-model", preset.da3_model,
    ]

    # Pull the base splat.ply from the prior job's reconstruction output
    base_prefix = f"jobs/{merge_from}/reconstruction/out"
    extra_repo_inputs = [f"{base_prefix}/splat.ply"]

    gs_ctx = StageContext(
        job_dir=ctx.root,
        job_id=prep.hf_job_id,
        stage_key="reconstruction",
        params={
            "image": _resolve_da3_image(ctx, preset, prep),
            "image_has_hub": True,
            "flavor": preset.sfm_flavor,
            "est_minutes": preset.sfm_est_minutes,
            "timeout_seconds": (
                preset.sfm_timeout_seconds
                if preset.sfm_timeout_seconds > 0
                else int(max(1800, preset.sfm_est_minutes * 60 * 4))
            ),
            "flavor_scheduling_timeout_seconds": _scheduling_timeout(ctx),
            "command": gs_command,
            "extra_repo_inputs": extra_repo_inputs,
        },
        inputs=[prep.frames_zip],
        expected_outputs=[prep.splat_path, prep.summary_path],
        log=ctx.log,
        cancel=ctx.cancel_token,
        skip_inputs_upload=bool(resume_job_id),
    )
    ctx.log(
        f"[da3-incremental] Merging new frames with base splat from job {merge_from} "
        f"({_flavor_label(preset.sfm_flavor)}, model={preset.da3_model}) — "
        f"est {preset.sfm_est_minutes:.0f} min ~${preset.sfm_cost().est_usd:.2f}"
    )
    result = ctx.remote_runner.run(gs_ctx)
    stage.runner = ctx.remote_runner.name
    stage.params = {
        "preset": preset.key, "sfm_method": "da3",
        "direct_gs": True, "incremental_merge": True,
        "merge_from_job_id": merge_from,
    }
    if result.metadata:
        record_spend(ctx.manifest, "reconstruction", result.metadata)
    return prep.splat_path if prep.splat_path.exists() else None


# ---------------------------------------------------------------------------
# Remote reconstruction (shared setup + split / non-split variants)
# ---------------------------------------------------------------------------

def _prepare_remote(
    ctx: "DigitalTwinStudioRunner",
    preset,
    resume_job_id: str | None = None,
) -> _RemotePrep:
    """Shared setup for both split and non-split remote reconstruction.

    Validates the worker image, packs frames.zip (or skips on resume),
    and creates the export / output directory layout.
    """
    runner_config = getattr(ctx.remote_runner, "config", None)
    image_name = getattr(runner_config, "worker_image", "") if runner_config else ""
    # When the preset has an image override (e.g. DA3 presets point to the
    # vw-studio-da3 Space), accept a default worker_image of python:3.12 —
    # the override will be used for the actual job, not this placeholder.
    has_override = bool(preset.sfm_image_override)
    if not image_name or (image_name.startswith("python:") and not has_override):
        raise RuntimeError(
            "Remote worker image not configured. Build it with "
            "tools/build_worker_image.ps1 and set worker_image in Settings."
        )

    from ..pipeline import list_frames

    frames_zip = ctx.recon_dir / "frames.zip"
    frame_paths = list_frames(ctx.frames_dir)
    hf_job_id = resume_job_id or ctx.manifest.job_id
    if resume_job_id:
        ctx.log(
            f"[resume] Reusing frames from prior job {resume_job_id} — "
            f"skipping frames.zip upload ({len(frame_paths)} local frames available for reference)"
        )
    else:
        with zipfile.ZipFile(frames_zip, "w", zipfile.ZIP_STORED) as archive:
            for frame in frame_paths:
                archive.write(frame, frame.name)
        ctx.log(
            f"Packed {len(frame_paths)} frames for remote reconstruction "
            f"({frames_zip.stat().st_size // 1_000_000} MB)"
        )

    export_dir = ctx.recon_dir / "gsplat_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    splat_path = export_dir / "splat.ply"
    summary_path = ctx.recon_dir / "summary.json"
    remote_out_dir = ctx.recon_dir / "remote_out"
    bundle_model = remote_out_dir / "model.zip"
    bundle_processed = remote_out_dir / "processed_min.zip"

    return _RemotePrep(
        image_name=image_name,
        frames_zip=frames_zip,
        frame_paths=frame_paths,
        export_dir=export_dir,
        splat_path=splat_path,
        summary_path=summary_path,
        remote_out_dir=remote_out_dir,
        bundle_model=bundle_model,
        bundle_processed=bundle_processed,
        hf_job_id=hf_job_id,
    )


def _run_split_remote(
    ctx: "DigitalTwinStudioRunner",
    stage: "StageRecord",
    preset,
    resume_job_id: str | None = None,
) -> Path | None:
    """Two-job split: SfM on cpu-upgrade, training on GPU.

    Job A (cpu-upgrade) runs COLMAP (--sfm-only) and uploads
    processed_min.zip to the HF artifact dataset under
    jobs/<job_id>/reconstruction_sfm/out/processed_min.zip.

    Job B (GPU, preset.flavor) pulls that artifact via extra_repo_inputs
    and runs --train-only (skipping ns-process-data/COLMAP entirely).
    Both are sequential — Job B blocks on Job A.
    """
    from ..pipeline import record_spend
    from ..runners import StageContext

    prep = _prepare_remote(ctx, preset, resume_job_id=resume_job_id)

    # ---- Job A: SfM only ----
    refine_from = ctx.manifest.metadata.get("refine_from_job_id")
    # Choose entrypoint based on SfM method
    from ..presets import SfmMethod
    sfm_entrypoint = "/opt/vw/da3_entrypoint.py" if preset.sfm_method == SfmMethod.DA3 else "/opt/vw/recon_entrypoint.py"
    sfm_command = [
        "python", sfm_entrypoint,
        "--sfm-only",
        "--downscale", str(preset.downscale_factor),
        "--keep-checkpoint",
    ]
    if preset.sfm_method == SfmMethod.DA3:
        sfm_command.extend(["--da3-model", preset.da3_model])
        if preset.da3_streaming:
            # Streaming poses every frame via overlapping chunks instead of
            # subsampling to --max-sfm-frames. Same processed_min.zip contract,
            # so the training leg below is untouched.
            sfm_command.append("--stream-sfm")
            sfm_command.extend([
                "--stream-resolution", preset.da3_stream_resolution,
                "--stream-chunk-size", str(preset.da3_stream_chunk_size),
                "--stream-overlap", str(preset.da3_stream_overlap),
            ])
            if preset.da3_stream_loop_closure:
                sfm_command.append("--stream-loop-closure")
        else:
            sfm_command.extend(["--max-sfm-frames", str(preset.da3_max_sfm_frames)])
    sfm_extra_inputs: list[str] = []
    if refine_from:
        base_prefix = f"jobs/{refine_from}/reconstruction/out"
        sfm_extra_inputs = [
            f"{base_prefix}/model.zip",
            f"{base_prefix}/processed_min.zip",
        ]
        sfm_command.append("--refine-mode")

    sfm_processed_out = ctx.recon_dir.parent / "reconstruction_sfm" / "remote_out" / "processed_min.zip"
    sfm_image = _resolve_da3_image(ctx, preset, prep)
    sfm_timeout = (
        preset.sfm_timeout_seconds
        if preset.sfm_timeout_seconds > 0
        else int(max(3600, preset.sfm_est_minutes * 60 * 4))
    )
    sfm_ctx = StageContext(
        job_dir=ctx.root,
        job_id=prep.hf_job_id,
        stage_key="reconstruction_sfm",
        params={
            "image": sfm_image,
            "image_has_hub": True,
            "flavor": preset.sfm_flavor,
            "est_minutes": preset.sfm_est_minutes,
            "timeout_seconds": sfm_timeout,
            "flavor_scheduling_timeout_seconds": _scheduling_timeout(ctx),
            "command": sfm_command,
            "extra_repo_inputs": sfm_extra_inputs,
        },
        inputs=[prep.frames_zip],
        expected_outputs=[sfm_processed_out],
        log=ctx.log,
        cancel=ctx.cancel_token,
        skip_inputs_upload=bool(resume_job_id),
    )
    ctx.log(
        f"[split] Job A (SfM, {_flavor_label(preset.sfm_flavor)}) — "
        f"est {preset.sfm_est_minutes:.0f} min ~${preset.sfm_cost().est_usd:.2f}"
    )
    sfm_result = ctx.remote_runner.run(sfm_ctx)
    stage.runner = ctx.remote_runner.name
    if sfm_result.metadata:
        record_spend(ctx.manifest, "reconstruction_sfm", sfm_result.metadata)
    if not sfm_processed_out.exists():
        raise RuntimeError(
            "SfM job (Job A) did not produce processed_min.zip — "
            "check the HF job logs for COLMAP errors."
        )

    # ---- Job B: training only on GPU ----
    sfm_out_prefix = f"jobs/{prep.hf_job_id}/reconstruction_sfm/out"
    # DA3 SfM produces processed_min.zip in the same format as COLMAP,
    # so the training leg uses the same entrypoint with --train-only.
    from ..presets import SfmMethod
    train_entrypoint = "/opt/vw/da3_entrypoint.py" if preset.sfm_method == SfmMethod.DA3 else "/opt/vw/recon_entrypoint.py"
    train_command = [
        "python", train_entrypoint,
        "--train-only",
        "--downscale", str(preset.downscale_factor),
        "--train-args", json.dumps(preset.train_args()),
        "--keep-checkpoint",
    ]
    if preset.sfm_method == SfmMethod.DA3:
        train_command.extend(["--da3-model", preset.da3_model])
    train_extra_inputs = [
        f"{sfm_out_prefix}/processed_min.zip",
        f"jobs/{prep.hf_job_id}/reconstruction_sfm/in/frames.zip",
    ]
    if refine_from:
        base_prefix = f"jobs/{refine_from}/reconstruction/out"
        train_extra_inputs.append(f"{base_prefix}/model.zip")
        train_command.append("--refine-mode")

    train_ctx = StageContext(
        job_dir=ctx.root,
        job_id=prep.hf_job_id,
        stage_key="reconstruction",
        params={
            "image": sfm_image if preset.sfm_method == SfmMethod.DA3 else prep.image_name,
            "image_has_hub": True,
            "flavor": preset.flavor,
            "est_minutes": preset.est_minutes,
            "timeout_seconds": int(max(1800, preset.est_minutes * 60 * 4)),
            "flavor_scheduling_timeout_seconds": _scheduling_timeout(ctx),
            "command": train_command,
            "extra_repo_inputs": train_extra_inputs,
        },
        inputs=[],
        expected_outputs=[prep.splat_path, prep.summary_path, prep.bundle_model, prep.bundle_processed],
        log=ctx.log,
        cancel=ctx.cancel_token,
        skip_inputs_upload=True,
    )
    ctx.log(
        f"[split] Job B (training, {preset.flavor}) — "
        f"est {preset.est_minutes:.0f} min ~${preset.train_cost().est_usd:.2f}"
    )
    train_result = ctx.remote_runner.run(train_ctx)
    stage.params = {"preset": preset.key, "flavor": preset.flavor, "split_jobs": True}
    if train_result.metadata:
        record_spend(ctx.manifest, "reconstruction", train_result.metadata)
    return prep.splat_path if prep.splat_path.exists() else None


def _run_remote(
    ctx: "DigitalTwinStudioRunner",
    stage: "StageRecord",
    preset,
    resume_job_id: str | None = None,
) -> Path | None:
    """Train the splat on rented GPU compute (HF Jobs). Returns the splat PLY.

    When ``resume_job_id`` is provided the frames.zip upload is skipped and
    the HF artifact prefix is set to the original job so the worker finds
    the already-uploaded input on the dataset.
    """
    from ..pipeline import record_spend
    from ..runners import StageContext

    prep = _prepare_remote(ctx, preset, resume_job_id=resume_job_id)

    refine_from = ctx.manifest.metadata.get("refine_from_job_id")
    from ..presets import SfmMethod
    entrypoint = "/opt/vw/da3_entrypoint.py" if preset.sfm_method == SfmMethod.DA3 else "/opt/vw/recon_entrypoint.py"
    worker_command = [
        "python", entrypoint,
        "--downscale", str(preset.downscale_factor),
        "--train-args", json.dumps(preset.train_args()),
        "--keep-checkpoint",
    ]
    if preset.sfm_method == SfmMethod.DA3:
        worker_command.extend(["--da3-model", preset.da3_model])
    extra_repo_inputs: list[str] = []
    if refine_from:
        base_prefix = f"jobs/{refine_from}/reconstruction/out"
        extra_repo_inputs = [
            f"{base_prefix}/model.zip",
            f"{base_prefix}/processed_min.zip",
        ]
        worker_command.append("--refine-mode")

    ctx_obj = StageContext(
        job_dir=ctx.root,
        job_id=prep.hf_job_id,
        stage_key="reconstruction",
        params={
            "image": prep.image_name,
            "image_has_hub": True,
            "flavor": preset.flavor,
            "est_minutes": preset.est_minutes,
            "timeout_seconds": int(max(3600, preset.est_minutes * 60 * 6)),
            "command": worker_command,
            "extra_repo_inputs": extra_repo_inputs,
        },
        inputs=[prep.frames_zip],
        expected_outputs=[prep.splat_path, prep.summary_path, prep.bundle_model, prep.bundle_processed],
        log=ctx.log,
        cancel=ctx.cancel_token,
        skip_inputs_upload=bool(resume_job_id),
    )
    result = ctx.remote_runner.run(ctx_obj)
    stage.runner = ctx.remote_runner.name
    stage.params = {"preset": preset.key, "flavor": preset.flavor}
    if result.metadata:
        record_spend(ctx.manifest, "reconstruction", result.metadata)
    return prep.splat_path if prep.splat_path.exists() else None


# ---------------------------------------------------------------------------
# Local DA3 draft (no Docker, runs on local GPU/CPU)
# ---------------------------------------------------------------------------

def _run_local_da3(
    ctx: "DigitalTwinStudioRunner",
    stage: "StageRecord",
    preset,
) -> Path | None:
    """Run DA3 direct 3DGS locally using the da3_entrypoint.py script.

    This bypasses Docker and runs the entrypoint directly with the local
    Python environment. Requires depth_anything_3 + torch installed.
    Falls back to _run_local (COLMAP) if DA3 deps are missing.
    """
    import importlib.util

    da3_entrypoint = (
        Path(__file__).resolve().parent.parent.parent
        / "docker" / "worker" / "da3_entrypoint.py"
    )
    if not da3_entrypoint.exists():
        ctx.log("[da3-local] da3_entrypoint.py not found, falling back to COLMAP path.")
        return None

    # Check if depth_anything_3 is installed
    if importlib.util.find_spec("depth_anything_3") is None:
        ctx.log(
            "[da3-local] depth_anything_3 is not installed. "
            "Install with: pip install depth-anything-3  "
            "Falling back to COLMAP path."
        )
        return None

    # Check if torch is available
    if importlib.util.find_spec("torch") is None:
        ctx.log("[da3-local] torch is not installed. Falling back to COLMAP path.")
        return None

    frames = list(ctx.frames_dir.glob("*.jpg")) + list(ctx.frames_dir.glob("*.png"))
    if not frames:
        ctx.log("[da3-local] No frames found.")
        return None

    ctx.log(
        f"[da3-local] Running DA3 direct 3DGS on {len(frames)} frames "
        f"(model: {preset.da3_model})"
    )

    out_dir = ctx.recon_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a temp input dir with frames (the entrypoint expects VW_IN)
    in_dir = ctx.recon_dir / "_da3_input"
    in_dir.mkdir(parents=True, exist_ok=True)

    # Copy/symlink frames into input dir
    for frame in frames:
        target = in_dir / frame.name
        if not target.exists():
            try:
                import os as _os
                if _os.name == "nt":
                    shutil.copy2(frame, target)
                else:
                    target.symlink_to(frame)
            except Exception:
                shutil.copy2(frame, target)

    run_env = os.environ.copy()
    run_env["VW_IN"] = str(in_dir)
    run_env["VW_OUT"] = str(out_dir)
    run_env["PYTHONUTF8"] = "1"
    run_env.setdefault("TORCHDYNAMO_DISABLE", "1")

    cmd = [
        sys.executable, str(da3_entrypoint),
        "--gs-only",
        "--da3-model", preset.da3_model,
    ]

    try:
        ctx._run_command(
            cmd,
            "DA3 local reconstruction failed.",
            timeout_seconds=3600,
            env=run_env,
        )
    except RuntimeError as exc:
        ctx.log(f"[da3-local] Failed: {exc}")
        if ctx.strict_mode:
            raise
        return None
    finally:
        # Clean up temp input dir
        try:
            shutil.rmtree(in_dir, ignore_errors=True)
        except Exception:
            pass

    splat_ply = out_dir / "splat.ply"
    if splat_ply.exists():
        stage.runner = "local-da3"
        stage.params = {"preset": preset.key, "sfm_method": "da3", "direct_gs": True}
        ctx.log(f"[da3-local] Success — splat.ply at {splat_ply}")
        return splat_ply

    ctx.log("[da3-local] splat.ply not found after entrypoint completed.")
    return None


# ---------------------------------------------------------------------------
# Local reconstruction
# ---------------------------------------------------------------------------

def _run_local(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> Path | None:
    """Local quick path (250-iteration smoke training). Heavy local runs stay opt-in."""
    from ..pipeline import resolve_binary
    from ..presets import get_preset

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
        str(ctx.frames_dir),
        "--output-dir",
        str(ctx.recon_dir),
    ]
    if str(colmap_bin).upper().endswith(".BAT"):
        cmd.extend(["--colmap-cmd", "COLMAP.bat"])
    try:
        ctx._run_command(cmd, "Reconstruction failed.", timeout_seconds=3600, env=run_env)
    except RuntimeError:
        if ctx.strict_mode:
            raise
        return None

    transforms_path = ctx.recon_dir / "transforms.json"
    if not (ns_train and transforms_path.exists()):
        return None
    local_preset = get_preset("local-debug")
    train_cmd = [
        ns_train,
        "splatfacto",
        "--data",
        str(ctx.recon_dir),
        "--output-dir",
        str(ctx.recon_dir / "gsplat_outputs"),
        *local_preset.train_args(),
    ]
    try:
        ctx._run_command(train_cmd, "gsplat training failed.", timeout_seconds=3600, env=run_env)
        return _export_gsplat_ply(ctx)
    except RuntimeError:
        if ctx.strict_mode:
            raise
        return None


# ---------------------------------------------------------------------------
# Post-reconstruction helpers
# ---------------------------------------------------------------------------

def _gravity_align(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    """Rotate cloud.ply + cloud_preview.ply so world +Y is up.

    Skips silently if the strict-mode build flag is off and the rotation
    fails (alignment is a polish step, not a correctness gate).
    """
    from ..gravity_align import align_cloud

    summary_path = ctx.recon_dir / "summary.json"
    try:
        result = align_cloud(
            ctx.recon_ply_path,
            ctx.recon_preview_ply_path,
            summary_path=summary_path,
            captured_cameras_path=ctx.usd_dir / "captured_cameras.json",
        )
    except Exception as exc:  # noqa: BLE001 - alignment is best-effort
        if ctx.strict_mode:
            raise
        ctx.log(f"Gravity alignment skipped: {exc}")
        return
    if result is None:
        ctx.log("Gravity alignment skipped (cloud already aligned).")
        stage.metadata["gravity_aligned"] = True
        return
    stage.metadata["gravity_aligned"] = True
    stage.metadata["alignment_tilt_degrees"] = round(result.angle_from_y_degrees, 2)
    ctx.log(
        "Gravity-aligned cloud: rotated "
        f"{result.angle_from_y_degrees:.1f}° to bring scene up to +Y "
        f"(skewness {result.skewness:+.2f}, flipped={result.flipped})."
    )


def _write_packed_splat(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    """Encode the splat PLY into the compact .splat format for fast loads.

    ~7x smaller than the equivalent PLY (32 bytes per gaussian vs the full
    ply row), and the viewer skips the PLY header parse entirely. Best
    effort: skips silently outside strict mode if the input isn't a 3DGS
    PLY (placeholder paths) or anything else goes sideways.
    """
    from ..splat_io import is_gaussian_ply
    from ..splat_packed import ply_to_splat

    if not is_gaussian_ply(ctx.recon_ply_path):
        return
    try:
        size = ply_to_splat(ctx.recon_ply_path, ctx.recon_splat_path)
    except Exception as exc:  # noqa: BLE001 - packing is opportunistic
        if ctx.strict_mode:
            raise
        ctx.log(f"Packed .splat skipped: {exc}")
        return
    stage.metadata["packedSplatBytes"] = size
    ctx.log(f"Packed .splat written: {size // 1_000_000} MB.")


def _export_gsplat_ply(ctx: "DigitalTwinStudioRunner") -> Path | None:
    """Run ns-export gaussian-splat on the trained model and return the PLY path, or None."""
    from ..pipeline import resolve_binary

    ns_export = resolve_binary("ns-export")
    if ns_export is None:
        ctx.log("ns-export not found; skipping gsplat PLY export.")
        return None
    config = _find_gsplat_config(ctx)
    if config is None:
        ctx.log("No splatfacto config.yml found in gsplat_outputs; skipping PLY export.")
        return None
    export_dir = ctx.recon_dir / "gsplat_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ns_export,
        "gaussian-splat",
        "--load-config", str(config),
        "--output-dir", str(export_dir),
    ]
    try:
        ctx._run_command(cmd, "ns-export gaussian-splat failed.", timeout_seconds=600)
    except RuntimeError as exc:
        ctx.log(f"gsplat PLY export failed: {exc}")
        return None
    candidates = sorted(export_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_gsplat_config(ctx: "DigitalTwinStudioRunner") -> Path | None:
    """Return the most recently modified config.yml under gsplat_outputs, or None."""
    gsplat_out = ctx.recon_dir / "gsplat_outputs"
    if not gsplat_out.exists():
        return None
    configs = sorted(gsplat_out.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return configs[0] if configs else None


def _convert_ply_to_cloud_files(ctx: "DigitalTwinStudioRunner", ply_source: Path) -> bool:
    """Read a gsplat PLY with open3d and write cloud.ply + cloud.usda. Returns True on success."""
    try:
        import open3d as o3d
    except ImportError:
        ctx.log("open3d not available; cannot convert PLY.")
        return False
    try:
        pcd = o3d.io.read_point_cloud(str(ply_source))
    except Exception as exc:
        ctx.log(f"open3d failed to read {ply_source}: {exc}")
        return False
    if len(pcd.points) == 0:
        ctx.log(f"Loaded PLY has no points: {ply_source}")
        return False
    o3d.io.write_point_cloud(str(ctx.recon_ply_path), pcd)
    ctx.log(f"Wrote {len(pcd.points)} points to {ctx.recon_ply_path}")
    _write_usd_from_point_cloud(ctx, pcd)
    return True


def _write_usd_from_point_cloud(ctx: "DigitalTwinStudioRunner", pcd) -> None:
    """Write cloud.usda populated with real geometry from an open3d PointCloud."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(ctx.recon_stage_path))
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
    ctx.log(f"Wrote USD stage with {len(pts_vec)} real points to {ctx.recon_stage_path}")


def _write_placeholder_reconstruction(ctx: "DigitalTwinStudioRunner") -> None:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(ctx.recon_stage_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    points = UsdGeom.Points.Define(stage, "/World/Reconstruction")
    points.GetPointsAttr().Set(
        [Gf.Vec3f(-0.5, 0.0, 0.0), Gf.Vec3f(0.0, 0.5, 0.0), Gf.Vec3f(0.5, 0.0, 0.0)]
    )
    points.GetWidthsAttr().Set([0.05, 0.05, 0.05])
    stage.GetRootLayer().Save()


def _write_placeholder_ply(ctx: "DigitalTwinStudioRunner") -> None:
    ctx.recon_ply_path.write_text(
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
