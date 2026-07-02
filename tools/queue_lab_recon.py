"""Fire a lab-cpu-3000 reconstruction against the vw-studio-recon-lab Space.

Self-contained launcher for the dense-frame curiosity test: take a source
video, extract every Nth frame (no sharpness prune, hard-cap at preset's
frame_cap), zip, and submit a Job A (--sfm-only) HF job pointed at the lab
image. Job B (GPU training) is intentionally NOT fired -- once the SfM result
lands the user iterates on 3DGRUT/3DGUT in the HF console before we bake a
GPU-side image.

Usage:
    .venv\\Scripts\\python.exe tools\\queue_lab_recon.py \\
        --source-video "inputs/cloudyday2_june14_348sec.MOV" \\
        [--preset lab-cpu-3000] [--space vw-studio-recon-lab]

Prereqs:
    - tools/push_lab_space.py has been run at least once and HF finished
      building the lab image.
    - HF token configured (Settings > Remote Compute, or HF_TOKEN env).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# Windows consoles default to cp1252 which chokes on unicode (arrows, em-dash,
# emoji) the moment any log line contains them. Reconfigure once at import so
# every downstream print() / huggingface_hub log survives.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - older python or already-replaced streams
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi  # noqa: E402

from vaultwares_studio.pipeline import (  # noqa: E402
    DigitalTwinStudioRunner,
    StageState,
    create_job_manifest,
    list_frames,
    load_job_manifest,
    save_job_manifest,
)
from vaultwares_studio.presets import get_preset  # noqa: E402
from vaultwares_studio.runners import (  # noqa: E402
    CancelToken,
    HfJobsConfig,
    HfJobsStageRunner,
    StageContext,
    get_hf_token,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-video",
        default="inputs/cloudyday2_june14_348sec.MOV",
        help="Path (absolute or relative to repo root) to the input video.",
    )
    parser.add_argument("--preset", default="lab-cpu-3000")
    parser.add_argument("--space", default="vw-studio-recon-lab")
    parser.add_argument(
        "--reuse-job",
        default=None,
        help=(
            "Local job id to reuse (skips video_intake + frame_extraction + zip). "
            "Use after a crash that already produced frames + frames.zip."
        ),
    )
    args = parser.parse_args()

    preset = get_preset(args.preset)
    if not preset.split_jobs or not preset.unrestricted_frames:
        print(
            f"Preset {args.preset!r} is not a lab preset "
            "(needs split_jobs=True + unrestricted_frames=True).",
            file=sys.stderr,
        )
        return 1

    token = get_hf_token()
    if not token:
        print("No HF token configured (Settings > Remote Compute).", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    space_image = f"hf.co/spaces/{owner}/{args.space}"
    print(f"[lab-queue] preset={preset.key} flavor={preset.sfm_flavor} image={space_image}")
    print(
        f"[lab-queue] frame plan: unrestricted_frames=True cap={preset.frame_cap} "
        f"timeout={preset.sfm_timeout_seconds}s"
    )

    if args.reuse_job:
        # Reuse path: prior crash already produced frames + frames.zip. Load the
        # existing manifest, skip extraction, jump straight to Job A submission.
        manifest_candidates = [
            ROOT / "data" / "jobs" / args.reuse_job / "manifest.json",
            Path("D:/vaultwares-studio-jobs/data/jobs") / args.reuse_job / "manifest.json",
        ]
        manifest_path = next((p for p in manifest_candidates if p.exists()), None)
        if manifest_path is None:
            print(f"[lab-queue] No manifest found for --reuse-job {args.reuse_job}", file=sys.stderr)
            return 1
        manifest = load_job_manifest(manifest_path)
        job_dir = Path(manifest.output_dir)
        frames_zip = job_dir / "reconstruction" / "frames.zip"
        if not frames_zip.exists():
            print(f"[lab-queue] {frames_zip} missing -- cannot reuse this job", file=sys.stderr)
            return 1
        frame_paths = list_frames(job_dir / "frames")
        print(
            f"[lab-queue] reusing job_id={manifest.job_id} "
            f"({len(frame_paths)} frames, frames.zip={frames_zip.stat().st_size // 1_000_000} MB)"
        )
    else:
        source = Path(args.source_video)
        if not source.is_absolute():
            source = (ROOT / source).resolve()
        if not source.exists():
            print(f"Source video not found: {source}", file=sys.stderr)
            return 1
        # Create a fresh local job manifest and run video_intake + frame_extraction
        # locally. The preset's lab flags (unrestricted_frames + frame_cap) are
        # honoured by _run_frame_extraction, so this stage produces our dense set.
        manifest = create_job_manifest(source_video=source)
        manifest.metadata["preset"] = preset.key
        manifest.metadata["lab_space"] = args.space
        save_job_manifest(manifest)
        job_dir = Path(manifest.output_dir)
        print(f"[lab-queue] job_id={manifest.job_id} dir={job_dir}")

        runner = DigitalTwinStudioRunner(manifest, log=lambda msg: print(f"[lab-queue:{msg}]"))
        runner.run_stage("video_intake")
        runner.run_stage("frame_extraction")

        frames_dir = job_dir / "frames"
        frame_paths = list_frames(frames_dir)
        if not frame_paths:
            print("[lab-queue] frame extraction produced zero frames.", file=sys.stderr)
            return 1
        print(f"[lab-queue] {len(frame_paths)} frames ready in {frames_dir.name}")

        # Pack frames.zip into the reconstruction dir (matches the pipeline path so
        # downstream tooling can find it).
        recon_dir = job_dir / "reconstruction"
        recon_dir.mkdir(parents=True, exist_ok=True)
        frames_zip = recon_dir / "frames.zip"
        with zipfile.ZipFile(frames_zip, "w", zipfile.ZIP_STORED) as archive:
            for frame in frame_paths:
                archive.write(frame, frame.name)
        print(
            f"[lab-queue] packed {len(frame_paths)} frames "
            f"({frames_zip.stat().st_size // 1_000_000} MB) -> {frames_zip.name}"
        )

    # Fire Job A (SfM-only). expected_outputs holds the SfM-side processed_min
    # so the runner waits for it before declaring success.
    remote_out = job_dir / "reconstruction" / "remote_out"
    remote_out.mkdir(parents=True, exist_ok=True)
    sfm_processed_out = remote_out / "sfm_processed_min.zip"

    sfm_command = [
        "python", "/opt/vw/recon_entrypoint.py",
        "--sfm-only",
        "--downscale", str(preset.downscale_factor),
        "--keep-checkpoint",
        # Lab matcher/extractor tunings: 5x fewer vocab candidates per query
        # and a ~2x smaller SIFT pyramid. Cuts the dominant matcher cost on a
        # 3000-frame set; tradeoff is fewer loop closures + more brittle SfM
        # on low-texture scenes. queue_lab_recon.py is the only caller setting
        # these, so prod recon_entrypoint runs keep the historical defaults.
        "--vocab-tree-num-images", "10",
        "--sift-max-image-size", "910",
    ]

    config = HfJobsConfig.load()
    hf_runner = HfJobsStageRunner(
        config=config,
        confirm_cost=lambda est: print(f"[lab-queue] cost pre-approved: {est.summary()}") or True,
    )
    ctx = StageContext(
        job_dir=job_dir,
        job_id=manifest.job_id,
        stage_key="reconstruction_sfm",
        params={
            "image": space_image,
            "image_has_hub": True,
            "flavor": preset.sfm_flavor,
            "est_minutes": preset.sfm_est_minutes,
            "timeout_seconds": preset.sfm_timeout_seconds or int(max(3600, preset.sfm_est_minutes * 60 * 4)),
            "command": sfm_command,
            "extra_repo_inputs": [],
        },
        inputs=[frames_zip],
        expected_outputs=[sfm_processed_out],
        log=lambda msg: print(f"[lab-queue:hf] {msg}"),
        cancel=CancelToken(),
        skip_inputs_upload=False,
    )
    print(
        f"[lab-queue] firing Job A: est {preset.sfm_est_minutes:.0f} min "
        f"~${preset.sfm_cost().est_usd:.2f}, hard timeout "
        f"{ctx.params['timeout_seconds']}s ({ctx.params['timeout_seconds'] // 3600}h)"
    )
    result = hf_runner.run(ctx)
    print(f"[lab-queue] result: {result.status}")
    print(f"[lab-queue] artifacts: {[str(p) for p in result.artifacts]}")

    # Mark the manifest's reconstruction stage as needs-user-input so the GUI
    # doesn't think Job A's output is the end of the road -- the user still owes
    # the GPU half (interactively or via a future queue_lab_train.py).
    for stage in manifest.stages:
        if stage.key == "reconstruction":
            stage.state = StageState.NEEDS_USER_INPUT.value
            stage.message = (
                f"Lab SfM complete; processed_min.zip in {remote_out.name}. "
                "Run 3DGRUT training manually in the lab Space HF console, "
                "or fire a future queue_lab_train.py once the GPU image is baked."
            )
            break
    save_job_manifest(manifest)
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
