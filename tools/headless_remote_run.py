"""Headless end-to-end pipeline run with remote reconstruction.

Runs the full 5-stage job (intake -> frames -> REMOTE reconstruction ->
USD/cameras -> output) without the GUI. Used for M1 live verification and
scriptable runs.

Cost: launches ONE paid HF Job for the reconstruction stage. The estimate is
printed before launch; pass --yes to approve it (without --yes the run stays
local-only).

Usage:
    .venv\\Scripts\\python.exe tools\\headless_remote_run.py --preset draft --yes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.pipeline import (  # noqa: E402
    DEFAULT_SOURCE_VIDEO,
    DigitalTwinStudioRunner,
    StageState,
    create_job_manifest,
    load_job_manifest,
    save_job_manifest,
)
from vaultwares_studio.presets import get_preset  # noqa: E402
from vaultwares_studio.runners import HfJobsConfig, HfJobsStageRunner  # noqa: E402


def _pull_base_frames_zip(base_job_id: str, dest: Path, log) -> Path:
    """Pull the base job's frames.zip from the HF Dataset into ``dest``.

    Mirrors the lookup hf_jobs.py does: ``<user>/vw-studio-artifacts`` repo,
    ``jobs/<base_job_id>/reconstruction/in/frames.zip``. Refine joins old and
    new captures into a single COLMAP run, so we need the original images at
    the same fidelity the base job had.
    """
    from huggingface_hub import HfApi
    from vaultwares_studio.runners.hf_jobs import get_hf_token

    token = get_hf_token()
    if not token:
        raise RuntimeError("HF token not configured (settings keyring).")
    api = HfApi(token=token)
    repo = f"{api.whoami()['name']}/vw-studio-artifacts"
    remote = f"jobs/{base_job_id}/reconstruction/in/frames.zip"
    log(f"[refine] pulling base frames from {repo}:{remote}")
    local = api.hf_hub_download(repo_id=repo, filename=remote, repo_type="dataset")
    dest.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copyfile(local, dest)
    size_mb = dest.stat().st_size // 1_000_000
    log(f"[refine] base frames.zip cached at {dest} ({size_mb} MB)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        action="append",
        default=None,
        help=(
            "Source video. Pass --video multiple times to feed several clips "
            "into the same reconstruction; each video is blur-pruned to ~280 "
            "sharpest frames independently so longer clips don't swamp shorter "
            "ones. First --video is the manifest's primary; additional clips "
            "land alongside with extra<N>_ prefixes."
        ),
    )
    parser.add_argument("--preset", default="draft")
    parser.add_argument("--yes", action="store_true", help="approve the remote job cost")
    parser.add_argument(
        "--refine-from",
        default=None,
        help=(
            "Base job id (e.g. local-run-20260613-211202). When set, the job's "
            "frames.zip is pulled from the HF Dataset and combined with frames "
            "extracted from --video so COLMAP gets a single joint reconstruction. "
            "Lighting drift between captures will cause splat ghosting where the "
            "supervising views disagree — match conditions when you can."
        ),
    )
    parser.add_argument(
        "--resume-job",
        default=None,
        metavar="JOB_ID",
        help=(
            "Resume a failed job by reusing its already-uploaded HF frames.zip. "
            "Pass the local job id (e.g. local-run-20260615-065732). The prior "
            "job's manifest is loaded, frame extraction is skipped (frames already "
            "on HF), and reconstruction is re-queued against the same artifact path. "
            "--video, --preset, and --refine-from are inferred from the prior manifest "
            "unless explicitly overridden on the command line."
        ),
    )
    args = parser.parse_args()

    preset = get_preset(args.preset)

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    remote_runner = None
    if args.yes:
        config = HfJobsConfig.load()
        remote_runner = HfJobsStageRunner(
            config=config,
            confirm_cost=lambda est: print(f"[run] cost pre-approved via --yes: {est.summary()}") or True,
        )
        print(f"[run] remote runner: image={config.worker_image}")
    else:
        print("[run] --yes not given: remote stage will fall back to the local quick path")

    if args.resume_job:
        resume_job_dir = ROOT / "data" / "jobs" / args.resume_job
        resume_manifest_path = resume_job_dir / "manifest.json"
        if not resume_manifest_path.exists():
            print(f"[resume] manifest not found: {resume_manifest_path}", file=sys.stderr)
            return 1
        prior = load_job_manifest(resume_manifest_path)
        # Inherit videos and preset from prior manifest unless explicitly overridden.
        if args.video is None:
            primary_video = prior.source_video
            extra_videos_from_prior = prior.metadata.get("extra_videos", [])
            extra_videos = [str(ROOT / v) if not Path(v).is_absolute() else v
                            for v in extra_videos_from_prior]
        else:
            primary_video, *extra_videos = args.video
        if prior.metadata.get("preset") and args.preset == "draft":
            preset = get_preset(prior.metadata["preset"])
        print(f"[resume] resuming {args.resume_job} | preset={preset.key} | reusing HF frames")
        print(f"[resume] remote cost estimate: {preset.cost().summary()}")
        print(f"[resume] refine_from: {prior.metadata.get('refine_from_job_id', '-')}")

        # Create a fresh local job that clones the prior manifest's intent.
        manifest = create_job_manifest(source_video=primary_video)
        manifest.metadata["preset"] = preset.key
        if extra_videos:
            manifest.metadata["extra_videos"] = extra_videos
        if prior.metadata.get("refine_from_job_id"):
            manifest.metadata["refine_from_job_id"] = prior.metadata["refine_from_job_id"]
            manifest.metadata["refine_base_frames_zip"] = prior.metadata.get(
                "refine_base_frames_zip", ""
            )
        # Key field: tells _run_remote_reconstruction which job's HF prefix to use.
        manifest.metadata["resume_job_id"] = args.resume_job

        # Mark video_intake and frame_extraction as complete so run_remaining
        # skips them — the frames already exist on HF from the prior run.
        for stage in manifest.stages:
            if stage.key in ("video_intake", "frame_extraction"):
                stage.state = StageState.COMPLETE.value
                stage.message = f"Skipped (resuming from {args.resume_job})"
        manifest.current_stage_key = "reconstruction"
        save_job_manifest(manifest)
        print(f"[resume] new job: {manifest.job_id} -> {manifest.output_dir}")

    else:
        videos = args.video or [str(DEFAULT_SOURCE_VIDEO)]
        primary_video, *extra_videos = videos
        estimate = preset.cost()
        print(f"[run] videos={videos} preset={preset.key} | remote cost estimate: {estimate.summary()}")
        if args.refine_from:
            print(f"[run] refine mode: combining frames with base job {args.refine_from}")
        if extra_videos:
            print(f"[run] additional videos ({len(extra_videos)}) will be extracted alongside the primary")

        manifest = create_job_manifest(source_video=primary_video)
        manifest.metadata["preset"] = preset.key
        if extra_videos:
            manifest.metadata["extra_videos"] = extra_videos
        print(f"[run] job: {manifest.job_id} -> {manifest.output_dir}")

        if args.refine_from:
            base_frames_path = Path(manifest.output_dir) / "frames" / "_base_frames.zip"
            try:
                _pull_base_frames_zip(args.refine_from, base_frames_path, log)
            except Exception as exc:  # noqa: BLE001
                print(f"[refine] base frames pull failed: {exc}")
                return 1
            manifest.metadata["refine_from_job_id"] = args.refine_from
            manifest.metadata["refine_base_frames_zip"] = str(base_frames_path)

    runner = DigitalTwinStudioRunner(manifest, log, remote_runner=remote_runner)
    started = time.monotonic()
    result = runner.run_remaining()
    elapsed = time.monotonic() - started

    print(f"\n[run] finished in {elapsed/60:.1f} min — state: {result.state}")
    for stage in result.stages:
        print(f"  {stage.key:18} {stage.state:10} {stage.message}")
    recon = next(stage for stage in result.stages if stage.key == "reconstruction")
    print(f"[run] reconstruction degraded: {recon.metadata.get('degraded')}, "
          f"gaussians: {recon.metadata.get('gaussians', 'n/a')}, runner: {recon.runner}")
    if result.spend_ledger:
        print(f"[run] spend ledger: {json.dumps(result.spend_ledger, indent=2)}")
    return 0 if result.state == "complete" and not recon.metadata.get("degraded") else 1


if __name__ == "__main__":
    sys.exit(main())
