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
    create_job_manifest,
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
    args = parser.parse_args()

    videos = args.video or [str(DEFAULT_SOURCE_VIDEO)]
    primary_video, *extra_videos = videos
    preset = get_preset(args.preset)
    estimate = preset.cost()
    print(f"[run] videos={videos} preset={preset.key} | remote cost estimate: {estimate.summary()}")
    if args.refine_from:
        print(f"[run] refine mode: combining frames with base job {args.refine_from}")
    if extra_videos:
        print(f"[run] additional videos ({len(extra_videos)}) will be extracted alongside the primary")

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

    manifest = create_job_manifest(source_video=primary_video)
    manifest.metadata["preset"] = preset.key
    if extra_videos:
        manifest.metadata["extra_videos"] = extra_videos
    print(f"[run] job: {manifest.job_id} -> {manifest.output_dir}")

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    if args.refine_from:
        # Cache the base frames.zip inside the new job dir; frame_extraction
        # reads the path from manifest.metadata and unpacks it next to the
        # freshly-extracted new frames.
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
