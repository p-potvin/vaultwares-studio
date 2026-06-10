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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(DEFAULT_SOURCE_VIDEO))
    parser.add_argument("--preset", default="draft")
    parser.add_argument("--yes", action="store_true", help="approve the remote job cost")
    args = parser.parse_args()

    preset = get_preset(args.preset)
    estimate = preset.cost()
    print(f"[run] video={args.video} preset={preset.key} | remote cost estimate: {estimate.summary()}")

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

    manifest = create_job_manifest(source_video=args.video)
    manifest.metadata["preset"] = preset.key
    print(f"[run] job: {manifest.job_id} -> {manifest.output_dir}")

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

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
