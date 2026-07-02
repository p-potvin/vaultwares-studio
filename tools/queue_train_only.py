"""Fire Job B (train-only) against an already-completed Job A (--sfm-only).

Use when:
- A split-flavor recon's Job A succeeded but the launcher died before Job B
  could fire (the common "launcher never pings" case).
- You want to re-train a splat with different ns-train args on an existing
  SfM output without redoing the ~30-90 min COLMAP work.

Usage:
    .venv/Scripts/python.exe tools/queue_train_only.py --job <local-job-id>
                                                       [--preset standard|refine]
                                                       [--refine-from <base-job-id>]

The script pulls nothing locally — it just constructs the same StageContext
the pipeline would have used for Job B and hands it to HfJobsStageRunner.
Frames + processed_min.zip flow into the worker via extra_repo_inputs from
the HF artifact dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.pipeline import load_job_manifest  # noqa: E402
from vaultwares_studio.presets import get_preset  # noqa: E402
from vaultwares_studio.runners import (  # noqa: E402
    CancelToken,
    HfJobsConfig,
    HfJobsStageRunner,
    StageContext,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Local job id, e.g. local-run-20260629-144053")
    parser.add_argument("--preset", default="standard")
    parser.add_argument(
        "--refine-from",
        default=None,
        help="If Job A was --refine-mode, also pull this base job's model.zip for --load-dir.",
    )
    args = parser.parse_args()

    preset = get_preset(args.preset)

    # Find the local manifest (the launcher uses a few different roots).
    manifest_candidates = [
        ROOT / "data" / "jobs" / args.job / "manifest.json",
        Path("D:/vaultwares-studio-jobs/data/jobs") / args.job / "manifest.json",
    ]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_path is None:
        print(f"No manifest.json found for {args.job} under: {[str(c) for c in manifest_candidates]}", file=sys.stderr)
        return 1
    manifest = load_job_manifest(manifest_path)
    job_dir = Path(manifest.output_dir)
    recon_dir = job_dir / "reconstruction"
    remote_out = recon_dir / "remote_out"
    remote_out.mkdir(parents=True, exist_ok=True)
    (recon_dir / "gsplat_export").mkdir(parents=True, exist_ok=True)

    splat_path = recon_dir / "gsplat_export" / "splat.ply"
    summary_path = recon_dir / "summary.json"
    bundle_model = remote_out / "model.zip"
    bundle_processed = remote_out / "processed_min.zip"

    config = HfJobsConfig.load()
    runner = HfJobsStageRunner(
        config=config,
        confirm_cost=lambda est: print(f"[queue-b] cost pre-approved: {est.summary()}") or True,
    )

    sfm_out_prefix = f"jobs/{manifest.job_id}/reconstruction_sfm/out"
    train_command = [
        "python", "/opt/vw/recon_entrypoint.py",
        "--train-only",
        "--downscale", str(preset.downscale_factor),
        "--train-args", json.dumps(preset.train_args()),
        "--keep-checkpoint",
    ]
    train_extra_inputs = [
        f"{sfm_out_prefix}/processed_min.zip",
        f"jobs/{manifest.job_id}/reconstruction_sfm/in/frames.zip",
    ]
    if args.refine_from:
        train_extra_inputs.append(f"jobs/{args.refine_from}/reconstruction/out/model.zip")
        train_command.append("--refine-mode")

    print(f"[queue-b] preset={preset.key} flavor={preset.flavor} est~{preset.est_minutes:.0f}min")
    print(f"[queue-b] extra_inputs: {train_extra_inputs}")
    print(f"[queue-b] expected_outputs: {[str(p.name) for p in [splat_path, summary_path, bundle_model, bundle_processed]]}")

    ctx = StageContext(
        job_dir=job_dir,
        job_id=manifest.job_id,
        stage_key="reconstruction",
        params={
            "image": config.worker_image,
            "image_has_hub": True,
            "flavor": preset.flavor,
            "est_minutes": preset.est_minutes,
            "timeout_seconds": int(max(1800, preset.est_minutes * 60 * 4)),
            "command": train_command,
            "extra_repo_inputs": train_extra_inputs,
        },
        inputs=[],
        expected_outputs=[splat_path, summary_path, bundle_model, bundle_processed],
        log=print,
        cancel=CancelToken(),
        skip_inputs_upload=True,
    )
    result = runner.run(ctx)
    print(f"[queue-b] result: {result.status}")
    print(f"[queue-b] artifacts: {[str(p) for p in result.artifacts]}")
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
