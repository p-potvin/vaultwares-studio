"""Run DA3 for its DEPTH only, over frames someone else has already posed.

The hybrid path (``docs/hybrid-colmap-da3-20260913.md``) takes poses and
intrinsics from COLMAP and depth from DA3. This fires the DA3 half.

Unlike every other DA3 launcher here, this one does **not** extract frames from a
video. It submits an existing ``frames.zip`` unchanged, because the join between
DA3's depth and COLMAP's poses is **by frame stem** — ``build_hybrid_seed`` pairs
``depths/frame_00123.npy`` with the pose of ``images/frame_00123.jpg``. Re-extract
the frames and the stems still look right while referring to different moments in
the video, which produces a plausible cloud built from the wrong
correspondences. So: the same zip COLMAP was given, or nothing.

DA3's poses and intrinsics come back too and are simply ignored. They are the
half we measured as worse.

    python tools/queue_da3_depth.py \\
        --frames "D:/.../local-run-20260614-234541/reconstruction/frames.zip" \\
        --job data/jobs/hybrid-backyard-20260913 --yes

Costs one paid HF job. The estimate prints before launch and ``--yes`` approves
it; without ``--yes`` nothing is submitted.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - older python or already-replaced streams
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.presets import get_preset  # noqa: E402
from vaultwares_studio.runners import (  # noqa: E402
    CancelToken,
    HfJobsConfig,
    HfJobsStageRunner,
    StageContext,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path,
                        help="An existing frames.zip — the SAME one the poses were solved on")
    parser.add_argument("--job", required=True, type=Path,
                        help="Job directory to write remote_out/ into")
    parser.add_argument("--preset", default="da3-standard")
    parser.add_argument("--max-frames", type=int, default=80,
                        help="Frames handed to DA3, subsampled evenly. Bounded by the "
                             "quadratic multi-view attention: 80 is what fits a 22GB L4.")
    parser.add_argument("--da3-model", default=None,
                        help="Overrides the preset's model")
    parser.add_argument("--flavor", action="append")
    parser.add_argument("--yes", action="store_true", help="approve the job cost")
    args = parser.parse_args()

    if not args.frames.is_file():
        print(f"[da3-depth] no frames.zip at {args.frames}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(args.frames) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not names:
        print(f"[da3-depth] {args.frames} contains no images", file=sys.stderr)
        return 1
    print(f"[da3-depth] {len(names)} frames in {args.frames.name} "
          f"({args.frames.stat().st_size / 1e6:.0f} MB); DA3 will use {args.max_frames}")
    print(f"[da3-depth] stems are the join key: {names[0]} .. {names[-1]}")

    preset = get_preset(args.preset)
    config = HfJobsConfig.load()
    owner = config.namespace
    image = (preset.sfm_image_override or "hf.co/spaces/{owner}/vw-studio-da3")
    if "{owner}" in image:
        if not owner:
            print("[da3-depth] no HF owner configured; cannot resolve the worker image",
                  file=sys.stderr)
            return 1
        image = image.format(owner=owner)

    command = [
        "python", "/opt/vw/da3_entrypoint.py",
        "--sfm-only",
        "--downscale", "1",
        "--max-sfm-frames", str(args.max_frames),
        "--da3-model", args.da3_model or preset.da3_model,
    ]
    # No --calibration here: this job is run for its depth maps, and a lens
    # calibration only affects the transforms.json that the hybrid discards.
    # Passing one would mean shipping the file into the container for no gain.
    job_dir = args.job
    (job_dir / "reconstruction" / "remote_out").mkdir(parents=True, exist_ok=True)
    remote_out = job_dir / "reconstruction" / "remote_out"

    estimate = preset.sfm_cost()
    print(f"[da3-depth] image={image} flavor={args.flavor or preset.sfm_flavor}")
    print(f"[da3-depth] est {preset.sfm_est_minutes:.0f} min, ~${estimate.est_usd:.2f}")
    if not args.yes:
        print("[da3-depth] not submitting — pass --yes to approve the cost.")
        return 0

    runner = HfJobsStageRunner(
        config=config,
        confirm_cost=lambda est: print(f"[da3-depth] cost pre-approved: {est.summary()}") or True,
    )
    ctx = StageContext(
        job_dir=job_dir,
        job_id=job_dir.name,
        stage_key="reconstruction_sfm",
        params={
            "image": image,
            "image_has_hub": True,
            "flavor": args.flavor or preset.sfm_flavor,
            "est_minutes": preset.sfm_est_minutes,
            "timeout_seconds": preset.sfm_timeout_seconds or 3600,
            "command": command,
            "extra_repo_inputs": [],
        },
        inputs=[args.frames],
        # depths.zip is the point of this job. processed_min.zip comes along and
        # is discarded by the hybrid — waiting on the depth bundle is what says
        # the run was useful.
        expected_outputs=[remote_out / "depths.zip"],
        log=lambda msg: print(f"[da3-depth:hf] {msg}"),
        cancel=CancelToken(),
        skip_inputs_upload=False,
    )

    result = runner.run(ctx)
    print(f"[da3-depth] result: {result.status}")
    for artifact in result.artifacts:
        print(f"[da3-depth]   {artifact}")
    if result.status == "complete":
        print("\n[da3-depth] next: unpack depths.zip, then\n"
              f"  python tools/build_hybrid_seed.py --colmap <colmap processed_min.zip> "
              f"--depths {remote_out / 'depths'} --out {job_dir / 'hybrid'}\n"
              "  then READ scale_spread before spending anything on training.")
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
