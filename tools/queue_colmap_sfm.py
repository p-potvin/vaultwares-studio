"""Run COLMAP SfM over an existing frames.zip, on a CPU flavor.

The hybrid seed needs ONE self-consistent COLMAP bundle: transforms, points and
tracks from a single reconstruction. COLMAP's coordinate gauge is arbitrary per
run, so visibility from one model says nothing about poses from another — which
is what blocked the first fusion attempt.

Nothing on disk satisfies that today. The 14 June bundle ships a 10-image,
2,325-point fragment as its sparse model against 491 posed frames, because the
entrypoint picked the sparse directory by modification time and COLMAP's mapper
writes several. That is fixed (``_largest_sparse_model``, by registered image
count) and this launcher overlays the corrected entrypoint, so the bundle this
produces carries the model its own transforms came from.

CPU rather than GPU on purpose: COLMAP's cost here is feature matching, which is
CPU-bound, and ``cpu-upgrade`` has no spot-capacity queue to wait through.
Slower in wall clock, cheap, and it does not tie up a local machine.

    python tools/queue_colmap_sfm.py         --frames "D:/.../local-run-20260614-234541/reconstruction/frames.zip"         --job data/jobs/colmap-backyard-20260913 --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
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


def package_worker(staging: Path, calibration: Path | None = None) -> Path:
    """The worker overlay: today's entrypoint, run by an image built in July.

    The deployed vw-studio-da3 image predates the depth bundle entirely — a run
    against it completes, returns processed_min.zip, and writes a summary with
    no `depth_maps` key, because its --sfm-only branch never calls
    make_depth_bundle. That cost a real job to discover. Overlaying the current
    file is the same trick prepare_zerogpu_training.py uses: ~20 KB uploaded,
    no image rebuild.
    """
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / "worker.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as worker:
        worker.write(ROOT / "docker/worker/recon_entrypoint.py", "recon_entrypoint.py")
        worker.write(ROOT / "vaultwares_studio/streaming_convert.py", "streaming_convert.py")
        worker.write(ROOT / "vaultwares_studio/camera_calibration.py", "camera_calibration.py")
        if calibration and Path(calibration).is_file():
            worker.write(calibration, "calibration.json")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path,
                        help="An existing frames.zip — the SAME one the poses were solved on")
    parser.add_argument("--job", required=True, type=Path,
                        help="Job directory to write remote_out/ into")
    parser.add_argument("--preset", default="lab-cpu-3000")
    parser.add_argument("--flavor", action="append")
    parser.add_argument("--scheduling-timeout", type=float, default=900.0,
                        help="Seconds to wait for a flavor to leave SCHEDULING before "
                             "trying the next. The runner's 120s default is optimistic — "
                             "l4x1 has been measured at 20+ min in SCHEDULING, and a "
                             "cancelled job costs nothing, so patience is free.")
    parser.add_argument("--calibration", type=Path,
                        default=ROOT / "config" / "calibrations" / "iphone-1920x1080.json",
                        help="Lens calibration JSON passed to COLMAP as camera_params. "
                             "Pass an empty string to deliberately run uncalibrated.")
    parser.add_argument("--sift-num-threads", type=int, default=-1,
                        help="-1 uses every core (cpu-upgrade has 64).")
    parser.add_argument("--match-num-threads", type=int, default=-1)
    parser.add_argument("--sift-max-num-features", type=int, default=4096)
    parser.add_argument("--ba-global-max-refinements", type=int, default=2,
                        help="COLMAP's 5 re-solves the whole global BA up to five "
                             "times per trigger. The biggest lever on mapper time.")
    parser.add_argument("--ba-global-ratio", type=float, default=1.3,
                        help="Model growth that triggers a global BA (COLMAP: 1.1).")
    parser.add_argument("--ba-global-max-num-iterations", type=int, default=30)
    parser.add_argument("--ba-global-function-tolerance", type=float, default=1e-6,
                        help="COLMAP's 0 disables Ceres' converged-early exit.")
    parser.add_argument("--ba-use-gpu", action="store_true",
                        help="cpu-upgrade has no GPU; only useful on a GPU flavor.")
    parser.add_argument("--no-timeout", action="store_true",
                        help="Submit with no remote time limit at all. COLMAP's cost is "
                             "superlinear in image count and a cap that fires uploads nothing.")
    parser.add_argument("--yes", action="store_true", help="approve the job cost")
    args = parser.parse_args()

    if not args.frames.is_file():
        print(f"[colmap] no frames.zip at {args.frames}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(args.frames) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not names:
        print(f"[colmap] {args.frames} contains no images", file=sys.stderr)
        return 1
    print(f"[colmap] {len(names)} frames in {args.frames.name} "
          f"({args.frames.stat().st_size / 1e6:.0f} MB)")
    print(f"[colmap] stems are the join key: {names[0]} .. {names[-1]}")

    preset = get_preset(args.preset)
    config = HfJobsConfig.load()
    owner = config.namespace
    image = "hf.co/spaces/{owner}/vw-studio-worker"
    if "{owner}" in image:
        if not owner:
            print("[colmap] no HF owner configured; cannot resolve the worker image",
                  file=sys.stderr)
            return 1
        image = image.format(owner=owner)

    job_dir = args.job
    worker_zip = package_worker(job_dir / "staging", args.calibration or None)
    # Unpack the overlay over /opt/vw, then exec the entrypoint from there.
    bootstrap = (
        "import os,sys,zipfile; "
        "zipfile.ZipFile(os.path.join(os.environ['VW_IN'],'worker.zip')).extractall('/opt/vw'); "
        "os.execv(sys.executable,[sys.executable,'/opt/vw/recon_entrypoint.py',*sys.argv[1:]])"
    )
    entry = [
        "--sfm-only",
        "--downscale", "1",
        "--keep-checkpoint",
        "--sift-num-threads", str(args.sift_num_threads),
        "--match-num-threads", str(args.match_num_threads),
        "--sift-max-num-features", str(args.sift_max_num_features),
        # Bundle adjustment, not matching, is what makes a thousand-image mapper
        # run take hours. See mapper_ba_options in the entrypoint.
        "--ba-global-max-refinements", str(args.ba_global_max_refinements),
        "--ba-global-ratio", str(args.ba_global_ratio),
        "--ba-global-max-num-iterations", str(args.ba_global_max_num_iterations),
        "--ba-global-function-tolerance", str(args.ba_global_function_tolerance),
    ]
    if args.ba_use_gpu:
        entry.append("--ba-use-gpu")
    if args.calibration:
        # Rides inside worker.zip, which the bootstrap extracts to /opt/vw, so
        # there is no second input to stage and no path to guess. Without it
        # COLMAP guesses the focal and verifies every pair as UNCALIBRATED —
        # measured at 0% calibrated pairs and 36 median matches per pair on the
        # 14 Sep run, against 85% and 754 with the params supplied.
        entry += ["--calibration", "/opt/vw/calibration.json"]
    command = ["python", "-c", bootstrap, *entry]
    (job_dir / "reconstruction" / "remote_out").mkdir(parents=True, exist_ok=True)
    remote_out = job_dir / "reconstruction" / "remote_out"

    estimate = preset.sfm_cost()
    print(f"[colmap] image={image} flavor={args.flavor or ['cpu-upgrade']}")
    print(f"[colmap] est {preset.sfm_est_minutes:.0f} min, ~${estimate.est_usd:.2f}; "
          f"waiting up to {args.scheduling_timeout:.0f}s per flavor for spot capacity")
    if not args.yes:
        print("[colmap] not submitting — pass --yes to approve the cost.")
        return 0

    runner = HfJobsStageRunner(
        config=config,
        confirm_cost=lambda est: print(f"[colmap] cost pre-approved: {est.summary()}") or True,
    )
    ctx = StageContext(
        job_dir=job_dir,
        job_id=job_dir.name,
        stage_key="reconstruction_sfm",
        params={
            "image": image,
            "image_has_hub": True,
            "flavor": args.flavor or ["cpu-upgrade"],
            "est_minutes": preset.sfm_est_minutes,
            "timeout_seconds": None if args.no_timeout else (preset.sfm_timeout_seconds or 3600),
            "command": command,
            "extra_repo_inputs": [],
            "flavor_scheduling_timeout_seconds": args.scheduling_timeout,
        },
        inputs=[args.frames, worker_zip],
        # processed_min.zip is the whole point: transforms + sparse_pc.ply +
        # the sparse model with its tracks, all from this one reconstruction.
        expected_outputs=[job_dir / "reconstruction_sfm" / "remote_out" / "processed_min.zip"],
        log=lambda msg: print(f"[colmap:hf] {msg}"),
        cancel=CancelToken(),
        skip_inputs_upload=False,
    )

    result = runner.run(ctx)
    print(f"[colmap] result: {result.status}")
    for artifact in result.artifacts:
        print(f"[colmap]   {artifact}")
    if result.status == "complete":
        print("\n[colmap] next: unpack depths.zip, then\n"
              f"  python tools/build_hybrid_seed.py --colmap <colmap processed_min.zip> "
              f"--depths {remote_out / 'depths'} --out {job_dir / 'hybrid'}\n"
              "  then READ scale_spread before spending anything on training.")
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
