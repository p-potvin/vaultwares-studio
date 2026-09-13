"""Train a splat from a ZeroGPU DA3 artifact: rebuild its frames, submit Job B.

The console selects its 500 frames itself (ffmpeg at ``fps`` then one
candidate per time bucket) and only ever returns poses. Training needs the
same frames at full resolution, so this replays that selection locally from
the source video, bundles ``transforms.json`` + ``sparse_pc.ply`` + the images,
and runs the worker's ``--train-only`` leg on HF Jobs with the current
entrypoint overlaid on the image (no rebuild).

    prepare   decode candidates, select, write frames.zip + processed_min.zip
    submit    upload the bundle and run splatfacto (paid; asks nothing else)

    .venv\\Scripts\\python.exe tools\\prepare_zerogpu_training.py --job zerogpu-... \\
        --video "D:\\...\\backyard_134s_sunny.mp4" --candidates "D:\\...\\candidates" \\
        --iterations 20000 --submit
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spaces" / "da3-zerogpu"))

from core import selected_frame_indices  # noqa: E402  (the console's own selector)

DA3_IMAGE = "hf.co/spaces/clopeux/vw-studio-da3"

# Train args. Two knobs here, and the history of getting them wrong is worth
# keeping because the obvious explanation was the wrong one.
#
# `stop_split_at` must track the iteration count: splatfacto defaults it to
# 15000, so a 20000-iteration run densifies for 15000 and then spends 5000
# iterations culling with nothing growing back. That is real and is fixed here.
#
# It is NOT, however, what made the 13 Sep splat narrow. Measured afterwards:
#
#   run              views  iters  stop_split  gaussians  extent/path
#   july              80    15000    15000     1,839,412     3.84
#   august           500    15000    15000       956,304       ?
#   13 Sep first     500    20000    15000     1,036,140     0.51
#   13 Sep refine    500    45000    40500       619,395     0.52
#
# Both 500-view runs land near a million whether or not they had a culling
# tail, and the 45000-iteration refine — densifying all the way to 40500 —
# came out *smaller* and no wider. Densification splits gaussians that already
# exist; it cannot invent them where the far field was already pruned. So the
# separator is the view count, not this flag: with 500 views every gaussian has
# to satisfy far more observations, and the far field is exactly where DA3's
# depth and poses are least reliable, so that is what gets culled.
#
# Scale regularisation was tried on 13 Sep and is dropped: unproven, and the
# August run without it was just as narrow.
SPLIT_FRACTION = 0.9


def base_train_args(iterations: int) -> list[str]:
    return [
        "--max-num-iterations", str(iterations),
        "--vis", "none",
        "--viewer.quit-on-train-completion", "True",
        "--pipeline.datamanager.cache-images", "cpu",
        "--steps-per-save", "1000",
        # Densify almost to the end; see above.
        "--pipeline.model.stop-split-at", str(int(iterations * SPLIT_FRACTION)),
        # Gentler than the 0.1 default: far-field gaussians are faint and
        # legitimately so, and the periodic alpha reset already prunes the
        # genuinely dead ones.
        "--pipeline.model.cull-alpha-thresh", "0.05",
    ]


def _job_dir(job_id: str) -> Path:
    for base in (ROOT / "data" / "jobs", Path("D:/vaultwares-studio-jobs/data/jobs")):
        if (base / job_id / "manifest.json").exists():
            return base / job_id
    raise SystemExit(f"No manifest.json found for {job_id}")


def replay_selection(candidates_dir: Path, frame_count: int, expected_candidates: int | None) -> list[Path]:
    frames = sorted(candidates_dir.glob("*.jpg"))
    if expected_candidates and len(frames) != expected_candidates:
        raise SystemExit(
            f"{len(frames)} local candidates but the console saw {expected_candidates}; "
            "extract with the same ffmpeg fps filter before training"
        )
    return [frames[i] for i in selected_frame_indices(len(frames), frame_count)]


def _package_worker(staging: Path) -> Path:
    """The worker overlay: the current entrypoint plus the retention wrapper,
    so the image built in July runs today's code without a rebuild."""
    target = staging / "worker.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as worker:
        worker.write(ROOT / "docker/worker/da3_entrypoint.py", "da3_entrypoint.py")
        worker.write(ROOT / "vaultwares_studio/streaming_convert.py", "streaming_convert.py")
        worker.write(ROOT / "vaultwares_studio/camera_calibration.py", "camera_calibration.py")
        worker.write(ROOT / "tools/run_train_only_with_retention.py", "run_train_only_with_retention.py")
    return target


def prepare(job_dir: Path, candidates_dir: Path, staging: Path, subsample: int = 1) -> dict:
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    meta = manifest["metadata"]
    frame_count = int(meta["frame_count"])
    selected = replay_selection(candidates_dir, frame_count, meta.get("candidate_frames"))
    staging.mkdir(parents=True, exist_ok=True)
    processed = job_dir / "reconstruction" / "remote_out" / "processed"
    transforms = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
    if len(transforms["frames"]) != len(selected):
        raise SystemExit(f"{len(transforms['frames'])} poses vs {len(selected)} selected frames")
    if subsample > 1:
        # Keep every Nth posed frame. The poses are untouched — this changes
        # only how many views supervise training, which is the variable that
        # actually separated the wide July splat from the narrow 500-view ones.
        order = sorted(range(len(selected)),
                       key=lambda i: transforms["frames"][i]["file_path"])
        keep = sorted(order[::subsample])
        transforms = {**transforms, "frames": [transforms["frames"][i] for i in keep]}
        selected = [selected[i] for i in keep]
    frames_zip = staging / "frames.zip"
    bundle = staging / "processed_min.zip"
    name_map = {}
    with zipfile.ZipFile(frames_zip, "w", zipfile.ZIP_STORED) as frames_archive, \
            zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as bundle_archive:
        bundle_archive.writestr("transforms.json", json.dumps(transforms, indent=1))
        bundle_archive.write(processed / "sparse_pc.ply", "sparse_pc.ply")
        for index, source in enumerate(selected):
            name = Path(transforms["frames"][index]["file_path"]).name
            name_map[name] = source.name
            frames_archive.write(source, name)
            bundle_archive.write(source, f"images/{name}")
        bundle_archive.writestr("frame_name_map.json", json.dumps(name_map, indent=2))
    _package_worker(staging)
    report = {
        "job_id": manifest["job_id"], "frames": len(selected), "subsample": subsample,
        "candidates": len(list(candidates_dir.glob("*.jpg"))),
        "first_frame": selected[0].name, "last_frame": selected[-1].name,
        "frames_zip_bytes": frames_zip.stat().st_size, "bundle_bytes": bundle.stat().st_size,
        "intrinsics_size": meta.get("intrinsics_size"), "fl_x": transforms["frames"][0]["fl_x"],
        "cx": transforms["frames"][0]["cx"], "cy": transforms["frames"][0]["cy"],
    }
    (staging / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def submit(job_dir: Path, staging: Path, iterations: int, flavors: list[str], scheduling_timeout: float,
           refine: bool = False) -> int:
    from vaultwares_studio.pipeline import load_job_manifest, record_spend
    from vaultwares_studio.runners import CancelToken, HfJobsConfig, HfJobsStageRunner, StageContext
    from vaultwares_studio.runners.hf_jobs import BOOTSTRAP_SOURCE  # noqa: F401  (image_has_hub path)

    manifest = load_job_manifest(job_dir / "manifest.json")
    recon = job_dir / "reconstruction"
    remote_out = recon / "remote_out"
    (recon / "gsplat_export").mkdir(parents=True, exist_ok=True)
    train_args = base_train_args(iterations)
    overlay = (
        "import os,sys,zipfile; "
        "zipfile.ZipFile(os.path.join(os.environ['VW_IN'],'worker.zip')).extractall('/opt/vw'); "
        "os.execv(sys.executable,[sys.executable,'/opt/vw/run_train_only_with_retention.py',*sys.argv[1:]])"
    )
    worker = [
        "python", "/opt/vw/da3_entrypoint.py", "--train-only", "--downscale", "1",
        "--train-args", json.dumps(train_args), "--keep-checkpoint",
    ]
    if refine:
        # Resume the banked checkpoint rather than starting over. nerfstudio
        # continues at the checkpoint's own step, so `iterations` is the new
        # TOTAL and stop_split_at (0.9x of it) lands well past the resume
        # point. Without that a resumed run would only cull, which is exactly
        # the failure this change is about.
        worker.append("--refine-mode")
    command = ["python", "-c", overlay, *worker]
    est_minutes = max(20.0, iterations / 650.0 + 4)  # measured 655 iter/min on an L4
    config = HfJobsConfig.load()
    runner = HfJobsStageRunner(config=config, confirm_cost=lambda est: print(f"[train] cost approved: {est.summary()}") or True)
    log_path = staging / "train.log"
    log_file = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    ctx = StageContext(
        job_dir=job_dir, job_id=manifest.job_id, stage_key="reconstruction",
        params={
            "image": DA3_IMAGE, "image_has_hub": True, "flavor": flavors,
            "est_minutes": est_minutes, "timeout_seconds": int(est_minutes * 60 * 2.5),
            "command": command,
            # A refine reuses what the first run already banked in the
            # artifact dataset: ~750 MB that never leaves the Hub.
            "extra_repo_inputs": ([
                f"jobs/{manifest.job_id}/reconstruction/out/frames.zip",
                f"jobs/{manifest.job_id}/reconstruction/out/processed_min.zip",
                f"jobs/{manifest.job_id}/reconstruction/out/model.zip",
            ] if refine else []),
            "flavor_scheduling_timeout_seconds": scheduling_timeout,
        },
        inputs=([staging / "worker.zip"] if refine
                else [staging / "frames.zip", staging / "processed_min.zip", staging / "worker.zip"]),
        expected_outputs=[recon / "gsplat_export" / "splat.ply", recon / "summary.json",
                          remote_out / "model.zip", remote_out / "processed_min.zip"],
        log=log, cancel=CancelToken(),
    )
    log(f"[train] {iterations} iterations on {flavors}, est {est_minutes:.0f} min, args {train_args}")
    started = time.time()
    result = runner.run(ctx)
    if result.metadata:
        record_spend(manifest, "reconstruction", result.metadata)
    log(f"[train] job {result.status} in {time.time() - started:.0f}s; {result.metadata}")
    if result.status != "complete":
        return 1
    code = finish(job_dir, log)
    # Logged after the local tail so a watcher keyed on this line sees the
    # gravity-aligned summary.json, not a splat mid-conversion.
    log(f"[train] {'complete' if code == 0 else 'failed'}: local tail exit {code}")
    return code


def finish(job_dir: Path, log) -> int:
    """The reconstruction tail the pipeline runs after a remote job: convert,
    gravity-align, pack for the viewer, and mark the stage complete."""
    from vaultwares_studio.pipeline import DigitalTwinStudioRunner, StageState, load_job_manifest, save_job_manifest
    from vaultwares_studio.splat_io import convert_splat_outputs, is_gaussian_ply, read_gaussian_ply, splat_to_usd

    manifest = load_job_manifest(job_dir / "manifest.json")
    runner = DigitalTwinStudioRunner(manifest, log)
    splat_path = runner.recon_dir / "gsplat_export" / "splat.ply"
    if not splat_path.exists() or not is_gaussian_ply(splat_path):
        log(f"[train] splat.ply missing or not a gaussian PLY: {splat_path}")
        return 1
    stage = runner.stage_for("reconstruction")
    # The point-cloud preview from the import is replaced by the trained splat.
    for stale in (runner.recon_ply_path, runner.recon_preview_ply_path, runner.recon_splat_path):
        stale.unlink(missing_ok=True)
    info = convert_splat_outputs(splat_path, runner.recon_ply_path, runner.recon_preview_ply_path,
                                 runner.recon_stage_path, log)
    stage.metadata.update(info)
    runner._gravity_align(stage)
    stage.metadata["usd_mode"] = splat_to_usd(read_gaussian_ply(runner.recon_ply_path), runner.recon_stage_path,
                                              source=runner.recon_ply_path.name)
    runner._write_packed_splat(stage)
    stage.metadata["degraded"] = False
    stage.metadata["trained"] = True
    stage.message = f"Trained splat ({stage.metadata.get('gaussians', '?')} gaussians) on the ZeroGPU poses."
    stage.state = StageState.COMPLETE.value
    runner._add_artifact(stage, "Reconstruction Stage", "usd", runner.recon_stage_path, "Reconstruction stage.")
    runner._add_artifact(stage, "Reconstruction PLY", "ply", runner.recon_ply_path, "Gaussian splat output.")
    manifest.current_stage_key = "camera_staging"
    manifest.state = StageState.QUEUED.value
    save_job_manifest(manifest)
    log("[train] reconstruction complete; next stage: camera_staging")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, help="imported ZeroGPU job id")
    parser.add_argument("--candidates", type=Path, help="folder of ffmpeg fps=N candidates (fresh runs only)")
    parser.add_argument("--staging", type=Path, help="where the bundles go (default: <job>/training_input)")
    parser.add_argument("--iterations", type=int, default=15_000)
    parser.add_argument("--subsample", type=int, default=1,
                        help="train on every Nth posed view (poses unchanged). 6 turns 500 views "
                             "into 84, which is July's configuration.")
    parser.add_argument("--flavor", action="append", help="flavor candidates in order; default l4x1 then a10g-small")
    parser.add_argument("--scheduling-timeout", type=float, default=600.0)
    parser.add_argument("--submit", action="store_true", help="run the paid training job after preparing")
    parser.add_argument("--refine", action="store_true",
                        help="resume this job's banked checkpoint instead of training from scratch; "
                             "--iterations is the new TOTAL. Uploads nothing but the worker overlay.")
    parser.add_argument("--finish-only", action="store_true", help="only run the local tail on downloaded outputs")
    args = parser.parse_args()
    job_dir = _job_dir(args.job)
    staging = args.staging or (job_dir / "training_input")
    if args.candidates is None and not (args.refine or args.finish_only):
        parser.error("--candidates is required unless --refine or --finish-only")
    if args.finish_only:
        return finish(job_dir, print)
    flavors = args.flavor or ["l4x1", "a10g-small"]
    if args.refine:
        staging.mkdir(parents=True, exist_ok=True)
        _package_worker(staging)
        print(json.dumps({"mode": "refine", "job": args.job, "total_iterations": args.iterations,
                          "stop_split_at": int(args.iterations * SPLIT_FRACTION)}, indent=2))
        if not args.submit:
            return 0
        return submit(job_dir, staging, args.iterations, flavors, args.scheduling_timeout, refine=True)
    report = prepare(job_dir, args.candidates, staging, subsample=args.subsample)
    print(json.dumps(report, indent=2))
    if not args.submit:
        return 0
    return submit(job_dir, staging, args.iterations, flavors, args.scheduling_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
