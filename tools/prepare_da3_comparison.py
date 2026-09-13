"""Offline preparation only: decode frames and package a two-job DA3 comparison.

No credentials, network calls, model loading or job submission. The output
folder must be new; both variants use one identical frames.zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaultwares_studio.frame_selection import select_sharpest_frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_worker(output: Path) -> str:
    with zipfile.ZipFile(output / "worker.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / "docker/worker/da3_entrypoint.py", "da3_entrypoint.py")
        archive.write(ROOT / "vaultwares_studio/streaming_convert.py", "streaming_convert.py")
        archive.write(ROOT / "vaultwares_studio/camera_calibration.py", "camera_calibration.py")
        archive.write(ROOT / "tools/run_prepared_da3.py", "run_prepared_da3.py")
    return sha256(output / "worker.zip")


def job_command(worker_args: list[str]) -> list[str]:
    bootstrap = (
        "import os,sys,zipfile; "
        "zipfile.ZipFile(os.path.join(os.environ['VW_IN'],'worker.zip')).extractall('/opt/vw'); "
        "os.execv(sys.executable,[sys.executable,'/opt/vw/run_prepared_da3.py',*sys.argv[1:]])"
    )
    return ["python", "-c", bootstrap, *worker_args]


def prepare(video: Path, output: Path) -> dict:
    video = video.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration,size:stream=width,height,avg_frame_rate:stream_side_data=rotation",
        "-of", "json", str(video)], text=True))
    (output / "video_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    candidates = output / "candidates"
    candidates.mkdir()
    # Auto-rotation honors the phone's display matrix. Keep every candidate on
    # disk; sharpness selection chooses one frame per temporal bucket.
    with (output / "frame_extraction.log").open("w", encoding="utf-8") as log:
        subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(video),
                        "-vf", "fps=7", "-q:v", "2", str(candidates / "frame_%05d.jpg")],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    frames = sorted(candidates.glob("*.jpg"))
    selected = select_sharpest_frames(frames, 500)
    if len(selected) != 500:
        raise ValueError(f"Expected 500 selected frames, got {len(selected)}.")
    # Ensure both endpoints remain represented, despite the temporal sharpness
    # preference. The candidate collection is still retained in full.
    selected[0], selected[-1] = frames[0], frames[-1]
    with zipfile.ZipFile(output / "frames.zip", "w", zipfile.ZIP_STORED) as archive:
        for frame in selected:
            archive.write(frame, frame.name)
    manifest = [{"name": p.name, "sample_time_seconds": (int(p.stem.split("_")[-1]) - 1) / 7,
                 "sha256": sha256(p)} for p in selected]
    (output / "selected_frames.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    worker_hash = package_worker(output)
    common = ["--stream-sfm", "--da3-model", "depth-anything/DA3-LARGE-1.1",
              "--stream-resolution", "504x280", "--stream-chunk-size", "60",
              "--stream-overlap", "30"]
    plan = {
        "status": "AWAITING_USER_GREENLIGHT",
        "source_video": str(video), "source_sha256": sha256(video),
        "source_duration_seconds": float(probe["format"]["duration"]),
        "candidate_frames": len(frames), "selected_frames": len(selected),
        "frames_zip_sha256": sha256(output / "frames.zip"),
        "frames_zip_bytes": (output / "frames.zip").stat().st_size,
        "worker_zip_sha256": worker_hash,
        "image": "hf.co/spaces/clopeux/vw-studio-da3-gs",
        "observed_space_sha": "89cd1561d7dd098ad25957ad741caeb306535e7b",
        "image_note": "Reuse built image with private per-job worker overlay; recheck Space SHA before launch.",
        "flavor": "l4x1", "hourly_usd": 0.80,
        "price_source": "https://huggingface.co/docs/hub/en/jobs-pricing",
        "expected_total_usd": [0.30, 0.60],
        "maximum_compute_usd": 0.80,
        "backend_timeout_seconds_per_job": 1800,
        "worker_timeout_seconds": 1200,
        "queue_timeout_seconds_per_job": 600,
        "poll_interval_seconds": 60, "maximum_status_checks_per_job": 42,
        "automatic_retries": 0, "parallel_jobs": 1,
        "submit_second_only_if_first_succeeds": True,
        "variants": [
            {"name": "loop-off", "worker_args": common},
            {"name": "loop-on", "worker_args": common + ["--stream-loop-closure"]},
        ],
        "training_jobs": 0, "remote_render_jobs": 0,
        "outputs": ["processed_min.zip", "streaming_artifacts.zip", "stage.log", "run_result.json"],
        "comparison": ["pose and intrinsic validity", "trajectory drift after similarity alignment",
                       "detected loop pairs with matching image contact sheets", "doorway/wall duplication in point clouds",
                       "depth/confidence continuity", "measured elapsed time and billed-duration estimate"],
        "remaining_runtime_validation": "First remote run; no CUDA inference performed during preparation.",
    }
    for variant in plan["variants"]:
        variant["command"] = job_command(variant["worker_args"])
        variant["inputs"] = ["frames.zip", "worker.zip"]
    (output / "comparison_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-worker", action="store_true", help="Refresh code package in an unapproved local plan")
    args = parser.parse_args()
    if args.refresh_worker:
        path = args.output / "comparison_plan.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan["status"] != "AWAITING_USER_GREENLIGHT":
            parser.error("Cannot alter a plan after approval or launch.")
        plan["worker_zip_sha256"] = package_worker(args.output)
        for variant in plan["variants"]:
            variant["command"] = job_command(variant["worker_args"])
            variant["inputs"] = ["frames.zip", "worker.zip"]
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    else:
        if args.video is None:
            parser.error("--video is required for new preparation.")
        plan = prepare(args.video, args.output)
    print(json.dumps({"plan": str((args.output / "comparison_plan.json").resolve()),
                      "frames": plan["selected_frames"], "frames_zip_bytes": plan["frames_zip_bytes"],
                      "expected_usd": plan["expected_total_usd"],
                      "maximum_compute_usd": plan["maximum_compute_usd"]}, indent=2))


if __name__ == "__main__":
    main()
