"""Several videos, one map: build a multi-capture DA3-Streaming SfM job.

The console takes one video and caps at 500 frames. To grow a scene from more
than one capture the frames have to be posed *together*, in one sequence, so
DA3-Streaming's SIM3 chunk alignment and loop closure can stitch them where
they overlap. That is the whole mechanism: splatfacto is never told where the
seams are, because by the time it runs there are no seams — the poses are
already in one world.

    videos -> ffmpeg candidates per video
           -> sharpest frame per time bucket, per video
           -> one ordered sequence, videos back to back
           -> frames.zip + sequence_manifest.json
           -> HF Jobs: da3_entrypoint --stream-sfm --stream-loop-closure
           -> processed_min.zip, ready for the training leg unchanged

Two things to know before spending on this:

**Order matters.** Streaming assumes consecutive frames are close together. It
gets one hard cut per video boundary, which it handles the way it handles any
low-overlap pair — badly, unless loop closure finds the revisit. So loop
closure is on by default here, unlike the single-video path.

**Lighting matters more.** splatfacto has no per-image appearance model, so
mixing a sunny capture with an overcast one puts two different colours on the
same surface and the optimiser splits the difference. Group captures by
weather. This tool will let you mix them and will say so in the manifest, but
the splat is the place it shows up.

    .venv\\Scripts\\python.exe tools\\compose_multi_video.py \\
        --video "D:\\...\\cloudyday1_june14_194sec.MOV" \\
        --video "D:\\...\\cloudyday2_june14_348sec.MOV" \\
        --output "D:\\...\\sep13\\cloudy-pair" --per-video 400 [--submit]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vaultwares_studio.frame_selection import select_sharpest_frames  # noqa: E402

DA3_IMAGE = "hf.co/spaces/clopeux/vw-studio-da3-gs"
STREAM_RESOLUTION = "504x280"
STREAM_CHUNK = 80
STREAM_OVERLAP = 40


def probe(video: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration,size:stream=width,height,avg_frame_rate",
        "-of", "json", str(video)], text=True)
    return json.loads(out)


def extract_candidates(video: Path, target: Path, fps: int) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    existing = sorted(target.glob("*.jpg"))
    if existing:
        return existing
    subprocess.run([
        "ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(video),
        "-vf", f"fps={fps}", "-q:v", "2", str(target / "candidate_%05d.jpg"),
    ], check=True, capture_output=True, timeout=1800)
    return sorted(target.glob("*.jpg"))


def compose(videos: list[Path], output: Path, per_video: int, fps: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sequence: list[dict] = []
    clips: list[dict] = []
    index = 0
    for clip_index, video in enumerate(videos):
        info = probe(video)
        duration = float(info["format"]["duration"])
        candidates = extract_candidates(video, output / "candidates" / video.stem, fps)
        selected = select_sharpest_frames(candidates, per_video)
        # Both ends of every clip, so a revisit at a boundary is representable.
        if len(selected) >= 2:
            selected[0], selected[-1] = candidates[0], candidates[-1]
        start = index
        for source in selected:
            sequence.append({
                "index": index, "clip": clip_index, "video": video.name, "source": str(source),
                "source_seconds": round((int(source.stem.split("_")[-1]) - 1) / fps, 3),
            })
            index += 1
        clips.append({
            "clip": clip_index, "video": video.name, "duration": round(duration, 3),
            "candidates": len(candidates), "selected": len(selected),
            "first_index": start, "last_index": index - 1,
        })
        print(f"[compose] {video.name}: {duration:.0f}s, {len(candidates)} candidates -> "
              f"{len(selected)} frames at indices {start}..{index - 1}")

    frames_zip = output / "frames.zip"
    with zipfile.ZipFile(frames_zip, "w", zipfile.ZIP_STORED) as archive:
        for entry in sequence:
            name = f"frame_{entry['index']:05d}.jpg"
            entry["frame"] = name
            archive.write(entry["source"], name)

    first = probe(videos[0])["streams"][0]
    manifest = {
        "generated": datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M"),
        "videos": [str(v) for v in videos],
        "clips": clips,
        "frames": len(sequence),
        "candidate_fps": fps,
        "per_video": per_video,
        "source_size": [int(first["width"]), int(first["height"])],
        "stream": {"resolution": STREAM_RESOLUTION, "chunk": STREAM_CHUNK,
                   "overlap": STREAM_OVERLAP, "loop_closure": True},
        "boundaries": [clip["first_index"] for clip in clips[1:]],
        "note": ("Frames are ordered clip after clip. Each boundary is a hard cut; "
                 "loop closure is what joins the captures where they overlap."),
        "sequence": sequence,
    }
    (output / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["frames_zip_bytes"] = frames_zip.stat().st_size
    return manifest


def submit(output: Path, manifest: dict, flavors: list[str], scheduling_timeout: float) -> int:
    """Run the streaming SfM leg on HF Jobs. Poses only; training is separate."""
    from vaultwares_studio.pipeline import (
        DigitalTwinStudioRunner, StageState, create_job_manifest, save_job_manifest,
    )
    from vaultwares_studio.runners import CancelToken, HfJobsConfig, HfJobsStageRunner, StageContext

    config = HfJobsConfig.load()
    if not config.enabled:
        raise SystemExit("Remote compute not enabled (data/remote_compute.json).")

    job = create_job_manifest(Path(manifest["videos"][0]))
    job.metadata.update({
        "preset": "da3-stream", "multi_video": True,
        "videos": [Path(v).name for v in manifest["videos"]],
        "clips": manifest["clips"], "frame_count": manifest["frames"],
        "boundaries": manifest["boundaries"], "candidate_fps": manifest["candidate_fps"],
        "source_size": manifest["source_size"], "sequence_manifest": str(output / "sequence_manifest.json"),
    })
    for stage in job.stages:
        if stage.key in ("video_intake", "frame_extraction"):
            stage.state = StageState.COMPLETE.value
    save_job_manifest(job)
    job_dir = Path(job.output_dir)
    (job_dir / "reconstruction").mkdir(parents=True, exist_ok=True)
    # The sequence manifest travels with the job; without it the frame indices
    # in transforms.json cannot be traced back to a video and a timestamp.
    (job_dir / "reconstruction" / "sequence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(output / "worker.zip", "w", zipfile.ZIP_DEFLATED) as worker:
        worker.write(ROOT / "docker/worker/da3_entrypoint.py", "da3_entrypoint.py")
        worker.write(ROOT / "vaultwares_studio/streaming_convert.py", "streaming_convert.py")
        worker.write(ROOT / "vaultwares_studio/camera_calibration.py", "camera_calibration.py")
    overlay = (
        "import os,sys,zipfile; "
        "zipfile.ZipFile(os.path.join(os.environ['VW_IN'],'worker.zip')).extractall('/opt/vw'); "
        "os.execv(sys.executable,[sys.executable,'/opt/vw/da3_entrypoint.py',*sys.argv[1:]])"
    )
    command = ["python", "-c", overlay,
               "--stream-sfm", "--da3-model", "depth-anything/DA3-LARGE-1.1",
               "--stream-resolution", STREAM_RESOLUTION,
               "--stream-chunk-size", str(STREAM_CHUNK),
               "--stream-overlap", str(STREAM_OVERLAP),
               "--stream-loop-closure"]
    # Measured: 500 frames posed in 120 s on a10g-small, plus loop closure over
    # every frame and the model pull.
    est_minutes = max(10.0, manifest["frames"] / 500 * 3.0 + 8)
    recon = job_dir / "reconstruction"
    log_path = output / "sfm.log"
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    ctx = StageContext(
        job_dir=job_dir, job_id=job.job_id, stage_key="reconstruction_sfm",
        params={
            "image": DA3_IMAGE, "image_has_hub": True, "flavor": flavors,
            "est_minutes": est_minutes, "timeout_seconds": int(est_minutes * 60 * 3),
            "command": command, "extra_repo_inputs": [],
            "flavor_scheduling_timeout_seconds": scheduling_timeout,
        },
        inputs=[output / "frames.zip", output / "worker.zip"],
        expected_outputs=[recon / "remote_out" / "processed_min.zip",
                          recon / "remote_out" / "summary.json"],
        log=log, cancel=CancelToken(),
    )
    log(f"[multi] {manifest['frames']} frames from {len(manifest['clips'])} clips, "
        f"loop closure on, est {est_minutes:.0f} min on {flavors}")
    runner = HfJobsStageRunner(
        config=config, confirm_cost=lambda est: log(f"[multi] cost approved: {est.summary()}") or True)
    started = time.time()
    result = runner.run(ctx)
    log(f"[multi] {result.status} in {time.time() - started:.0f}s; {result.metadata}")
    print(json.dumps({"job_id": job.job_id, "job_dir": str(job_dir),
                      "status": result.status}, indent=2))
    return 0 if result.status == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, action="append", required=True,
                        help="source video; repeat, in the order they should be walked")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-video", type=int, default=400)
    parser.add_argument("--fps", type=int, default=7, help="candidate extraction rate")
    parser.add_argument("--flavor", action="append")
    parser.add_argument("--scheduling-timeout", type=float, default=600.0)
    parser.add_argument("--submit", action="store_true", help="run the paid SfM job after composing")
    args = parser.parse_args()
    videos = [v.resolve(strict=True) for v in args.video]
    manifest = compose(videos, args.output.resolve(), args.per_video, args.fps)
    print(json.dumps({k: manifest[k] for k in
                      ("frames", "clips", "boundaries", "frames_zip_bytes", "source_size")}, indent=2, default=str))
    if not args.submit:
        return 0
    return submit(args.output.resolve(), manifest, args.flavor or ["a10g-small", "l4x1"],
                  args.scheduling_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
