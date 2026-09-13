"""Materialize a ZeroGPU artifact ZIP as a Studio job on D:.

The console returns poses, intrinsics, the merged point cloud and (since the
September builds) the per-frame depth/confidence fields. This turns that into
the job layout the GUI and the training leg expect:

    reconstruction/cloud.ply, cloud_preview.ply     the merged DA3 point cloud
    reconstruction/transforms.json                  nerfstudio poses + intrinsics
    reconstruction/remote_out/processed/            transforms.json + sparse_pc.ply
    reconstruction/remote_out/streaming/            the raw streaming outputs,
                                                    incl. results_output/*.npz

Intrinsics resolution is inferred from the artifact, not taken from the
preset: DA3 works at its own resolution (504x280 for LARGE) whatever it was
fed, and the earlier assumption that intrinsics were in the 672x378 input
frame scaled every focal length 25% short. See streaming_convert.infer_stream_size.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

JOBS_ROOT = Path("D:/vaultwares-studio-jobs/data/jobs")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _extract(archive: zipfile.ZipFile, name: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(name) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def _source_size(input_manifest: dict) -> tuple[int, int]:
    """Original frame size from the console's ffprobe record, honouring rotation."""
    probe = input_manifest.get("probe") or {}
    width = height = 0
    for stream in probe.get("streams", []):
        if stream.get("width") and stream.get("height"):
            width, height = int(stream["width"]), int(stream["height"])
            rotation = 0
            for side in stream.get("side_data_list", []) or []:
                if "rotation" in side:
                    rotation = int(abs(float(side["rotation"])))
            if rotation in (90, 270):
                width, height = height, width
            break
    return (width, height) if width and height else (1920, 1080)


def import_artifact(archive_path: Path, job_id: str | None = None, *, require_d: bool = True) -> Path:
    archive_path = archive_path.resolve(strict=True)
    if require_d and archive_path.drive.upper() != "D:":
        raise ValueError("The source artifact must be on D:.")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        required = {"input_manifest.json", "streaming/camera_poses.txt", "streaming/intrinsic.txt", "streaming/pcd/combined_pcd.ply"}
        if missing := required - names:
            raise ValueError(f"Artifact is missing required files: {sorted(missing)}")
        input_manifest = json.loads(archive.read("input_manifest.json"))
        frame_count = int(input_manifest.get("selected_frames", 0))
        if frame_count <= 0:
            raise ValueError("input_manifest.json does not record selected_frames.")
        job_id = job_id or f"zerogpu-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job_dir = JOBS_ROOT / job_id
        if job_dir.exists():
            raise FileExistsError(f"Job already exists: {job_dir}")
        recon, remote = job_dir / "reconstruction", job_dir / "reconstruction" / "remote_out"
        streaming = remote / "streaming"
        _extract(archive, "streaming/pcd/combined_pcd.ply", recon / "cloud.ply")
        shutil.copyfile(recon / "cloud.ply", recon / "cloud_preview.ply")
        # Keep every streaming output, results_output included: the depth and
        # confidence fields are what depth_fusion turns into the mesh.
        for name in sorted(names):
            if name.startswith("streaming/") and not name.endswith("/"):
                _extract(archive, name, streaming / name[len("streaming/"):])
        (remote / "pcd").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(recon / "cloud.ply", remote / "pcd" / "combined_pcd.ply")
        for name in ["camera_poses.txt", "intrinsic.txt", "loop_closures.txt", "config.json"]:
            if f"streaming/{name}" in names:
                shutil.copyfile(streaming / name, remote / name)
        shutil.copyfile(archive_path, remote / "streaming_artifacts.zip")
        (remote / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")

        from vaultwares_studio.streaming_convert import infer_stream_size, write_processed_bundle

        fed_size = tuple(int(v) for v in input_manifest.get("gpu_input_size", (0, 0)))
        stream_size = infer_stream_size(streaming, fallback=fed_size if all(fed_size) else None)
        original_size = _source_size(input_manifest)
        write_processed_bundle(streaming, [f"frame_{i:05d}.jpg" for i in range(frame_count)], remote / "processed",
                               stream_size=stream_size, original_size=original_size)
        shutil.copyfile(remote / "processed" / "transforms.json", recon / "transforms.json")
        loop_pairs = 0
        if (streaming / "loop_closures.txt").exists():
            loop_pairs = sum(
                1 for line in (streaming / "loop_closures.txt").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            )
        timestamp = datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M")
        preset = input_manifest.get("preset") or {}
        manifest = {"job_id": job_id, "source_video": input_manifest["source_name"], "output_dir": str(job_dir),
                    "execution_profile": "ZeroGPU DA3 console", "mode": "batch", "state": "complete",
                    "current_stage_key": "reconstruction", "walkthrough_video": None,
                    "live_viewer_supported": True, "schema_version": 2, "created_at": timestamp,
                    "updated_at": timestamp, "metadata": {"preset": preset, "loop_closure": True,
                    "loop_pairs": loop_pairs, "source_artifact": str(archive_path), "imported": True,
                    "frame_count": frame_count, "gpu_input_size": list(fed_size),
                    "intrinsics_size": list(stream_size), "source_size": list(original_size),
                    "candidate_frames": input_manifest.get("candidate_frames"),
                    "candidate_fps": input_manifest.get("fps")}, "spend_ledger": [], "stages": [
                        {"key": "video_intake", "title": "Video Intake", "description": "Imported from ZeroGPU.", "placement": "local", "state": "complete", "artifacts": []},
                        {"key": "frame_extraction", "title": "Frame Extraction", "description": "Imported from ZeroGPU.", "placement": "local", "state": "complete", "artifacts": []},
                        {"key": "reconstruction", "title": "Reconstruction", "description": "DA3 Streaming output.", "placement": "remote", "state": "complete", "message": "Imported ZeroGPU DA3 Streaming artifact.", "artifacts": []},
                        {"key": "camera_staging", "title": "Camera Staging", "description": "Compose cameras.", "placement": "local", "state": "queued", "artifacts": []},
                        {"key": "cosmos_output", "title": "Cosmos + Output", "description": "Cosmos and walkthrough.", "placement": "local", "state": "queued", "artifacts": []}]}
        (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (job_dir / "usd").mkdir(exist_ok=True)
        return job_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    job_dir = import_artifact(args.archive, args.job_id)
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"job_dir": str(job_dir), **manifest["metadata"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
