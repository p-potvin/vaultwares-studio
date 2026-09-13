"""Materialize a completed Splatfacto artifact directory as a Studio job.

The HF Jobs download layout intentionally mirrors the remote artifact prefix;
the desktop loader expects a Studio job root containing ``manifest.json`` and
``reconstruction/cloud.ply``.  This importer creates that bridge without
discarding the checkpoint, source-frame archive, or training handoff.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = Path("D:/vaultwares-studio-jobs/data/jobs")
sys.path.insert(0, str(ROOT))


def _link_or_copy(source: Path, target: Path) -> None:
    """Expose immutable downloaded artifacts without duplicating disk usage."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _read_transforms(archive_path: Path, target: Path) -> int:
    with zipfile.ZipFile(archive_path) as archive:
        try:
            payload = archive.read("transforms.json")
        except KeyError as exc:
            raise ValueError("processed_min.zip is missing transforms.json") from exc
    parsed = json.loads(payload)
    frames = parsed.get("frames", [])
    if not frames:
        raise ValueError("transforms.json contains no camera frames")
    target.write_bytes(payload)
    return len(frames)


def import_artifact(artifact_dir: Path, job_id: str | None = None, *, require_d: bool = True) -> Path:
    artifact_dir = artifact_dir.resolve(strict=True)
    if require_d and artifact_dir.drive.upper() != "D:":
        raise ValueError("The downloaded artifact directory must be on D:.")
    required = {"splat.ply", "model.zip", "processed_min.zip", "frames.zip", "training_input.zip", "summary.json", "stage.log", "run_result.json"}
    present = {path.name for path in artifact_dir.iterdir() if path.is_file()}
    if missing := required - present:
        raise ValueError(f"Artifact directory is missing: {sorted(missing)}")

    from vaultwares_studio.splat_io import is_gaussian_ply

    source_splat = artifact_dir / "splat.ply"
    if not is_gaussian_ply(source_splat):
        raise ValueError("splat.ply is not a trained Gaussian-splat PLY")

    job_id = job_id or f"splatfacto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    job_dir = JOBS_ROOT / job_id
    if job_dir.exists():
        raise FileExistsError(f"Job already exists: {job_dir}")
    recon = job_dir / "reconstruction"
    remote = recon / "remote_out"
    remote.mkdir(parents=True)

    # Keep every retained remote artifact accessible under the conventional
    # remote_out tree. Hard links keep the download as the canonical bytes.
    for source in sorted(path for path in artifact_dir.iterdir() if path.is_file()):
        _link_or_copy(source, remote / source.name)
    _link_or_copy(remote / "splat.ply", recon / "cloud.ply")
    _link_or_copy(remote / "splat.ply", recon / "gsplat_export" / "splat.ply")
    frame_count = _read_transforms(remote / "processed_min.zip", recon / "transforms.json")

    timestamp = datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M")
    summary = json.loads((remote / "summary.json").read_text(encoding="utf-8"))
    manifest = {
        "job_id": job_id,
        "source_video": "IMG_1274.MOV",
        "output_dir": str(job_dir),
        "execution_profile": "HF Jobs L4 Splatfacto",
        "mode": "batch",
        "state": "complete",
        "current_stage_key": "reconstruction",
        "walkthrough_video": None,
        "live_viewer_supported": True,
        "schema_version": 2,
        "created_at": timestamp,
        "updated_at": timestamp,
        "metadata": {
            "imported": True,
            "import_kind": "splatfacto_artifact",
            "source_artifact_dir": str(artifact_dir),
            "frame_count": frame_count,
            "gaussian_splat": True,
            "worker_summary": summary,
        },
        "spend_ledger": [],
        "stages": [
            {"key": "video_intake", "title": "Video Intake", "description": "Retained source frames from L4 training.", "placement": "local", "state": "complete", "artifacts": []},
            {"key": "frame_extraction", "title": "Frame Extraction", "description": "500 training frames retained in remote_out/frames.zip.", "placement": "local", "state": "complete", "artifacts": []},
            {"key": "reconstruction", "title": "Reconstruction", "description": "Trained Splatfacto Gaussian splat.", "placement": "remote", "state": "complete", "message": "Imported completed L4 Splatfacto artifact.", "artifacts": [
                {"label": "Gaussian splat", "kind": "ply", "path": str(recon / "cloud.ply"), "description": "Trained Gaussian-splat export."},
                {"label": "Checkpoint", "kind": "archive", "path": str(remote / "model.zip"), "description": "Splatfacto training checkpoint."},
            ]},
            {"key": "camera_staging", "title": "Camera Staging", "description": "Compose cameras.", "placement": "local", "state": "queued", "artifacts": []},
            {"key": "cosmos_output", "title": "Cosmos + Output", "description": "Optional output.", "placement": "local", "state": "queued", "artifacts": []},
        ],
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return job_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    print(import_artifact(args.artifact_dir, args.job_id))
