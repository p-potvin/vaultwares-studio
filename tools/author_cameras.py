"""Generate a camera route for a reconstruction from its geometry.

Writes usd/captured_cameras.json (the same file the viewport's Capture
Camera produces), so the usd_cameras stage threads the stops into a
"Captured Walkthrough" render path. Any existing captures are backed up.

The default "spiral" pattern descends around the scene's robust bounding
volume: establishing high shot -> sweeping mid orbit -> low close pass,
always aimed slightly below the centroid (where the ground content lives).

Usage:
    .venv\\Scripts\\python.exe tools\\author_cameras.py [job-dir] [--stops 7]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from plyfile import PlyData  # noqa: E402

from vaultwares_studio.pipeline import (  # noqa: E402
    completed_stage_count,
    list_job_manifests,
    load_job_manifest,
)


def latest_started_job_dir() -> Path | None:
    for manifest_path in list_job_manifests():
        try:
            manifest = load_job_manifest(manifest_path)
        except Exception:  # noqa: BLE001
            continue
        if completed_stage_count(manifest) > 0:
            return Path(manifest.output_dir)
    return None


def scene_bounds(preview_ply: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(preview_ply))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    low, high = np.percentile(points, [5, 95], axis=0)
    return (low + high) / 2, low, high


def spiral_cameras(center: np.ndarray, low: np.ndarray, high: np.ndarray, stops: int) -> list[dict]:
    extent = high - low
    radius = float(np.linalg.norm(extent[[0, 2]]) / 2)  # horizontal radius
    height_span = float(extent[1])
    # Aim slightly below the centroid: outdoor scenes put content near the ground.
    target = center - np.array([0.0, 0.25 * height_span, 0.0])
    cameras = []
    for index in range(stops):
        u = index / max(1, stops - 1)  # 0 -> 1 along the route
        angle = 2 * math.pi * 1.15 * u  # slightly more than one revolution
        # Stay below the canopy: outdoor scene bounds include overhanging
        # trees, so high/wide stops end up inside foliage splats.
        orbit_radius = radius * (0.72 - 0.32 * u)  # spiral inward
        height = center[1] + height_span * (0.30 - 0.28 * u)  # descend to ground level
        position = [
            float(center[0] + orbit_radius * math.cos(angle)),
            float(height),
            float(center[2] + orbit_radius * math.sin(angle)),
        ]
        cameras.append(
            {
                "name": f"Spiral {index + 1}",
                "position": position,
                "lookAt": [float(v) for v in target],
                "fovDegrees": 60,
            }
        )
    return cameras


def walk_cameras(job_dir: Path, stops: int) -> list[dict] | None:
    """Stops along the ORIGINAL capture trajectory (smoothed) — guaranteed
    free space, since the user physically walked it. Requires the render
    bundle (processed_min.zip: transforms.json raw poses; model.zip:
    dataparser_transforms.json mapping raw poses into the trained frame).
    """
    import zipfile

    remote_out = job_dir / "reconstruction" / "remote_out"
    transforms_zip = remote_out / "processed_min.zip"
    model_zip = remote_out / "model.zip"
    if not (transforms_zip.exists() and model_zip.exists()):
        return None
    with zipfile.ZipFile(transforms_zip) as archive:
        if "transforms.json" not in archive.namelist():
            return None
        transforms = json.loads(archive.read("transforms.json"))
    with zipfile.ZipFile(model_zip) as archive:
        candidates = [n for n in archive.namelist() if n.endswith("dataparser_transforms.json")]
        if not candidates:
            return None
        dataparser = json.loads(archive.read(candidates[0]))

    frames = sorted(transforms.get("frames", []), key=lambda f: f.get("file_path", ""))
    if len(frames) < stops:
        return None
    raw_positions = np.array(
        [np.array(frame["transform_matrix"])[:3, 3] for frame in frames], dtype=np.float64
    )
    normalize = np.array(dataparser["transform"], dtype=np.float64)  # 3x4
    scale = float(dataparser["scale"])
    homogeneous = np.hstack([raw_positions, np.ones((len(raw_positions), 1))])
    positions = scale * (homogeneous @ normalize.T)

    # Moving-average smoothing kills handheld jitter.
    window = max(3, len(positions) // 25) | 1
    kernel = np.ones(window) / window
    smoothed = np.column_stack(
        [np.convolve(positions[:, axis], kernel, mode="valid") for axis in range(3)]
    )

    indices = np.linspace(0, len(smoothed) - 1, stops).astype(int)
    ahead = max(1, len(smoothed) // 12)
    cameras = []
    for order, index in enumerate(indices, start=1):
        target_index = min(len(smoothed) - 1, index + ahead)
        if target_index == index:
            target_index = index - ahead  # walk's end: look back along the path
        cameras.append(
            {
                "name": f"Walk {order}",
                "position": [float(v) for v in smoothed[index]],
                "lookAt": [float(v) for v in smoothed[target_index]],
                "fovDegrees": 60,
            }
        )
    return cameras


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", nargs="?", default=None)
    parser.add_argument("--stops", type=int, default=7)
    parser.add_argument("--pattern", choices=["auto", "walk", "spiral"], default="auto")
    args = parser.parse_args()

    job_dir = Path(args.job_dir) if args.job_dir else latest_started_job_dir()
    if job_dir is None:
        print("No job with completed stages found.")
        return 1
    preview = job_dir / "reconstruction" / "cloud_preview.ply"
    if not preview.exists():
        print(f"No preview cloud at {preview} — run a reconstruction first.")
        return 1

    cameras = None
    if args.pattern in ("auto", "walk"):
        cameras = walk_cameras(job_dir, args.stops)
        if cameras:
            print(f"pattern: walk ({len(cameras)} stops along the capture trajectory)")
        elif args.pattern == "walk":
            print("walk pattern needs the render bundle (model.zip + processed_min.zip); falling back to spiral")
    if cameras is None:
        center, low, high = scene_bounds(preview)
        cameras = spiral_cameras(center, low, high, args.stops)
        print("pattern: spiral")

    store = job_dir / "usd" / "captured_cameras.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    if store.exists():
        backup = store.with_name(f"captured_cameras.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        backup.write_text(store.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"existing captures backed up: {backup.name}")
    store.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    print(f"wrote {len(cameras)} cameras to {store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
