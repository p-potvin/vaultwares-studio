"""Is the scale inconsistent, or is the ground just uneven?

``grid_from_depth_frames`` measures each frame's camera height above the ground
that frame sees, and on the 13 Sep capture that number swings between 0.7 and
1.9 apparent metres. I first read that as chunk scale drift. It is not that
simple: the capture walks up steps, and a frame standing on a step often sees
the floor below as well, so the plane fit takes the lower surface and the
apparent height inflates. Terrain, estimator and drift are all mixed together
in that one number.

This separates them. Two frames standing at the same physical spot, verified by
feature matching rather than by pose, must report the same camera height — the
ground under the photographer is the same ground and their arm did not move. Any
difference between such a pair is scale inconsistency and nothing else, because
the terrain is held fixed by construction.

Pairs are drawn from frames far apart in the sequence (a revisit), verified with
SIFT and a RANSAC fundamental matrix, and required to agree closely in viewing
direction so that both frames see the same ground.

    python tools/measure_scale_drift.py --job zerogpu-backyard134-loop-20260913
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.camera_scene import scene_frame_transform  # noqa: E402
from vaultwares_studio.robot_lab.occupancy import _frame_in_scene, _ground_plane  # noqa: E402
from vaultwares_studio.tsdf_volume import load_streaming_frames  # noqa: E402

JOB_ROOTS = (ROOT / "data" / "jobs", Path("D:/vaultwares-studio-jobs/data/jobs"))


def _job_dir(job_id: str) -> Path:
    for base in JOB_ROOTS:
        if (base / job_id / "manifest.json").exists():
            return base / job_id
    raise SystemExit(f"no manifest.json for {job_id}")


def camera_height(frame, transform: np.ndarray, pixel_stride: int, depth_trunc: float) -> float | None:
    """Camera height above the ground plane fitted to this frame's own depth."""
    placed = _frame_in_scene(frame, transform, pixel_stride, depth_trunc)
    if placed is None:
        return None
    surface, centre = placed
    plane = _ground_plane(surface, centre, 5.0)
    if plane is None:
        return None
    a_x, a_z, c = plane
    height = centre[1] - (a_x * centre[0] + a_z * centre[2] + c)
    return float(height) if height > 1e-9 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--min-separation", type=int, default=60,
                        help="frames apart, so a pair is a revisit and not a neighbour")
    parser.add_argument("--min-inliers", type=int, default=120)
    parser.add_argument("--max-view-angle", type=float, default=35.0,
                        help="degrees between optical axes; both frames must see the same ground")
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--depth-trunc", type=float, default=0.74)
    parser.add_argument("--stride", type=int, default=2, help="subsample frames before matching")
    parser.add_argument("--min-height-fraction", type=float, default=0.34,
                        help="reject a frame whose fitted camera height is below this "
                             "fraction of the capture's median; those are failed plane fits")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import cv2

    job_dir = _job_dir(args.job)
    streaming = job_dir / "reconstruction" / "remote_out" / "streaming"
    transform = scene_frame_transform(job_dir)
    poses = np.loadtxt(streaming / "camera_poses.txt").reshape(-1, 4, 4)

    frames = list(load_streaming_frames(streaming))
    heights = {}
    for index, frame in enumerate(frames):
        if index % args.stride:
            continue
        height = camera_height(frame, transform, args.pixel_stride, args.depth_trunc)
        if height is not None:
            heights[index] = height

    # A plane fitted to a frame that barely sees the ground returns a height
    # near zero, and a ratio against it is arithmetic rather than measurement:
    # on the 17 Sep cloudy capture 13 of 16 pairs were of that kind and the
    # "drift" came out as 281x. The floor is a fraction of the capture's own
    # median, so it carries no assumption about scene scale.
    rejected = 0
    if heights:
        floor = args.min_height_fraction * float(np.median(list(heights.values())))
        kept = {i: h for i, h in heights.items() if h >= floor}
        rejected = len(heights) - len(kept)
        heights = kept
        print(f"[drift] ground-plane floor {floor:.5f} ({args.min_height_fraction:g} x median); "
              f"{rejected} frames rejected as failed fits", flush=True)
    if len(heights) < 2:
        print(json.dumps({"job": args.job, "frames_with_ground": len(heights),
                          "verified_revisits": 0, "note": "too few usable ground planes"}, indent=2))
        return 0

    images = {}
    for index in heights:
        with np.load(streaming / "results_output" / f"frame_{index}.npz") as data:
            images[index] = np.asarray(data["image"])
    print(f"[drift] {len(heights)} frames with a usable ground plane", flush=True)

    # Optical axis in the scene frame: +Z forward, OpenCV.
    axes = {i: (poses[i][:3, :3] @ np.array([0.0, 0.0, 1.0])) @ transform[:3, :3].T for i in heights}
    for i in axes:
        axes[i] = axes[i] / (np.linalg.norm(axes[i]) + 1e-12)

    sift = cv2.SIFT_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features = {}
    for i, image in images.items():
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features[i] = sift.detectAndCompute(grey, None)

    cos_limit = np.cos(np.radians(args.max_view_angle))
    order = sorted(heights)
    pairs = []
    started = time.perf_counter()
    for a_pos, i in enumerate(order):
        for j in order[a_pos + 1:]:
            if j - i < args.min_separation:
                continue
            if float(axes[i] @ axes[j]) < cos_limit:
                continue
            (ka, da), (kb, db) = features[i], features[j]
            if da is None or db is None or len(ka) < 8 or len(kb) < 8:
                continue
            good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
            if len(good) < args.min_inliers:
                continue
            pa = np.float32([ka[m.queryIdx].pt for m in good])
            pb = np.float32([kb[m.trainIdx].pt for m in good])
            _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
            inliers = int(mask.sum()) if mask is not None else 0
            if inliers < args.min_inliers:
                continue
            ratio = max(heights[i], heights[j]) / min(heights[i], heights[j])
            pairs.append({"a": i, "b": j, "inliers": inliers,
                          "height_a": round(heights[i], 5), "height_b": round(heights[j], 5),
                          "ratio": round(float(ratio), 4)})
    elapsed = time.perf_counter() - started

    report = {"job": args.job, "frames_with_ground": len(heights),
              "frames_rejected_as_failed_fits": rejected, "verified_revisits": len(pairs),
              "seconds": round(elapsed, 1)}
    if pairs:
        # Pairs cluster: a start-versus-end revisit alone can supply dozens of
        # them, which is one observation, not dozens. Count the clusters.
        events = {(p["a"] // 40, p["b"] // 40) for p in pairs}
        report["independent_revisit_events"] = len(events)
    if pairs:
        ratios = np.array([p["ratio"] for p in pairs])
        report["ratio_p50"] = round(float(np.median(ratios)), 4)
        report["ratio_p90"] = round(float(np.percentile(ratios, 90)), 4)
        report["ratio_max"] = round(float(ratios.max()), 4)
        # The whole-capture spread, for contrast: this one DOES include terrain.
        values = np.array(list(heights.values()))
        report["all_frames_spread_p90_over_p10"] = round(
            float(np.percentile(values, 90) / np.percentile(values, 10)), 4)
        report["worst"] = sorted(pairs, key=lambda p: -p["ratio"])[:10]
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps({**report, "pairs": pairs}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
