"""Build a dense seed cloud from COLMAP poses + DA3 depth, offline.

Takes a COLMAP ``processed_min.zip`` (or an unpacked directory) and a directory
of DA3 depth maps, and writes a new ``processed_min.zip`` whose transforms are
COLMAP's untouched and whose ``sparse_pc.ply`` is dense.

This runs on the CPU in a couple of minutes. No GPU job, no upload — which is
the point: the expensive half (COLMAP's matching, DA3's inference) is already
paid for and sitting in the artifact dataset, and every experiment with the
fusion parameters is free.

    python tools/build_hybrid_seed.py \\
        --colmap data/jobs/<job>/reconstruction/remote_out/processed_min.zip \\
        --depths data/jobs/<job>/depths \\
        --out    data/jobs/<job>/hybrid

The report it prints is the decision: ``scale_spread`` says whether DA3's depth
is consistent against COLMAP's geometry at all. Near zero and the two agree and
the fusion is sound. Wide and DA3's per-frame depth is drifting, which is worth
knowing before anything is trained on it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.hybrid_seed import build_hybrid_seed  # noqa: E402
from vaultwares_studio.splat_io import read_point_ply, write_point_ply  # noqa: E402


def unpack(source: Path, work: Path) -> Path:
    """Accept either the zip we ship between jobs or an unpacked directory."""
    if source.is_dir():
        return source
    target = work / "colmap"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(target)
    return target


def load_frames(transforms: dict, depths_dir: Path, colors_dir: Path | None) -> list[dict]:
    """Pair each posed frame with its depth map.

    DA3 writes depth named after the source frame's stem, so the join is by
    stem rather than by index — an index join silently misaligns the moment the
    two sides disagree about frame selection, and produces a plausible-looking
    cloud built from the wrong correspondences.
    """
    shared = {k: transforms[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h") if k in transforms}
    frames = []
    missing = []
    for entry in transforms["frames"]:
        stem = Path(entry["file_path"]).stem
        depth_path = depths_dir / f"{stem}.npy"
        if not depth_path.exists():
            missing.append(stem)
            continue
        depth = np.load(depth_path)
        height, width = depth.shape[:2]

        # The intrinsics in transforms.json are in the TRAINING resolution
        # (1920x1080). The depth map is in DA3's working resolution. Scale the
        # intrinsics down to the depth map rather than upsampling the depth:
        # resampling a depth map interpolates across object boundaries and
        # invents surfaces that are on neither side of the edge.
        fl_x = float(entry.get("fl_x", shared["fl_x"])) * width / shared["w"]
        fl_y = float(entry.get("fl_y", shared["fl_y"])) * height / shared["h"]
        cx = float(entry.get("cx", shared["cx"])) * width / shared["w"]
        cy = float(entry.get("cy", shared["cy"])) * height / shared["h"]

        frame = {
            "depth": depth,
            "intrinsics": np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]]),
            "c2w": np.array(entry["transform_matrix"], dtype=np.float64),
            "stem": stem,
        }
        conf_path = depths_dir.parent / "confidence" / f"{stem}.npy"
        if conf_path.exists():
            frame["confidence"] = np.load(conf_path)
        if colors_dir is not None:
            rgb_path = colors_dir / f"{stem}.npy"
            if rgb_path.exists():
                frame["colors"] = np.load(rgb_path)
        frames.append(frame)

    if missing:
        print(f"[hybrid] {len(missing)} posed frames have no depth map "
              f"(first: {', '.join(missing[:5])})")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap", required=True, type=Path,
                        help="COLMAP processed_min.zip or an unpacked directory")
    parser.add_argument("--depths", required=True, type=Path,
                        help="Directory of DA3 depth .npy maps, named by frame stem")
    parser.add_argument("--colors", type=Path,
                        help="Optional directory of matching RGB .npy arrays")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=2,
                        help="Pixel stride when back-projecting (2 = a quarter of the pixels)")
    parser.add_argument("--keep-quantile", type=float, default=0.6,
                        help="Confidence fraction retained per frame, matching da3_to_sparse_pc")
    parser.add_argument("--voxel", type=float, default=0.0,
                        help="Voxel size for dedup, in the scene's units. 0 disables. "
                             "500 overlapping depth maps are mostly duplicates, so this is "
                             "usually where the point count actually gets decided.")
    parser.add_argument("--max-points", type=int, default=2_000_000,
                        help="Hard cap after dedup, subsampled uniformly.")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="vw-hybrid-"))
    try:
        colmap_dir = unpack(args.colmap, work)
        transforms_path = colmap_dir / "transforms.json"
        if not transforms_path.exists():
            print(f"[hybrid] no transforms.json in {colmap_dir}", file=sys.stderr)
            return 1
        transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
        if "fl_x" not in transforms:
            print("[hybrid] this bundle has per-frame intrinsics, not a shared camera — "
                  "it is not a COLMAP bundle", file=sys.stderr)
            return 1

        sparse_path = colmap_dir / "sparse_pc.ply"
        sparse_points, _ = read_point_ply(sparse_path)
        print(f"[hybrid] COLMAP: {len(transforms['frames'])} posed frames, "
              f"{len(sparse_points):,} triangulated points")

        frames = load_frames(transforms, args.depths, args.colors)
        if not frames:
            print("[hybrid] no frame had both a pose and a depth map", file=sys.stderr)
            return 1
        print(f"[hybrid] fusing {len(frames)} frames at stride {args.stride}...")

        points, colors, report, alignments = build_hybrid_seed(
            frames,
            sparse_points,
            voxel=args.voxel,
            stride=args.stride,
            keep_quantile=args.keep_quantile,
        )

        if len(points) > args.max_points:
            keep = np.linspace(0, len(points) - 1, args.max_points).astype(int)
            points, colors = points[keep], colors[keep]
            print(f"[hybrid] capped to {args.max_points:,} points")

        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(transforms_path, args.out / "transforms.json")
        write_point_ply(points, colors, args.out / "sparse_pc.ply")

        detail = {
            "report": report.as_dict(),
            "frames": [
                {"stem": f["stem"], **a.as_dict()}
                for f, a in zip(frames, alignments)
            ],
        }
        (args.out / "hybrid_report.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")

        bundle = args.out / "processed_min.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(args.out / "transforms.json", "transforms.json")
            archive.write(args.out / "sparse_pc.ply", "sparse_pc.ply")

        print(json.dumps(report.as_dict(), indent=2))
        print(f"[hybrid] wrote {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
        if report.scale_spread > 0.25:
            print(f"[hybrid] WARNING: per-frame depth scale spreads "
                  f"{report.scale_spread * 100:.0f}% across the capture. DA3's depth is "
                  "not consistent against COLMAP's geometry; the fused cloud is only as "
                  "good as the worst frames in it.")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
