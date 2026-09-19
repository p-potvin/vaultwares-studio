"""Occupancy from the level set, with the free space carved from the rays.

    volume.nvdb  ->  grid_from_level_set  ->  ray_pass_counts  ->  carve_free_space

The level set gives the obstacles; it cannot give the free space, because a
TSDF stores nothing far from a surface. The rays can: every retained depth
sample is a segment the camera saw straight through, and the part of it inside
the body band is a body-height observation of empty space.

The check that matters is printed last: the fraction of camera positions that
land in FREE cells. The person walked there, so that number is the one thing
about free space that is known without the grid. On the 13 Sep backyard it is
what separates "filled UNKNOWN with something" from "filled it with the truth".

    python tools/occupancy_from_volume.py --job zerogpu-backyard134-loop-20260913
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
from vaultwares_studio.depth_fusion import _load_global_poses  # noqa: E402
from vaultwares_studio.robot_lab.occupancy import (  # noqa: E402
    FREE, OCCUPIED, UNKNOWN, OccupancyGrid,
    carve_free_space, cells_free_fraction, grid_from_depth_frames, grid_from_level_set,
    grid_from_preview_ply, ray_pass_counts,
)
from vaultwares_studio.tsdf_volume import load_streaming_frames  # noqa: E402

JOB_ROOTS = (ROOT / "data" / "jobs", Path("D:/vaultwares-studio-jobs/data/jobs"))


def _job_dir(job_id: str) -> Path:
    for base in JOB_ROOTS:
        if (base / job_id / "manifest.json").exists():
            return base / job_id
    raise SystemExit(f"no manifest.json for {job_id} under {[str(b) for b in JOB_ROOTS]}")


def _shares(grid: OccupancyGrid) -> dict:
    total = grid.cells.size
    return {
        "shape": list(grid.shape),
        "floor_y": round(grid.floor_y, 4),
        "free": round(100 * float((grid.cells == FREE).sum()) / total, 1),
        "occupied": round(100 * float((grid.cells == OCCUPIED).sum()) / total, 1),
        "unknown": round(100 * float((grid.cells == UNKNOWN).sum()) / total, 1),
    }


def _write_png(grid: OccupancyGrid, passes: np.ndarray, cameras_xz: np.ndarray, path: Path,
               scale: int = 8) -> None:
    """Free white, occupied black, unknown grey, camera path red. No axes,
    no dependency beyond PIL; it is a picture to look at, not a figure."""
    from PIL import Image

    palette = {FREE: (245, 245, 245), OCCUPIED: (20, 20, 20), UNKNOWN: (140, 140, 140)}
    rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
    for code, colour in palette.items():
        rgb[grid.cells == code] = colour
    image = Image.fromarray(rgb).resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    pixels = image.load()
    for x, z in cameras_xz:
        row, col = grid.world_to_cell(float(x), float(z))
        if grid.in_bounds(row, col):
            for dr in range(scale // 2 - 1, scale // 2 + 2):
                for dc in range(scale // 2 - 1, scale // 2 + 2):
                    pixels[col * scale + dc, row * scale + dr] = (220, 30, 30)
    image.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--camera-height-m", type=float, default=1.4,
                        help="how high the phone was held. The scene is not metric; the "
                             "median camera height above the estimated floor is the ruler.")
    parser.add_argument("--cell-m", type=float, default=0.25)
    parser.add_argument("--body-band-m", type=float, nargs=2, default=(0.10, 0.60),
                        help="metres above the floor the robot body sweeps")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-votes", type=int, default=2,
                        help="frames that must agree on a cell (frame-local grid)")
    parser.add_argument("--min-passes", type=int, default=3)
    parser.add_argument("--obstacle-fraction", type=float, default=0.5,
                        help="share of the frames that saw a cell which must call it an obstacle")
    parser.add_argument("--margin-m", type=float, default=0.2,
                        help="metres held back before the surface when carving (frame-local grid)")
    parser.add_argument("--conf-floor", type=float, default=0.0,
                        help="drop depth samples with DA3 confidence at or below this")
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--depth-trunc", type=float, default=0.74,
                        help="DA3 units, the same cut fuse_nanovdb applies")
    parser.add_argument("--margin", type=float, default=None,
                        help="scene units held back before the surface; default is the "
                             "TSDF band (4 voxels of 0.0156) taken through the scene scale")
    parser.add_argument("--no-carve", action="store_true")
    parser.add_argument("--out", type=Path, default=None,
                        help="default <job>/reconstruction/occupancy.npz (+ .png)")
    args = parser.parse_args(argv)

    job_dir = _job_dir(args.job)
    recon = job_dir / "reconstruction"
    nvdb = recon / "volume.nvdb"
    streaming = recon / "remote_out" / "streaming"
    if not nvdb.exists():
        print(f"[occ] no {nvdb}; run tools/fuse_nanovdb.py first", file=sys.stderr)
        return 1
    transform = scene_frame_transform(job_dir)
    scale = float(np.linalg.norm(transform[:3, 0]))
    margin = args.margin if args.margin is not None else 4 * 0.0156 * scale

    poses = _load_global_poses(streaming)[::args.frame_stride]
    centres = poses[:, :3, 3] @ transform[:3, :3].T + transform[:3, 3]
    cameras_xz = centres[:, [0, 2]]

    # The ruler. Nothing in a DA3 scene is metric, and a body band stated in
    # scene units is a guess dressed as a number: on the 13 Sep backyard the
    # cameras sit 0.165 units above the floor, so (0.10, 0.60) was a band from
    # 0.85 m to 5 m up and swept in every wall and every tree. The floor is
    # estimated first with a band that cannot matter, only to measure this.
    ruler = grid_from_level_set(nvdb, scene_transform=transform, cell_size=0.05,
                                body_band=(0.0, 1e9), min_support=1)
    camera_height = float(np.median(centres[:, 1] - ruler.floor_y))
    units_per_m = camera_height / args.camera_height_m
    cell = args.cell_m * units_per_m
    band = (args.body_band_m[0] * units_per_m, args.body_band_m[1] * units_per_m)
    print(f"[occ] median camera {camera_height:.4f} units above the floor; at "
          f"{args.camera_height_m} m that is {1 / units_per_m:.2f} m per unit -> cell {cell:.4f}, "
          f"band ({band[0]:.4f}, {band[1]:.4f})")

    started = time.perf_counter()
    base = grid_from_level_set(nvdb, scene_transform=transform, cell_size=cell,
                               body_band=band, min_support=args.min_support)
    print(f"[occ] level set -> grid {base.shape} in {time.perf_counter() - started:.1f}s: "
          f"{json.dumps(_shares(base))}")

    report = {"job": args.job, "scene_scale": round(scale, 4), "margin": round(margin, 4),
              "metres_per_unit": round(1 / units_per_m, 3), "cell_m": args.cell_m,
              "body_band_m": list(args.body_band_m), "cameras": int(len(cameras_xz)),
              "level_set": {**_shares(base), "cameras_in_free": round(cells_free_fraction(base, cameras_xz), 3)}}

    preview = recon / "cloud_preview.ply"
    if preview.exists():
        splat = grid_from_preview_ply(preview, cell_size=cell, body_band=band,
                                      min_support=args.min_support)
        report["splat_preview"] = {**_shares(splat),
                                   "cameras_in_free": round(cells_free_fraction(splat, cameras_xz), 3)}

    grid, passes = base, np.zeros(base.shape, dtype=np.int64)
    if not args.no_carve:
        started = time.perf_counter()
        frames = load_streaming_frames(streaming, stride=args.frame_stride)
        passes = ray_pass_counts(base, frames, body_band=band, scene_transform=transform,
                                 pixel_stride=args.pixel_stride, depth_trunc=args.depth_trunc,
                                 margin=margin)
        carve_s = time.perf_counter() - started
        grid = carve_free_space(base, passes, min_passes=args.min_passes)
        report["carved"] = {**_shares(grid),
                            "cameras_in_free": round(cells_free_fraction(grid, cameras_xz), 3),
                            "seconds": round(carve_s, 1),
                            "cells_with_passes": int((passes > 0).sum()),
                            "promoted": int(((base.cells == UNKNOWN) & (grid.cells == FREE)).sum())}
        print(f"[occ] carved {report['carved']['promoted']} cells in {carve_s:.1f}s: "
              f"{json.dumps(_shares(grid))}")

    out = args.out or (recon / "occupancy.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    level_set_out = out.with_name(out.stem + "_levelset.npz")
    grid.save(level_set_out)
    _write_png(grid, passes, cameras_xz, level_set_out.with_suffix(".png"))

    # The frame-local grid: each frame measured against the ground it saw,
    # with its own camera-to-ground distance as the ruler. Same extent and
    # cells as the level-set grid, so the two are comparable cell for cell.
    started = time.perf_counter()
    local, local_report = grid_from_depth_frames(
        base, load_streaming_frames(streaming, stride=args.frame_stride),
        scene_transform=transform, camera_height_m=args.camera_height_m,
        body_band_m=tuple(args.body_band_m), margin_m=args.margin_m,
        pixel_stride=args.pixel_stride, depth_trunc=args.depth_trunc,
        conf_floor=args.conf_floor, min_votes=args.min_votes, min_passes=args.min_passes,
        obstacle_fraction=args.obstacle_fraction,
    )
    report["frame_local"] = {**_shares(local),
                             "cameras_in_free": round(cells_free_fraction(local, cameras_xz), 3),
                             "seconds": round(time.perf_counter() - started, 1),
                             "frames_used": local_report["frames_used"],
                             "frames_skipped": local_report["frames_skipped"],
                             "metres_per_unit_p10_50_90": local_report["metres_per_unit_p10_50_90"]}
    print(f"[occ] frame-local grid in {report['frame_local']['seconds']}s: "
          f"{json.dumps(_shares(local))} cameras_in_free {report['frame_local']['cameras_in_free']}")
    local.save(out)
    np.savez_compressed(out.with_name(out.stem + "_votes.npz"),
                        obstacle=local_report["obstacle_votes"], support=local_report["support_votes"],
                        passes=local_report["passes"])
    _write_png(local, local_report["passes"], cameras_xz, out.with_suffix(".png"))
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
