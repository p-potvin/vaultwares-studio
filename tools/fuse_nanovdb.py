"""Fuse a capture's depth maps into a ``.nvdb`` level set.

The whole path, locally, with no GPU and no OpenVDB:

    DA3-Streaming depth  ->  sparse TSDF  ->  NanoVDB float grid  ->  verify

Measured at the production voxel size on the 13 Sep backyard capture: ~13 s of
integration for 500 frames and ~1.2 M active voxels, so this is a stage that
runs on a workstation rather than a job that ties one up.

The write is gated on ``nanovdb_read.verify``, which walks every leaf's value
mask and checks the count against the tree's own bookkeeping. A grid with wrong
node offsets or masks still opens and still renders something, so "it produced
a file" is not evidence of anything.

    python tools/fuse_nanovdb.py \\
        --streaming data/jobs/<job>/reconstruction/remote_out/streaming \\
        --out data/jobs/<job>/reconstruction/volume.nvdb
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

from vaultwares_studio.nanovdb_read import verify  # noqa: E402
from vaultwares_studio.nanovdb_write import write_float_grid  # noqa: E402
from vaultwares_studio.tsdf_volume import integrate, load_streaming_frames  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming", required=True, type=Path,
                        help="DA3-Streaming output directory (holds results_output/)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--voxel", type=float, default=0.0156,
                        help="World units. The 13 Sep run's own choice.")
    parser.add_argument("--pixel-stride", type=int, default=2,
                        help="4.3x faster than 1 for 8%% fewer voxels at this voxel size")
    parser.add_argument("--depth-trunc", type=float, default=0.74)
    parser.add_argument("--conf-floor", type=float, default=0.0)
    parser.add_argument("--min-weight", type=float, default=2.0,
                        help="Drop voxels seen fewer times than this. 1 keeps every "
                             "voxel a single noisy pixel ever touched.")
    parser.add_argument("--name", default="surface")
    args = parser.parse_args(argv)

    if not (args.streaming / "results_output").is_dir():
        print(f"[nvdb] no results_output/ under {args.streaming}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    frames = list(load_streaming_frames(args.streaming, args.frames, args.frame_stride))
    load_s = time.perf_counter() - started
    if not frames:
        print("[nvdb] no frames", file=sys.stderr)
        return 1
    samples = sum(f["depth"].size for f in frames)
    print(f"[nvdb] {len(frames)} frames, {samples/1e6:.1f} M depth samples, "
          f"loaded in {load_s:.1f}s")

    started = time.perf_counter()
    volume = integrate(
        frames, voxel_size=args.voxel, depth_trunc=args.depth_trunc,
        conf_floor=args.conf_floor, pixel_stride=args.pixel_stride,
    )
    fuse_s = time.perf_counter() - started
    print(f"[nvdb] fused in {fuse_s:.1f}s -> {volume.active:,} touched voxels "
          f"(voxel {args.voxel}, band +/-{volume.sdf_trunc:.4f})")

    ijk, values = volume.to_arrays(min_weight=args.min_weight)
    if not len(ijk):
        print(f"[nvdb] nothing survived --min-weight {args.min_weight}", file=sys.stderr)
        return 1
    dropped = volume.active - len(ijk)
    print(f"[nvdb] {len(ijk):,} voxels kept, {dropped:,} dropped below weight "
          f"{args.min_weight} ({100*dropped/max(volume.active,1):.1f}%)")

    started = time.perf_counter()
    report = write_float_grid(
        args.out, ijk, values,
        voxel_size=args.voxel, name=args.name,
        grid_class=1,                 # LevelSet: values are metres from the surface
        background=float(volume.sdf_trunc),
    )
    write_s = time.perf_counter() - started
    size_mb = args.out.stat().st_size / 1e6
    print(f"[nvdb] wrote {args.out} in {write_s:.1f}s ({size_mb:.1f} MB) "
          f"{json.dumps(report.as_dict())}")

    # The gate. A file that fails here should never reach a renderer, because a
    # renderer will happily show it.
    checked = verify(args.out, expect={"name": args.name, "voxel_count": len(ijk)})
    entry = checked["grids"][0]
    print(f"[nvdb] verified: {entry['active_voxels_walked']:,} voxels walked, "
          f"value range {entry.get('value_range')}, bbox {entry['index_bbox']}")
    print(f"[nvdb] total {load_s + fuse_s + write_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
