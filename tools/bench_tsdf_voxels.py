"""How much does a sparse TSDF actually cost on this machine?

Before committing to a local pipeline stage, measure it. The reference point is
COLMAP: 43 minutes of CPU for 500 images, which is long enough that nobody wants
it on a workstation. The question is whether voxelising the same capture is in
that class or nowhere near it.

It should be nowhere near it, and for a reason worth stating: COLMAP's cost is
feature *matching*, which is quadratic-ish in image count and runs a descriptor
search per pair. TSDF integration is linear in pixels — every depth sample is
touched once, independently. Same input, completely different shape of work.

What this measures, per configuration:

* wall time, split into reading the npz files and integrating
* **active voxel count**, which is what decides the ``.nvdb`` size and is the
  number that matters for anything downstream
* peak resident memory

Two integrators are timed. Open3D's ``ScalableTSDFVolume`` is the incumbent
(``depth_fusion.py`` uses it and the 13 Sep run reported 6.2 s for 500 frames),
but it will not hand back its voxels — only ``extract_voxel_point_cloud``, which
packs the TSDF into a colour channel. The numpy integrator exists because the
voxels ARE the deliverable here; a mesh is a lossy summary of them.

Run it before deciding anything:

    python tools/bench_tsdf_voxels.py --streaming <.../streaming> --frames 500
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _peak_memory_mb() -> float:
    """Peak working set, or 0 where the platform will not say."""
    try:
        import psutil

        return psutil.Process().memory_info().peak_wset / 1e6  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a missing metric must not fail a benchmark
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:  # noqa: BLE001
            return 0.0


@dataclass
class Frame:
    depth: np.ndarray
    conf: np.ndarray
    intrinsics: np.ndarray
    c2w: np.ndarray


def load_frames(streaming: Path, limit: int, stride: int = 1) -> tuple[list[Frame], float]:
    """Read the retained per-frame fields, already placed in the global map.

    Mirrors ``depth_fusion``: streaming keeps a per-chunk SIM3 scale ``s``, so a
    frame's depth is only in global units after multiplying by it.
    """
    from vaultwares_studio.depth_fusion import _frame_files, _load_global_poses

    started = time.perf_counter()
    poses = _load_global_poses(streaming)
    files = _frame_files(streaming)[:limit:stride]
    frames = []
    for index, path in enumerate(files):
        with np.load(path) as npz:
            scale = float(np.asarray(npz["s"])) if "s" in npz.files else 1.0
            frames.append(
                Frame(
                    depth=np.asarray(npz["depth"], dtype=np.float32) * np.float32(scale),
                    conf=(np.asarray(npz["conf"], dtype=np.float32)
                          if "conf" in npz.files else None),
                    intrinsics=np.asarray(npz["intrinsics"], dtype=np.float64),
                    c2w=poses[index * stride],
                )
            )
    return frames, time.perf_counter() - started


def integrate_numpy(
    frames: list[Frame],
    voxel_size: float,
    sdf_trunc: float,
    *,
    depth_trunc: float,
    conf_floor: float,
    pixel_stride: int = 1,
) -> tuple[dict, float, int]:
    """Sparse TSDF by hashing integer voxel coordinates.

    No volume is allocated: only voxels a depth sample actually falls near ever
    exist, which is the same property that makes a VDB small and is why this is
    the natural shape for the output.

    Each surface sample writes a short band of voxels ALONG ITS OWN RAY rather
    than a single voxel. A single voxel per sample leaves a shell riddled with
    holes wherever the sampling is sparser than the grid, and no amount of
    smoothing afterwards puts a surface back.
    """
    started = time.perf_counter()
    band = max(1, int(round(sdf_trunc / voxel_size)))
    offsets = np.arange(-band, band + 1, dtype=np.float32)

    # Voxel coordinates packed into one int64: 21 bits per axis, biased to
    # unsigned. np.unique on a flat int64 array is a sort; on a 3-field
    # structured dtype it is far slower, and a Python dict keyed by tuples is
    # slower again by an order of magnitude — that version spent all its time in
    # a 130k-iteration loop per frame rather than in any actual arithmetic.
    BITS, BIAS = 21, 1 << 20

    def pack(keys: np.ndarray) -> np.ndarray:
        k = keys.astype(np.int64) + BIAS
        if k.min() < 0 or k.max() >= (1 << BITS):
            raise ValueError(
                f"voxel index out of the {BITS}-bit packing range at voxel_size="
                f"{voxel_size}; the scene spans more than {(1 << BITS) * voxel_size:.0f} units"
            )
        return (k[:, 0] << (2 * BITS)) | (k[:, 1] << BITS) | k[:, 2]

    g_keys = np.zeros(0, dtype=np.int64)
    g_sums = np.zeros(0, dtype=np.float64)
    g_wts = np.zeros(0, dtype=np.float64)
    buf_keys: list[np.ndarray] = []
    buf_vals: list[np.ndarray] = []
    buffered = 0
    FLUSH_AT = 8_000_000  # entries; keeps the merge arrays around 100 MB

    def flush():
        nonlocal g_keys, g_sums, g_wts, buf_keys, buf_vals, buffered
        if not buf_keys:
            return
        keys = np.concatenate([g_keys] + buf_keys)
        vals = np.concatenate([g_sums] + buf_vals)
        wts = np.concatenate([g_wts] + [np.ones(len(b)) for b in buf_vals])
        uniq, inverse = np.unique(keys, return_inverse=True)
        g_keys = uniq
        g_sums = np.bincount(inverse, weights=vals, minlength=len(uniq))
        g_wts = np.bincount(inverse, weights=wts, minlength=len(uniq))
        buf_keys, buf_vals, buffered = [], [], 0

    for frame in frames:
        depth = frame.depth[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(depth) & (depth > 0) & (depth < depth_trunc)
        if frame.conf is not None and conf_floor > 0:
            valid &= frame.conf[::pixel_stride, ::pixel_stride] > conf_floor
        if not valid.any():
            continue

        rows, cols = np.nonzero(valid)
        z = depth[rows, cols].astype(np.float32)
        fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
        cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]
        x = (cols * pixel_stride - cx) / fx
        y = (rows * pixel_stride - cy) / fy
        rays = np.stack([x, y, np.ones_like(x)], axis=1).astype(np.float32)
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        rays /= norms

        ray_len = (z * norms[:, 0])[:, None] + offsets[None, :] * voxel_size
        points = rays[:, None, :] * ray_len[:, :, None]
        world = points.reshape(-1, 3) @ frame.c2w[:3, :3].T + frame.c2w[:3, 3]
        signed = np.tile(-offsets * voxel_size, (len(z), 1)).reshape(-1)

        packed = pack(np.floor(world / voxel_size).astype(np.int64))
        buf_keys.append(packed)
        buf_vals.append(np.clip(signed / sdf_trunc, -1.0, 1.0).astype(np.float64))
        buffered += len(packed)
        if buffered >= FLUSH_AT:
            flush()
    flush()

    elapsed = time.perf_counter() - started
    return {"keys": g_keys, "sums": g_sums, "weights": g_wts}, elapsed, len(g_keys)


def integrate_open3d(frames: list[Frame], voxel_size: float, sdf_trunc: float,
                     depth_trunc: float) -> tuple[int, float]:
    import open3d as o3d

    started = time.perf_counter()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    for frame in frames:
        height, width = frame.depth.shape
        intr = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            frame.intrinsics[0, 0], frame.intrinsics[1, 1],
            frame.intrinsics[0, 2], frame.intrinsics[1, 2],
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((height, width, 3), np.uint8)),
            o3d.geometry.Image(frame.depth),
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intr, np.linalg.inv(frame.c2w))
    elapsed = time.perf_counter() - started
    voxels = len(volume.extract_voxel_point_cloud().points)
    return voxels, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--voxel", type=float, action="append",
                        help="Voxel size(s) to test. Default: the 13 Sep run's 0.0156, "
                             "plus half and double it.")
    parser.add_argument("--pixel-stride", type=int, action="append")
    parser.add_argument("--depth-trunc", type=float, default=0.74)
    parser.add_argument("--conf-floor", type=float, default=0.0)
    parser.add_argument("--skip-open3d", action="store_true")
    args = parser.parse_args()

    voxels = args.voxel or [0.0312, 0.0156, 0.0078]
    strides = args.pixel_stride or [1, 2]

    frames, load_s = load_frames(args.streaming, args.frames, args.frame_stride)
    pixels = sum(f.depth.size for f in frames)
    print(f"  loaded {len(frames)} frames in {load_s:.1f}s "
          f"({pixels/1e6:.1f} M depth samples, {pixels/max(load_s,1e-9)/1e6:.1f} M/s)")
    print(f"  reference: COLMAP SfM on this capture took 43 min on a cloud CPU\n")

    print(f"  {'voxel':>8} {'px stride':>9} {'integrate':>10} {'active voxels':>14} {'peak MB':>9}")

    for voxel in voxels:
        for stride in strides:
            grid, elapsed, _ = integrate_numpy(
                frames, voxel, voxel * 4.0,
                depth_trunc=args.depth_trunc, conf_floor=args.conf_floor,
                pixel_stride=stride,
            )
            active = len(grid["keys"])
            # ~28 bytes/voxel in NanoVDB's leaf payload for a float grid, before
            # its tree overhead. Rough, but it answers "will this fit".
            print(f"  {voxel:8.4f} {stride:9} {elapsed:9.1f}s {active:14,} "
                  f"{_peak_memory_mb():8.0f}   ~{active * 28 / 1e6:.0f} MB nvdb")

    if not args.skip_open3d:
        for voxel in voxels:
            count, elapsed = integrate_open3d(frames, voxel, voxel * 4.0, args.depth_trunc)
            print(f"  open3d   {voxel:.4f}          {elapsed:9.1f}s {count:14,} {_peak_memory_mb():8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
