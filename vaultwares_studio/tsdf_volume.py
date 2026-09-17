"""Fuse posed depth maps into a sparse TSDF, and keep the volume.

``depth_fusion`` already computes a truncated signed distance volume and then
throws it away to keep a mesh. The volume is the more useful artefact: it
carries free space as well as surface, it is what a VDB stores natively, and a
mesh can be extracted from it at any time while the reverse is not true.

**Distances are in world units, not normalised.** The benchmark that sized this
work stored ``signed / sdf_trunc`` clipped to [-1, 1], which is what a renderer
wants and what a level set does not. OpenVDB and NanoVDB level sets hold the
signed distance in world units inside a narrow band, so that is what is stored
here — the value at a voxel is metres from the surface, positive outside.
Getting this wrong produces a grid that still displays and whose isosurface sits
in the wrong place, scaled by the truncation distance.

**Camera convention: OpenCV, +Z forward.** Rays are built as ``(x, y, 1)``, so
``c2w`` must be the camera-to-world matrix in OpenCV axes — which is what
DA3-Streaming's ``camera_poses.txt`` holds and what ``depth_fusion`` feeds
Open3D. nerfstudio's ``transforms.json`` is OpenGL (+Z *backward*) and has to be
rebased first; ``hybrid_seed.OPENGL_TO_OPENCV`` is that matrix. Passing an
OpenGL pose here mirrors the scene through the camera and still produces a
tidy-looking volume.

A consequence worth stating because it is easy to get backwards: a surface one
metre ahead lands at world ``z = +1``, and voxels NEARER the camera have
*smaller* z and read *positive* (outside the surface).

**Sparse by construction.** Only voxels within the truncation band of some depth
sample ever exist. That is the same property that makes a VDB small, so no
conversion step is needed between this and the file: the active set here is the
active set there.

Measured on the 13 Sep backyard capture (500 frames, 70.6 M depth samples) at
the production voxel size of 0.0156: 12.8 s and 1.2 M active voxels at pixel
stride 2, against 55.3 s and 1.31 M at stride 1. The depth maps are 504x280 and
the grid is coarser than that sampling, so stride 2 is 4.3x faster for 8% fewer
voxels and is the sensible default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Voxel coordinates pack into one int64, 21 bits per axis, biased to unsigned.
# np.unique on a flat int64 array is a sort; on a structured dtype it is much
# slower, and a Python dict keyed by tuples is slower again by an order of
# magnitude — the first version of this spent all its time in a 130k-iteration
# loop per frame rather than in arithmetic.
PACK_BITS = 21
PACK_BIAS = 1 << 20


def pack_keys(ijk: np.ndarray) -> np.ndarray:
    k = np.asarray(ijk, dtype=np.int64) + PACK_BIAS
    if k.min() < 0 or k.max() >= (1 << PACK_BITS):
        raise ValueError(
            f"voxel index outside the {PACK_BITS}-bit packing range; the scene "
            f"spans more than {1 << PACK_BITS} voxels on some axis"
        )
    return (k[:, 0] << (2 * PACK_BITS)) | (k[:, 1] << PACK_BITS) | k[:, 2]


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    mask = (1 << PACK_BITS) - 1
    return np.stack([
        (keys >> (2 * PACK_BITS)) & mask,
        (keys >> PACK_BITS) & mask,
        keys & mask,
    ], axis=1).astype(np.int32) - PACK_BIAS


@dataclass
class TsdfVolume:
    """Accumulated signed distances, one entry per touched voxel."""

    keys: np.ndarray       # (N,) int64, packed voxel coordinates
    sums: np.ndarray       # (N,) float64, sum of signed distances (world units)
    weights: np.ndarray    # (N,) float64, number of samples that hit the voxel
    voxel_size: float
    sdf_trunc: float

    def to_arrays(self, min_weight: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """``(ijk, distance)`` for voxels seen at least ``min_weight`` times.

        Raising ``min_weight`` drops voxels that only one depth sample ever
        touched. Those sit at the ragged edge of the band where a single noisy
        pixel put them, and they are the ones that become speckle in a render
        and false obstacles in an occupancy query.
        """
        keep = self.weights >= min_weight
        values = (self.sums[keep] / self.weights[keep]).astype(np.float32)
        return unpack_keys(self.keys[keep]), values

    @property
    def active(self) -> int:
        return len(self.keys)


def integrate(
    frames,
    *,
    voxel_size: float,
    sdf_trunc: float | None = None,
    depth_trunc: float = 1e9,
    conf_floor: float = 0.0,
    pixel_stride: int = 2,
    flush_at: int = 8_000_000,
) -> TsdfVolume:
    """Fuse an iterable of frames into a sparse TSDF.

    Each frame needs ``depth`` (H, W) in world units, ``intrinsics`` (3, 3) for
    that depth map's resolution, ``c2w`` (4, 4) camera-to-world, and optionally
    ``conf``.

    Every depth sample writes a short band of voxels ALONG ITS OWN RAY rather
    than a single voxel. One voxel per sample leaves a shell full of holes
    wherever the sampling is sparser than the grid, and nothing downstream can
    put the surface back.
    """
    if sdf_trunc is None:
        sdf_trunc = voxel_size * 4.0
    band = max(1, int(round(sdf_trunc / voxel_size)))
    offsets = np.arange(-band, band + 1, dtype=np.float32)

    g_keys = np.zeros(0, dtype=np.int64)
    g_sums = np.zeros(0, dtype=np.float64)
    g_weights = np.zeros(0, dtype=np.float64)
    buf_keys: list[np.ndarray] = []
    buf_values: list[np.ndarray] = []
    buffered = 0

    def flush() -> None:
        nonlocal g_keys, g_sums, g_weights, buf_keys, buf_values, buffered
        if not buf_keys:
            return
        keys = np.concatenate([g_keys] + buf_keys)
        values = np.concatenate([g_sums] + buf_values)
        counts = np.concatenate([g_weights] + [np.ones(len(b)) for b in buf_values])
        unique, inverse = np.unique(keys, return_inverse=True)
        g_keys = unique
        g_sums = np.bincount(inverse, weights=values, minlength=len(unique))
        g_weights = np.bincount(inverse, weights=counts, minlength=len(unique))
        buf_keys, buf_values, buffered = [], [], 0

    for frame in frames:
        depth = np.asarray(frame["depth"])[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(depth) & (depth > 0) & (depth < depth_trunc)
        conf = frame.get("conf")
        if conf is not None and conf_floor > 0:
            valid &= np.asarray(conf)[::pixel_stride, ::pixel_stride] > conf_floor
        if not valid.any():
            continue

        rows, cols = np.nonzero(valid)
        z = depth[rows, cols].astype(np.float32)
        intr = np.asarray(frame["intrinsics"], dtype=np.float64)
        x = (cols * pixel_stride - intr[0, 2]) / intr[0, 0]
        y = (rows * pixel_stride - intr[1, 2]) / intr[1, 1]
        rays = np.stack([x, y, np.ones_like(x)], axis=1).astype(np.float32)
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        rays /= norms

        # Walk outward from the surface point along the ray. The signed distance
        # at a sample is minus its offset: in front of the surface (towards the
        # camera) is outside, and outside is positive.
        ray_length = (z * norms[:, 0])[:, None] + offsets[None, :] * voxel_size
        points = rays[:, None, :] * ray_length[:, :, None]
        c2w = np.asarray(frame["c2w"], dtype=np.float64)
        world = points.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]
        signed = np.tile(-offsets * voxel_size, (len(z), 1)).reshape(-1)

        buf_keys.append(pack_keys(np.floor(world / voxel_size).astype(np.int64)))
        # World units, clipped to the band — see the module docstring.
        buf_values.append(np.clip(signed, -sdf_trunc, sdf_trunc).astype(np.float64))
        buffered += len(buf_keys[-1])
        if buffered >= flush_at:
            flush()
    flush()

    return TsdfVolume(keys=g_keys, sums=g_sums, weights=g_weights,
                      voxel_size=voxel_size, sdf_trunc=sdf_trunc)


def load_streaming_frames(streaming: Path, limit: int | None = None, stride: int = 1):
    """Frames from a DA3-Streaming ``results_output`` directory.

    Streaming keeps a per-chunk SIM3 scale ``s``; a frame's depth is only in
    global units after multiplying by it, which is the same correction
    ``depth_fusion`` applies.
    """
    from .depth_fusion import _frame_files, _load_global_poses

    poses = _load_global_poses(streaming)
    files = _frame_files(streaming)
    if limit is not None:
        files = files[:limit]
    files = files[::stride]
    for index, path in enumerate(files):
        with np.load(path) as npz:
            scale = float(np.asarray(npz["s"])) if "s" in npz.files else 1.0
            yield {
                "depth": np.asarray(npz["depth"], dtype=np.float32) * np.float32(scale),
                "conf": (np.asarray(npz["conf"], dtype=np.float32)
                         if "conf" in npz.files else None),
                "intrinsics": np.asarray(npz["intrinsics"], dtype=np.float64),
                "c2w": poses[index * stride],
            }
