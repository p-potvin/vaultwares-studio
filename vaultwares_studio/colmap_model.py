"""Read COLMAP's binary sparse model, for the one thing the PLY cannot tell us.

``sparse_pc.ply`` has the triangulated points and nothing else. What it loses is
**visibility** — which frames actually saw each point.

That loss is not cosmetic. The hybrid seed aligns a predicted depth map against
COLMAP's points by pairing (predicted, true) at each point's pixel, and without
visibility the only option is to project every point into every frame and hope a
robust estimator rejects the ones hiding behind walls. Measured on the June 14
backyard bundle, that gives the median frame **45,966** projected points out of
185,355 — where COLMAP's own tracks say it saw a few hundred. The occluded
fraction is not a tail an M-estimator can absorb; it is the overwhelming
majority, and a Huber loss whose threshold comes from the median residual fits
the occluders instead of the surface.

The symptom was unmistakable once measured: correlation between DA3's depth and
COLMAP's "true" depth ranged +0.60 to -0.10 across frames, and the per-frame
scale spread came out at 630%. Neither was telling us anything about DA3.

``points3D.bin`` carries a track per point — the (image_id, point2D_idx) pairs
its matcher verified — and ``images.bin`` maps image_id to filename. Together
they give exactly the points each frame saw, which is the correspondence set the
alignment wanted all along.

Format reference: COLMAP's ``src/colmap/scene/reconstruction.cc``. Both files are
little-endian and dense; there is no index, so reading one means walking it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _read(stream, fmt: str):
    size = struct.calcsize(fmt)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"truncated COLMAP model: wanted {size} bytes, got {len(data)}")
    return struct.unpack(fmt, data)


@dataclass(frozen=True)
class SparseModel:
    """COLMAP's points, plus which image saw which."""

    point_ids: np.ndarray  # (P,) int64, COLMAP's own ids
    xyz: np.ndarray  # (P, 3) float64, in COLMAP's world
    rgb: np.ndarray  # (P, 3) uint8
    # image filename -> indices into xyz. The filename, not COLMAP's image_id,
    # because transforms.json identifies frames by path and that is what every
    # caller already has.
    visible: dict[str, np.ndarray]

    def points_seen_by(self, name: str) -> np.ndarray:
        """The points COLMAP verified in this frame. Empty if it saw none."""
        index = self.visible.get(name)
        if index is None:
            # transforms.json writes "images/frame_00001.jpg"; images.bin stores
            # "frame_00001.jpg". Try the basename before giving up.
            index = self.visible.get(Path(name).name)
        if index is None:
            return np.zeros((0, 3), dtype=np.float64)
        return self.xyz[index]


def read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """``(point_ids, xyz, rgb, tracks)`` where each track is an array of image ids."""
    ids, xyz, rgb, tracks = [], [], [], []
    with open(path, "rb") as stream:
        (count,) = _read(stream, "<Q")
        for _ in range(count):
            point_id, x, y, z, r, g, b, _error = _read(stream, "<QdddBBBd")
            (track_length,) = _read(stream, "<Q")
            # Each element is (image_id, point2D_idx); only the image id matters
            # here, but both must be consumed to stay aligned in the stream.
            raw = stream.read(8 * track_length)
            if len(raw) != 8 * track_length:
                raise ValueError("truncated COLMAP track")
            image_ids = np.frombuffer(raw, dtype="<u4")[0::2].astype(np.int64)
            ids.append(point_id)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
            tracks.append(image_ids)
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
        tracks,
    )


def read_image_names(path: Path) -> dict[int, str]:
    """``image_id -> filename`` from images.bin.

    The per-image 2D observations are skipped rather than parsed: this walks the
    file only to reach each name, and the tracks in points3D.bin already carry
    the correspondence the caller wants.
    """
    names: dict[int, str] = {}
    with open(path, "rb") as stream:
        (count,) = _read(stream, "<Q")
        for _ in range(count):
            image_id, *_qvec_tvec, _camera_id = _read(stream, "<idddddddi")
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char or char == b"\x00":
                    break
                name += char
            (num_points2d,) = _read(stream, "<Q")
            # Each observation is (double x, double y, uint64 point3D_id).
            stream.seek(24 * num_points2d, 1)
            names[int(image_id)] = name.decode("utf-8")
    return names


def read_sparse_model(sparse_dir: Path) -> SparseModel:
    """Load points + visibility from a COLMAP ``sparse/`` directory."""
    sparse_dir = Path(sparse_dir)
    points_path = sparse_dir / "points3D.bin"
    images_path = sparse_dir / "images.bin"
    for path in (points_path, images_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — this is not a COLMAP binary sparse model. "
                "Only COLMAP bundles carry visibility; a DA3 bundle has none."
            )

    point_ids, xyz, rgb, tracks = read_points3d(points_path)
    names = read_image_names(images_path)

    # Invert the tracks: point -> images becomes image -> points.
    per_image: dict[str, list[int]] = {name: [] for name in names.values()}
    for index, image_ids in enumerate(tracks):
        for image_id in image_ids:
            name = names.get(int(image_id))
            if name is not None:
                per_image[name].append(index)

    visible = {
        name: np.asarray(sorted(indices), dtype=np.int64)
        for name, indices in per_image.items()
    }
    return SparseModel(point_ids=point_ids, xyz=xyz, rgb=rgb, visible=visible)
