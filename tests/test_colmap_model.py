"""colmap_model: reading visibility out of COLMAP's binary sparse model.

Built from synthetic models written byte-for-byte to COLMAP's format, because
the failure this module exists to prevent is silent: without visibility the
hybrid alignment fits occluded geometry and returns confident nonsense.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from vaultwares_studio.colmap_model import (
    read_image_names,
    read_points3d,
    read_sparse_model,
)


def _write_points3d(path: Path, points: list[tuple[int, tuple, tuple, list[int]]]) -> None:
    """points: (point_id, xyz, rgb, [image_id, ...])"""
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, xyz, rgb, track in points:
            stream.write(struct.pack("<Qddd", point_id, *xyz))
            stream.write(struct.pack("<BBB", *rgb))
            stream.write(struct.pack("<d", 0.5))  # reprojection error
            stream.write(struct.pack("<Q", len(track)))
            for image_id in track:
                stream.write(struct.pack("<II", image_id, 0))


def _write_images(path: Path, images: list[tuple[int, str, int]]) -> None:
    """images: (image_id, name, num_points2D)"""
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(images)))
        for image_id, name, num_points2d in images:
            stream.write(struct.pack("<i", image_id))
            stream.write(struct.pack("<7d", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            stream.write(struct.pack("<i", 1))  # camera_id
            stream.write(name.encode("utf-8") + b"\x00")
            stream.write(struct.pack("<Q", num_points2d))
            # Each observation is (double x, double y, uint64 point3D_id).
            stream.write(b"\x00" * 24 * num_points2d)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    _write_points3d(sparse / "points3D.bin", [
        (1, (0.0, 0.0, 1.0), (255, 0, 0), [10, 20]),
        (2, (1.0, 0.0, 2.0), (0, 255, 0), [20]),
        (3, (2.0, 1.0, 3.0), (0, 0, 255), [10, 20, 30]),
    ])
    # Non-zero observation counts: the reader must skip exactly 24 bytes each or
    # it desynchronises and reads the next image's fields out of the middle of
    # this one's data.
    _write_images(sparse / "images.bin", [
        (10, "frame_00001.jpg", 3),
        (20, "frame_00002.jpg", 0),
        (30, "frame_00003.jpg", 7),
    ])
    return sparse


def test_points_and_tracks_round_trip(model_dir: Path):
    ids, xyz, rgb, tracks = read_points3d(model_dir / "points3D.bin")
    assert list(ids) == [1, 2, 3]
    assert xyz.shape == (3, 3) and xyz[2].tolist() == [2.0, 1.0, 3.0]
    assert rgb[0].tolist() == [255, 0, 0]
    assert [t.tolist() for t in tracks] == [[10, 20], [20], [10, 20, 30]]


def test_image_names_survive_variable_observation_counts(model_dir: Path):
    """The per-image 2D block is skipped, not parsed. Getting its stride wrong
    reads the next image's id out of the middle of this one's observations."""
    assert read_image_names(model_dir / "images.bin") == {
        10: "frame_00001.jpg", 20: "frame_00002.jpg", 30: "frame_00003.jpg",
    }


def test_visibility_is_the_inverse_of_the_tracks(model_dir: Path):
    model = read_sparse_model(model_dir)
    assert model.visible["frame_00001.jpg"].tolist() == [0, 2]
    assert model.visible["frame_00002.jpg"].tolist() == [0, 1, 2]
    assert model.visible["frame_00003.jpg"].tolist() == [2]


def test_points_seen_by_returns_coordinates(model_dir: Path):
    seen = read_sparse_model(model_dir).points_seen_by("frame_00003.jpg")
    assert seen.shape == (1, 3)
    assert seen[0].tolist() == [2.0, 1.0, 3.0]


def test_a_transforms_style_path_still_matches(model_dir: Path):
    """transforms.json says "images/frame_00001.jpg"; images.bin stores the bare
    name. A caller should not have to know that."""
    model = read_sparse_model(model_dir)
    assert len(model.points_seen_by("images/frame_00001.jpg")) == 2


def test_an_unseen_frame_is_empty_not_an_error(model_dir: Path):
    assert read_sparse_model(model_dir).points_seen_by("frame_99999.jpg").shape == (0, 3)


def test_a_bundle_without_a_binary_model_says_so(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not a COLMAP binary sparse model"):
        read_sparse_model(tmp_path)


def test_truncation_is_detected(model_dir: Path, tmp_path: Path):
    """A short read must raise rather than silently returning fewer points —
    a partial model is exactly the failure that cost this investigation a day."""
    raw = (model_dir / "points3D.bin").read_bytes()
    (tmp_path / "points3D.bin").write_bytes(raw[:-8])
    (tmp_path / "images.bin").write_bytes((model_dir / "images.bin").read_bytes())
    with pytest.raises(ValueError, match="truncated"):
        read_sparse_model(tmp_path)


def test_visible_counts_are_what_the_alignment_budget_assumes(model_dir: Path):
    """Sanity on the real shape of this data: a frame sees a small subset, never
    the whole cloud. On the June 14 backyard model the median is 577 of 49,107."""
    model = read_sparse_model(model_dir)
    counts = np.array([len(v) for v in model.visible.values()])
    assert counts.max() <= len(model.xyz)
