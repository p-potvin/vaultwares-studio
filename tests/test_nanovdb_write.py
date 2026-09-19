"""nanovdb_write: does a written grid contain what was put into it?

The payload is deliberately an **analytic function** — a sphere's signed
distance — rather than arbitrary numbers. Arbitrary values only prove the file
round-trips; a function proves the coordinate mapping is right in an absolute
sense, because a voxel written to the wrong place comes back holding a value
that no longer matches the function at the place it landed. Leaf indexing,
node origins and the root key are all wrong in ways that survive a naive
round trip and fail this.

The node sizes asserted here come from PNanoVDB's per-grid-type table, which is
an independent statement of the format from the C++ structs in NanoVDB.h.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaultwares_studio.nanovdb_read import NanoVDBError, read, verify
from vaultwares_studio.nanovdb_write import (
    LEAF_SIZE,
    LOWER_SIZE,
    ROOT_BASE_SIZE,
    ROOT_TILE_SIZE,
    UPPER_SIZE,
    write_float_grid,
)


def sphere(radius=12.0, half=20, band=3.0, offset=(0, 0, 0)):
    g = np.arange(-half, half + 1)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    ijk = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) + np.asarray(offset)
    centred = (ijk - np.asarray(offset)).astype(np.float64)
    distance = np.linalg.norm(centred, axis=1) - radius
    keep = np.abs(distance) <= band
    return ijk[keep].astype(np.int32), distance[keep].astype(np.float32)


def test_node_sizes_match_pnanovdb_float_table():
    """From PNanoVDB.h's grid-type constants, the Float row."""
    assert (LEAF_SIZE, LOWER_SIZE, UPPER_SIZE) == (2144, 33856, 270400)
    assert (ROOT_BASE_SIZE, ROOT_TILE_SIZE) == (64, 32)
    # Internal node size = table offset + 8 bytes per child slot.
    assert UPPER_SIZE == 8256 + (1 << 15) * 8
    assert LOWER_SIZE == 1088 + (1 << 12) * 8


def test_round_trip_is_exact(tmp_path):
    ijk, values = sphere()
    path = tmp_path / "sphere.nvdb"
    report = write_float_grid(path, ijk, values, voxel_size=0.25, name="surface")
    assert report.voxels == len(ijk)

    (grid,) = read(path)
    back_ijk, back_values = grid.active_voxels()
    a, b = np.lexsort(ijk.T), np.lexsort(back_ijk.T)
    assert np.array_equal(ijk[a], back_ijk[b])
    assert np.array_equal(values[a], back_values[b])


def test_values_still_match_the_function_they_came_from(tmp_path):
    """The check a naive round trip cannot make. If a voxel is written to the
    wrong coordinate, it returns holding a value that no longer agrees with the
    sphere evaluated where it actually landed."""
    ijk, values = sphere()
    path = tmp_path / "sphere.nvdb"
    write_float_grid(path, ijk, values)
    (grid,) = read(path)
    back_ijk, back_values = grid.active_voxels()
    analytic = np.linalg.norm(back_ijk.astype(np.float64), axis=1) - 12.0
    assert np.abs(analytic - back_values).max() < 1e-5


def test_negative_coordinates_survive(tmp_path):
    """RootData's key casts each signed coordinate to uint32 before shifting,
    so negative space does not sign-extend. Reproducing that cast is the whole
    subtlety; get it wrong and everything below zero lands in the wrong tile."""
    ijk, values = sphere(half=20)
    assert (ijk < 0).any()
    path = tmp_path / "neg.nvdb"
    write_float_grid(path, ijk, values)
    (grid,) = read(path)
    back_ijk, _ = grid.active_voxels()
    assert back_ijk.min() < 0
    assert set(map(tuple, back_ijk.tolist())) == set(map(tuple, ijk.tolist()))


def test_a_scene_spanning_several_upper_nodes(tmp_path):
    """An upper node covers 4096^3 voxels, so two clusters far apart force
    separate root tiles and exercise the root table's key ordering."""
    near, near_values = sphere(radius=6.0, half=9, band=2.0)
    far, far_values = sphere(radius=6.0, half=9, band=2.0, offset=(9000, 0, -9000))
    ijk = np.concatenate([near, far])
    values = np.concatenate([near_values, far_values])
    path = tmp_path / "wide.nvdb"
    report = write_float_grid(path, ijk, values)
    assert report.upper >= 2 and report.root_tiles == report.upper

    (grid,) = read(path)
    back_ijk, _ = grid.active_voxels()
    assert len(back_ijk) == len(ijk)
    assert set(map(tuple, back_ijk.tolist())) == set(map(tuple, ijk.tolist()))


def test_verify_accepts_what_we_wrote(tmp_path):
    ijk, values = sphere()
    path = tmp_path / "ok.nvdb"
    write_float_grid(path, ijk, values, voxel_size=0.5, name="tsdf", grid_class=1)
    report = verify(path, expect={"name": "tsdf", "voxel_count": len(ijk),
                                  "grid_class": 1, "voxel_size": (0.5, 0.5, 0.5)})
    entry = report["grids"][0]
    assert entry["active_voxels_walked"] == len(ijk)
    assert entry["grid_type"] == "Float"


def test_world_transform_is_recoverable(tmp_path):
    ijk, values = sphere(half=8, radius=5.0)
    path = tmp_path / "xform.nvdb"
    write_float_grid(path, ijk, values, voxel_size=0.25, translation=(1.0, 2.0, 3.0))
    (grid,) = read(path)
    assert grid.voxel_size == pytest.approx((0.25, 0.25, 0.25))
    assert grid.translation == pytest.approx((1.0, 2.0, 3.0))
    world = grid.index_to_world(np.array([[4, 8, 12]]))
    assert world[0] == pytest.approx([2.0, 4.0, 6.0])


def test_index_bbox_matches_the_voxels(tmp_path):
    ijk, values = sphere(half=10, radius=7.0)
    path = tmp_path / "bbox.nvdb"
    report = write_float_grid(path, ijk, values)
    assert report.index_bbox[:3] == tuple(ijk.min(axis=0).tolist())
    assert report.index_bbox[3:] == tuple(ijk.max(axis=0).tolist())
    verify(path)  # the bbox check inside verify would catch a mismatch


def test_duplicate_coordinates_are_refused(tmp_path):
    """A duplicate sets its mask bit once and writes twice, so the file would
    claim fewer active voxels than the caller handed over."""
    ijk = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="duplicate"):
        write_float_grid(tmp_path / "dup.nvdb", ijk, np.ones(3, np.float32))


def test_empty_input_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no active voxels"):
        write_float_grid(tmp_path / "empty.nvdb",
                         np.zeros((0, 3), np.int32), np.zeros(0, np.float32))


def test_mismatched_lengths_are_refused(tmp_path):
    with pytest.raises(ValueError, match="values"):
        write_float_grid(tmp_path / "bad.nvdb",
                         np.zeros((4, 3), np.int32), np.zeros(3, np.float32))


def test_a_single_voxel_is_a_valid_grid(tmp_path):
    """The degenerate case still needs one of every node level."""
    path = tmp_path / "one.nvdb"
    report = write_float_grid(path, np.array([[3, 4, 5]], np.int32),
                              np.array([0.5], np.float32))
    assert (report.leaves, report.lower, report.upper) == (1, 1, 1)
    (grid,) = read(path)
    ijk, values = grid.active_voxels()
    assert ijk.tolist() == [[3, 4, 5]] and values.tolist() == [0.5]
    verify(path, expect={"voxel_count": 1})


def test_grid_size_in_the_header_matches_the_bytes_written(tmp_path):
    ijk, values = sphere(half=6, radius=4.0)
    path = tmp_path / "size.nvdb"
    report = write_float_grid(path, ijk, values)
    # 16 B file header + 176 B metadata + name + grid.
    expected = 16 + 176 + len("density") + report.grid_bytes
    assert path.stat().st_size == expected
    read(path)  # the reader cross-checks mGridSize against the metadata


def test_coordinates_beyond_21_bits_are_refused(tmp_path):
    """The writer packs 21 bits per axis, and out of range the shifts collide.

    A colliding key makes two distinct voxels look like duplicates, so the
    duplicate check would fire on coordinates that are fine — or, worse, pass
    while the masks and the values disagree.
    """
    ijk = np.array([[0, 0, 0], [1 << 21, 0, 0]], dtype=np.int64)
    values = np.zeros(len(ijk), dtype=np.float32)
    with pytest.raises(ValueError, match="21-bit packing range"):
        write_float_grid(tmp_path / "out.nvdb", ijk, values, voxel_size=0.1)
