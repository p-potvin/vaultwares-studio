"""nanovdb_read: the verifier, tested against bytes assembled from the spec.

The fixture here is built by hand with ``struct.pack`` straight from the field
layout in ``nanovdb/NanoVDB.h``. It deliberately does NOT use the writer: a
reader tested against its own writer agrees with itself and proves nothing
about the format. When the writer lands, its round trip becomes an additional
check, not this one's replacement.

What these tests can and cannot establish, stated plainly: they show the reader
decodes the layout it claims to, and that it refuses files that are internally
inconsistent. They cannot show that layout is what OpenVDB itself writes — only
a real ``.nvdb``, or Omniverse opening ours, settles that. There is no ``.vdb``
anywhere on this machine to check against.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from vaultwares_studio.nanovdb_read import (
    FILE_HEADER_SIZE,
    FILE_META_SIZE,
    GRID_DATA_SIZE,
    LEAF_VOXELS,
    MAGIC_FILE,
    MAGIC_GRID,
    MAGIC_NUMB,
    TREE_DATA_SIZE,
    NanoVDBError,
    decode_version,
    read,
    verify,
)

LEAF_STRIDE = 12 + 3 + 1 + 64 + 16 + LEAF_VOXELS * 4  # 2144
VERSION = (32 << 21) | (7 << 10) | 0


def _leaf(origin, active: dict[int, float]) -> bytes:
    """One LeafData<float,3>: origin, masks, stats, 512 values."""
    mask = bytearray(64)
    values = np.zeros(LEAF_VOXELS, dtype="<f4")
    for index, value in active.items():
        mask[index // 8] |= 1 << (index % 8)
        values[index] = value
    present = np.array(list(active.values()), dtype=np.float32)
    return b"".join([
        struct.pack("<3i", *origin),
        bytes(3),                      # mBBoxDif
        bytes([2]),                    # mFlags: bit1 = has bbox
        bytes(mask),                   # mValueMask
        struct.pack("<ffff", float(present.min()), float(present.max()),
                    float(present.mean()), 0.0),
        values.tobytes(),
    ])


def _grid_buffer(leaves: list[bytes], voxel_size=(0.5, 0.5, 0.5),
                 translation=(1.0, 2.0, 3.0), name="density",
                 grid_class=1, grid_type=1, voxel_count=None) -> bytes:
    """GridData + TreeData + leaves, with the leaves immediately after the tree."""
    body = b"".join(leaves)
    grid_size = GRID_DATA_SIZE + TREE_DATA_SIZE + len(body)

    grid = bytearray(GRID_DATA_SIZE)
    struct.pack_into("<Q", grid, 0, MAGIC_GRID)
    struct.pack_into("<I", grid, 16, VERSION)
    struct.pack_into("<Q", grid, 32, grid_size)
    grid[40:40 + len(name)] = name.encode()
    # Map: the double translation sits at 296 + 36 + 36 + 12 + 4 + 72 + 72.
    struct.pack_into("<3d", grid, 296 + 36 + 36 + 12 + 4 + 72 + 72, *translation)
    struct.pack_into("<3d", grid, 608, *voxel_size)
    struct.pack_into("<II", grid, 632, grid_class, grid_type)

    if voxel_count is None:
        voxel_count = sum(
            int(np.unpackbits(np.frombuffer(leaf[16:80], dtype=np.uint8),
                              bitorder="little").sum())
            for leaf in leaves
        )
    tree = struct.pack(
        "<4q3I3IQ",
        TREE_DATA_SIZE, TREE_DATA_SIZE, TREE_DATA_SIZE, TREE_DATA_SIZE,
        len(leaves), 0, 0,
        0, 0, 0,
        voxel_count,
    )
    return bytes(grid) + tree + body


def _write_file(path: Path, grid: bytes, *, name="density", voxel_size=(0.5, 0.5, 0.5),
                index_bbox=(0, 0, 0, 7, 7, 7), voxel_count=None,
                grid_class=1, grid_type=1, magic=MAGIC_FILE, codec=0) -> Path:
    if voxel_count is None:
        voxel_count = struct.unpack_from("<Q", grid, GRID_DATA_SIZE + 56)[0]
    meta = struct.pack(
        "<QQQQ" "II" "6d" "6i" "3d" "I" "4I" "3I" "HHI",
        len(grid), len(grid), 0, voxel_count,
        grid_type, grid_class,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        *index_bbox,
        *voxel_size,
        len(name),
        1, 0, 0, 1,
        0, 0, 0,
        codec, 0, VERSION,
    )
    assert len(meta) == FILE_META_SIZE
    header = struct.pack("<QIHH", magic, VERSION, 1, codec)
    path.write_bytes(header + meta + name.encode() + grid)
    return path


@pytest.fixture
def simple_file(tmp_path: Path) -> Path:
    # Three active voxels at known leaf-local indices. n = x<<6 | y<<3 | z.
    leaf = _leaf((0, 0, 0), {0: -1.0, (1 << 6) | (2 << 3) | 3: 0.25, 511: 1.0})
    return _write_file(tmp_path / "simple.nvdb", _grid_buffer([leaf]))


def test_the_struct_sizes_match_what_the_header_asserts():
    """NanoVDB.h static_asserts these. If a transcription drifted, one breaks."""
    assert struct.calcsize("<QIHH") == FILE_HEADER_SIZE == 16
    assert struct.calcsize("<QQQQII6d6i3dI4I3IHHI") == FILE_META_SIZE == 176
    assert struct.calcsize("<4q3I3IQ") == TREE_DATA_SIZE == 64
    assert GRID_DATA_SIZE == 672


def test_version_unpacks_as_11_11_10_bits():
    assert decode_version((32 << 21) | (7 << 10) | 3) == (32, 7, 3)


def test_reads_metadata_and_grid_header(simple_file: Path):
    (grid,) = read(simple_file)
    assert grid.grid_name == "density" == grid.meta.name
    assert grid.grid_type == 1 and grid.grid_class == 1
    assert grid.voxel_size == (0.5, 0.5, 0.5)
    assert grid.translation == (1.0, 2.0, 3.0)
    assert grid.tree_voxel_count == 3


def test_walks_leaves_and_recovers_the_values(simple_file: Path):
    """The coordinates matter as much as the values: a leaf's voxel index is
    x<<6 | y<<3 | z, and getting that ordering backwards still yields the right
    COUNT of voxels in the wrong places."""
    (grid,) = read(simple_file)
    ijk, values = grid.active_voxels()
    assert len(ijk) == 3
    lookup = {tuple(c): v for c, v in zip(ijk.tolist(), values.tolist())}
    assert lookup[(0, 0, 0)] == pytest.approx(-1.0)
    assert lookup[(1, 2, 3)] == pytest.approx(0.25)
    assert lookup[(7, 7, 7)] == pytest.approx(1.0)


def test_index_to_world_applies_voxel_size_and_translation(simple_file: Path):
    (grid,) = read(simple_file)
    world = grid.index_to_world(np.array([[2, 4, 6]]))
    assert world[0].tolist() == pytest.approx([1 + 1.0, 2 + 2.0, 3 + 3.0])


def test_two_leaves_are_both_walked(tmp_path: Path):
    leaves = [_leaf((0, 0, 0), {0: 1.0}), _leaf((8, 0, 0), {0: 2.0, 1: 3.0})]
    path = _write_file(tmp_path / "two.nvdb", _grid_buffer(leaves),
                       index_bbox=(0, 0, 0, 15, 7, 7))
    (grid,) = read(path)
    ijk, values = grid.active_voxels()
    assert len(ijk) == 3
    assert sorted(values.tolist()) == pytest.approx([1.0, 2.0, 3.0])
    assert (8, 0, 0) in {tuple(c) for c in ijk.tolist()}


def test_verify_passes_a_consistent_file(simple_file: Path):
    report = verify(simple_file, expect={"name": "density", "voxel_count": 3,
                                         "grid_class": 1, "voxel_size": (0.5, 0.5, 0.5)})
    entry = report["grids"][0]
    assert entry["active_voxels_walked"] == 3
    assert entry["grid_type"] == "Float" and entry["grid_class"] == "LevelSet"


def test_legacy_magic_is_accepted(tmp_path: Path):
    leaf = _leaf((0, 0, 0), {0: 1.0})
    path = _write_file(tmp_path / "legacy.nvdb", _grid_buffer([leaf]), magic=MAGIC_NUMB)
    assert read(path)[0].tree_voxel_count == 1


def test_a_foreign_file_is_refused(tmp_path: Path):
    path = tmp_path / "not.nvdb"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(64))
    with pytest.raises(NanoVDBError, match="not NanoVDB"):
        read(path)


def test_a_compressed_file_is_refused_rather_than_misread(tmp_path: Path):
    """ZIP/BLOSC payloads would parse as garbage if the codec were ignored."""
    leaf = _leaf((0, 0, 0), {0: 1.0})
    path = _write_file(tmp_path / "zip.nvdb", _grid_buffer([leaf]), codec=1)
    with pytest.raises(NanoVDBError, match="codec"):
        read(path)


def test_a_truncated_grid_is_caught(tmp_path: Path):
    leaf = _leaf((0, 0, 0), {0: 1.0})
    path = _write_file(tmp_path / "short.nvdb", _grid_buffer([leaf]))
    raw = path.read_bytes()
    path.write_bytes(raw[:-500])
    with pytest.raises(NanoVDBError, match="remain"):
        read(path)


def test_a_voxel_count_that_disagrees_with_the_masks_is_caught(tmp_path: Path):
    """The check that actually matters. A tree claiming more voxels than its
    masks contain is exactly what a wrong bitmask or node offset produces, and
    it is invisible to anything that trusts the header."""
    leaf = _leaf((0, 0, 0), {0: 1.0})
    grid = _grid_buffer([leaf], voxel_count=99)
    path = _write_file(tmp_path / "liar.nvdb", grid, voxel_count=99)
    with pytest.raises(NanoVDBError, match="walked 1 active voxels"):
        verify(path)


def test_metadata_and_grid_header_must_agree_on_size(tmp_path: Path):
    leaf = _leaf((0, 0, 0), {0: 1.0})
    grid = bytearray(_grid_buffer([leaf]))
    struct.pack_into("<Q", grid, 32, 12345)  # corrupt GridData.mGridSize only
    path = _write_file(tmp_path / "mismatch.nvdb", bytes(grid))
    with pytest.raises(NanoVDBError, match="disagrees with the metadata"):
        read(path)


def test_voxels_outside_the_declared_bbox_are_caught(tmp_path: Path):
    leaf = _leaf((64, 0, 0), {0: 1.0})
    path = _write_file(tmp_path / "oob.nvdb", _grid_buffer([leaf]),
                       index_bbox=(0, 0, 0, 7, 7, 7))
    with pytest.raises(NanoVDBError, match="outside the declared index bbox"):
        verify(path)


def test_expectations_are_checked(simple_file: Path):
    with pytest.raises(NanoVDBError, match="voxel_count"):
        verify(simple_file, expect={"voxel_count": 4})
    with pytest.raises(NanoVDBError, match="voxel size"):
        verify(simple_file, expect={"voxel_size": (0.25, 0.25, 0.25)})


def test_a_grid_buffer_shorter_than_griddata_is_rejected():
    """file_size comes out of the file, so it can point at far too few bytes.

    Every offset _parse_grid reads is a fixed position inside GridData, and the
    tree read starts at GRID_DATA_SIZE itself, so a truncated header used to
    surface as struct.error from whichever unpack happened to go first.
    """
    from vaultwares_studio.nanovdb_read import FileMeta, _parse_grid

    meta = FileMeta(
        grid_size=GRID_DATA_SIZE, file_size=16, name_key=0, voxel_count=0,
        grid_type=1, grid_class=1, world_bbox=(0.0,) * 6, index_bbox=(0,) * 6,
        voxel_size=(1.0, 1.0, 1.0), name="density", node_count=(0, 0, 0, 1),
        tile_count=(0, 0, 0), codec=0, version=(32, 7, 0),
    )
    with pytest.raises(NanoVDBError, match="shorter than GridData"):
        _parse_grid(meta, b"\x00" * (GRID_DATA_SIZE - 1))
