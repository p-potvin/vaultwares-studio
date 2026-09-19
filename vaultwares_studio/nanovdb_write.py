"""Write a ``.nvdb`` float grid from numpy arrays, without linking OpenVDB.

The format is a fixed binary layout, so a C++ toolchain is a convenience rather
than a requirement. Every offset used here is cross-checked against two
independent statements of the spec:

* ``nanovdb/NanoVDB.h`` — the C++ structs, with their ``static_assert``ed sizes
* ``nanovdb/PNanoVDB.h`` — the portable C99/HLSL reader, which states every
  offset as an explicit ``#define`` and carries a per-grid-type table of
  node sizes

Those two agreeing with each other, and with ``nanovdb_read``'s transcription
derived before either was consulted, is the evidence that the layout is right.
For a Float grid PNanoVDB's table gives ``leaf_size 2144``, ``lower_size
33856``, ``upper_size 270400``, ``root_size 64``, ``root_tile_size 32``.

**The tree.** NanoVDB is a fixed four-level structure, not a general octree:

    root  -> upper 32^3 children, each covering 4096^3 voxels (TOTAL 12)
          -> lower 16^3 children, each covering  128^3 voxels (TOTAL  7)
          -> leaf   8^3 voxels                                (TOTAL  3)

A node's index within its parent comes from the bits above its own TOTAL, and a
node's origin is its coordinate with the low TOTAL bits cleared. Getting either
wrong still produces a file that opens — the voxels simply land somewhere else —
which is why ``nanovdb_read.verify`` exists and why it is the acceptance gate
rather than a look at the render.

**Child pointers are relative byte offsets**, signed, each measured from the
node that holds them. Absolute offsets would also "work" for a file written and
read in one process and fail the moment the buffer moved.

**Checksum is left EMPTY** (all 64 bits set), which NanoVDB defines as
"disabled or undefined". A wrong checksum would be rejected; an absent one is
legal.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .nanovdb_read import (
    FILE_META_SIZE,
    GRID_DATA_SIZE,
    MAGIC_FILE,
    MAGIC_GRID,
    TREE_DATA_SIZE,
)

# Sharing the spec constants with the reader is deliberate; sharing encode /
# decode logic is not. One transcription of the format, two implementations
# that must agree about what the bytes mean.

CHECKSUM_EMPTY = 0xFFFFFFFFFFFFFFFF
# NanoVDB's own version, as of the header this was written against.
VERSION = (32 << 21) | (7 << 10) | 0

# Node geometry. TOTAL is how many coordinate bits a node spans.
LEAF_TOTAL, LOWER_TOTAL, UPPER_TOTAL = 3, 7, 12
LEAF_SIZE, LOWER_SIZE, UPPER_SIZE = 2144, 33856, 270400
ROOT_BASE_SIZE, ROOT_TILE_SIZE = 64, 32

# Offsets within each node, from PNanoVDB's float row.
LEAF_OFF_VALUE_MASK, LEAF_OFF_TABLE = 16, 96
LOWER_OFF_VALUE_MASK, LOWER_OFF_CHILD_MASK, LOWER_OFF_TABLE = 32, 544, 1088
UPPER_OFF_VALUE_MASK, UPPER_OFF_CHILD_MASK, UPPER_OFF_TABLE = 32, 4128, 8256
ROOT_OFF_TABLE_SIZE, ROOT_OFF_BACKGROUND = 24, 28

GRID_FLAGS_HAS_BBOX = 1 << 3
GRID_FLAG_IS_BREADTH_FIRST = 1 << 6


def _origin(ijk: np.ndarray, total: int) -> np.ndarray:
    """A node's origin: the coordinate with its own span's bits cleared.

    Arithmetic shift, so it stays correct for negative coordinates — masking
    with ``& ~m`` on a signed array is the same thing, but this says why.
    """
    return (ijk >> total) << total


def _child_index(ijk: np.ndarray, child_total: int, log2dim: int) -> np.ndarray:
    """Index of a child within its parent's table: x-major, like the leaf."""
    shifted = ijk >> child_total
    mask = (1 << log2dim) - 1
    return (((shifted[:, 0] & mask) << (2 * log2dim))
            | ((shifted[:, 1] & mask) << log2dim)
            | (shifted[:, 2] & mask))


def _root_key(origin: np.ndarray) -> np.ndarray:
    """RootData's single 64-bit key, exactly as CoordToKey builds it.

    The C++ casts each signed coordinate to uint32 *before* shifting right, so
    negative coordinates become large unsigned values rather than sign-extending.
    Reproducing that cast is the whole subtlety: do it wrong and every node in
    negative space gets a different key than the reader will compute.
    """
    u = origin.astype(np.int64) & 0xFFFFFFFF
    return ((u[:, 2] >> UPPER_TOTAL)
            | ((u[:, 1] >> UPPER_TOTAL) << 21)
            | ((u[:, 0] >> UPPER_TOTAL) << 42)).astype(np.uint64)


def _set_bits(size_bytes: int, indices: np.ndarray) -> bytes:
    bits = np.zeros(size_bytes * 8, dtype=np.uint8)
    bits[indices] = 1
    return np.packbits(bits, bitorder="little").tobytes()


@dataclass(frozen=True)
class WriteReport:
    leaves: int
    lower: int
    upper: int
    root_tiles: int
    voxels: int
    index_bbox: tuple[int, int, int, int, int, int]
    grid_bytes: int

    def as_dict(self) -> dict:
        return {
            "leaves": self.leaves, "lower": self.lower, "upper": self.upper,
            "root_tiles": self.root_tiles, "voxels": self.voxels,
            "index_bbox": list(self.index_bbox), "grid_bytes": self.grid_bytes,
        }


def write_float_grid(
    path: Path | str,
    ijk: np.ndarray,
    values: np.ndarray,
    *,
    voxel_size: float | tuple[float, float, float] = 1.0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    name: str = "density",
    grid_class: int = 1,  # LevelSet
    background: float = 0.0,
) -> WriteReport:
    """Write one Float grid containing exactly the given active voxels."""
    ijk = np.ascontiguousarray(np.asarray(ijk, dtype=np.int32).reshape(-1, 3))
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(ijk) != len(values):
        raise ValueError(f"{len(ijk)} coordinates but {len(values)} values")
    if len(ijk) == 0:
        raise ValueError("refusing to write a grid with no active voxels")
    if len(name.encode()) > 255:
        raise ValueError("grid name must fit in GridData's 256-byte field")
    if isinstance(voxel_size, (int, float)):
        voxel_size = (float(voxel_size),) * 3

    # Duplicate coordinates would set a mask bit once and write the value twice,
    # so the file would claim fewer voxels than it stores. Last one wins, but
    # say so rather than silently disagreeing with the caller's count.
    packed_all = (ijk.astype(np.int64) + (1 << 20))
    # 21 bits per axis, same packing as tsdf_volume.pack_keys, which has always
    # checked this and this has not. Out of range, the shifts collide instead of
    # overflowing, so the duplicate check below would pass on coordinates that
    # are not actually distinct and the masks would disagree with the values.
    if packed_all.min() < 0 or packed_all.max() >= (1 << 21):
        raise ValueError(
            "voxel index outside the 21-bit packing range; coordinates must be "
            f"within [{-(1 << 20)}, {(1 << 20) - 1}]"
        )
    flat = (packed_all[:, 0] << 42) | (packed_all[:, 1] << 21) | packed_all[:, 2]
    unique_flat, first = np.unique(flat, return_index=True)
    if len(unique_flat) != len(ijk):
        raise ValueError(
            f"{len(ijk) - len(unique_flat)} duplicate coordinates; de-duplicate "
            "before writing or the voxel count will not match the masks"
        )

    # ---- group voxels into the three node levels -------------------------
    leaf_origin = _origin(ijk, LEAF_TOTAL)
    lower_origin = _origin(ijk, LOWER_TOTAL)
    upper_origin = _origin(ijk, UPPER_TOTAL)

    def _group(origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keyed = (origins.astype(np.int64) + (1 << 20))
        key = (keyed[:, 0] << 42) | (keyed[:, 1] << 21) | keyed[:, 2]
        uniq, inverse = np.unique(key, return_inverse=True)
        representative = np.zeros((len(uniq), 3), dtype=np.int32)
        representative[inverse] = origins
        return representative, inverse

    leaf_origins, voxel_leaf = _group(leaf_origin)
    lower_origins, _ = _group(lower_origin)
    upper_origins, _ = _group(upper_origin)

    n_leaf, n_lower, n_upper = len(leaf_origins), len(lower_origins), len(upper_origins)

    # Parent of each node, by the same grouping.
    leaf_lower_origin = _origin(leaf_origins, LOWER_TOTAL)
    lower_upper_origin = _origin(lower_origins, UPPER_TOTAL)

    def _index_of(origins: np.ndarray, table: np.ndarray) -> np.ndarray:
        keyed = (table.astype(np.int64) + (1 << 20))
        table_key = (keyed[:, 0] << 42) | (keyed[:, 1] << 21) | keyed[:, 2]
        order = np.argsort(table_key)
        keyed_q = (origins.astype(np.int64) + (1 << 20))
        query = (keyed_q[:, 0] << 42) | (keyed_q[:, 1] << 21) | keyed_q[:, 2]
        return order[np.searchsorted(table_key[order], query)]

    leaf_parent = _index_of(leaf_lower_origin, lower_origins)
    lower_parent = _index_of(lower_upper_origin, upper_origins)

    # ---- buffer layout ---------------------------------------------------
    root_size = ROOT_BASE_SIZE + n_upper * ROOT_TILE_SIZE
    off_root = GRID_DATA_SIZE + TREE_DATA_SIZE
    off_upper = off_root + root_size
    off_lower = off_upper + n_upper * UPPER_SIZE
    off_leaf = off_lower + n_lower * LOWER_SIZE
    grid_size = off_leaf + n_leaf * LEAF_SIZE

    buf = bytearray(grid_size)

    # ---- leaves ----------------------------------------------------------
    voxel_index_in_leaf = (((ijk[:, 0] & 7) << 6)
                           | ((ijk[:, 1] & 7) << 3)
                           | (ijk[:, 2] & 7))
    order = np.argsort(voxel_leaf, kind="stable")
    boundaries = np.searchsorted(voxel_leaf[order], np.arange(n_leaf + 1))
    for index in range(n_leaf):
        members = order[boundaries[index]:boundaries[index + 1]]
        base = off_leaf + index * LEAF_SIZE
        slots = voxel_index_in_leaf[members]
        leaf_values = np.zeros(512, dtype="<f4")
        leaf_values[slots] = values[members]
        present = values[members]
        struct.pack_into("<3i", buf, base, *leaf_origins[index].tolist())
        buf[base + 15] = GRID_FLAGS_HAS_BBOX >> 2  # bit1: has bbox
        buf[base + LEAF_OFF_VALUE_MASK:base + LEAF_OFF_VALUE_MASK + 64] = _set_bits(64, slots)
        struct.pack_into("<ffff", buf, base + 80, float(present.min()),
                         float(present.max()), float(present.mean()), 0.0)
        buf[base + LEAF_OFF_TABLE:base + LEAF_OFF_TABLE + 2048] = leaf_values.tobytes()

    # ---- lower internal nodes -------------------------------------------
    leaf_slot = _child_index(leaf_origins, LEAF_TOTAL, 4)
    for index in range(n_lower):
        base = off_lower + index * LOWER_SIZE
        children = np.nonzero(leaf_parent == index)[0]
        slots = leaf_slot[children]
        struct.pack_into("<3i", buf, base, *lower_origins[index].tolist())
        struct.pack_into("<3i", buf, base + 12,
                         *(lower_origins[index] + (1 << LOWER_TOTAL) - 1).tolist())
        mask = _set_bits(512, slots)
        # A child slot is NOT an active value slot: the value mask marks tiles
        # that hold a constant, the child mask marks slots holding a node.
        buf[base + LOWER_OFF_CHILD_MASK:base + LOWER_OFF_CHILD_MASK + 512] = mask
        table = np.zeros(4096, dtype="<i8")
        table[slots] = (off_leaf + children * LEAF_SIZE) - base
        buf[base + LOWER_OFF_TABLE:base + LOWER_OFF_TABLE + 4096 * 8] = table.tobytes()

    # ---- upper internal nodes -------------------------------------------
    lower_slot = _child_index(lower_origins, LOWER_TOTAL, 5)
    for index in range(n_upper):
        base = off_upper + index * UPPER_SIZE
        children = np.nonzero(lower_parent == index)[0]
        slots = lower_slot[children]
        struct.pack_into("<3i", buf, base, *upper_origins[index].tolist())
        struct.pack_into("<3i", buf, base + 12,
                         *(upper_origins[index] + (1 << UPPER_TOTAL) - 1).tolist())
        buf[base + UPPER_OFF_CHILD_MASK:base + UPPER_OFF_CHILD_MASK + 4096] = _set_bits(4096, slots)
        table = np.zeros(32768, dtype="<i8")
        table[slots] = (off_lower + children * LOWER_SIZE) - base
        buf[base + UPPER_OFF_TABLE:base + UPPER_OFF_TABLE + 32768 * 8] = table.tobytes()

    # ---- root ------------------------------------------------------------
    lo = ijk.min(axis=0)
    hi = ijk.max(axis=0)
    struct.pack_into("<3i", buf, off_root, *lo.tolist())
    struct.pack_into("<3i", buf, off_root + 12, *hi.tolist())
    struct.pack_into("<I", buf, off_root + ROOT_OFF_TABLE_SIZE, n_upper)
    struct.pack_into("<fffff", buf, off_root + ROOT_OFF_BACKGROUND,
                     background, float(values.min()), float(values.max()),
                     float(values.mean()), 0.0)
    keys = _root_key(upper_origins)
    tile_order = np.argsort(keys)  # the root table is sorted by key
    for slot, index in enumerate(tile_order):
        tile = off_root + ROOT_BASE_SIZE + slot * ROOT_TILE_SIZE
        struct.pack_into("<Q", buf, tile, int(keys[index]))
        struct.pack_into("<q", buf, tile + 8,
                         (off_upper + int(index) * UPPER_SIZE) - off_root)
        struct.pack_into("<I", buf, tile + 16, 0)   # state: a child, not a tile value
        struct.pack_into("<f", buf, tile + 20, background)

    # ---- tree ------------------------------------------------------------
    tree = off_root - TREE_DATA_SIZE  # TreeData starts here; offsets are from it
    struct.pack_into(
        "<4q3I3IQ", buf, GRID_DATA_SIZE,
        off_leaf - GRID_DATA_SIZE, off_lower - GRID_DATA_SIZE,
        off_upper - GRID_DATA_SIZE, off_root - GRID_DATA_SIZE,
        n_leaf, n_lower, n_upper,
        0, 0, 0,
        len(ijk),
    )

    # ---- grid header -----------------------------------------------------
    struct.pack_into("<Q", buf, 0, MAGIC_GRID)
    struct.pack_into("<Q", buf, 8, CHECKSUM_EMPTY)
    struct.pack_into("<I", buf, 16, VERSION)
    struct.pack_into("<I", buf, 20, GRID_FLAGS_HAS_BBOX | GRID_FLAG_IS_BREADTH_FIRST)
    struct.pack_into("<II", buf, 24, 0, 1)      # grid index, grid count
    struct.pack_into("<Q", buf, 32, grid_size)
    encoded = name.encode()
    buf[40:40 + len(encoded)] = encoded

    # Map: index-to-world is a diagonal scale plus a translation. Both the
    # float and double copies are filled — NanoVDB keeps both precisions and a
    # consumer may read either.
    scale = np.diag(voxel_size).astype(np.float64)
    inverse = np.diag([1.0 / v for v in voxel_size]).astype(np.float64)
    m = GRID_DATA_SIZE - (GRID_DATA_SIZE - 296)  # 296, written out for clarity
    struct.pack_into("<9f", buf, 296, *scale.astype(np.float32).reshape(-1).tolist())
    struct.pack_into("<9f", buf, 296 + 36, *inverse.astype(np.float32).reshape(-1).tolist())
    struct.pack_into("<3f", buf, 296 + 72, *[float(t) for t in translation])
    struct.pack_into("<f", buf, 296 + 84, 1.0)   # taperf
    struct.pack_into("<9d", buf, 296 + 88, *scale.reshape(-1).tolist())
    struct.pack_into("<9d", buf, 296 + 160, *inverse.reshape(-1).tolist())
    struct.pack_into("<3d", buf, 296 + 232, *[float(t) for t in translation])
    struct.pack_into("<d", buf, 296 + 256, 1.0)  # taperd

    world_lo = lo * np.asarray(voxel_size) + np.asarray(translation)
    world_hi = (hi + 1) * np.asarray(voxel_size) + np.asarray(translation)
    struct.pack_into("<6d", buf, 560, *world_lo.tolist(), *world_hi.tolist())
    struct.pack_into("<3d", buf, 608, *voxel_size)
    struct.pack_into("<II", buf, 632, grid_class, 1)  # class, GridType::Float
    struct.pack_into("<q", buf, 640, 0)   # blind metadata offset
    struct.pack_into("<I", buf, 648, 0)   # blind metadata count

    index_bbox = (*lo.tolist(), *hi.tolist())
    _write_file(Path(path), bytes(buf), name=name, voxel_size=voxel_size,
                translation=translation, index_bbox=index_bbox,
                voxel_count=len(ijk), grid_class=grid_class,
                world_bbox=(*world_lo.tolist(), *world_hi.tolist()),
                node_count=(n_leaf, n_lower, n_upper, 1))

    return WriteReport(leaves=n_leaf, lower=n_lower, upper=n_upper,
                       root_tiles=n_upper, voxels=len(ijk),
                       index_bbox=index_bbox, grid_bytes=grid_size)


def _write_file(path: Path, grid: bytes, *, name: str, voxel_size, translation,
                index_bbox, voxel_count: int, grid_class: int, world_bbox,
                node_count) -> None:
    """FileHeader, then one FileMetaData + name, then the grid buffer."""
    meta = struct.pack(
        "<QQQQ" "II" "6d" "6i" "3d" "I" "4I" "3I" "HHI",
        len(grid), len(grid), 0, voxel_count,
        1, grid_class,                      # GridType::Float, class
        *world_bbox,
        *index_bbox,
        *voxel_size,
        len(name.encode()),
        *node_count,
        0, 0, 0,                            # tile counts
        0, 0, VERSION,                      # codec NONE, no blind data
    )
    if len(meta) != FILE_META_SIZE:
        raise AssertionError(f"FileMetaData is {len(meta)}B, must be {FILE_META_SIZE}")
    header = struct.pack("<QIHH", MAGIC_FILE, VERSION, 1, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + meta + name.encode() + grid)
