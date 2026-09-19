"""Read a ``.nvdb`` file, so that writing one can be checked rather than hoped.

This exists before the writer and deliberately does not share a line of code
with it. A writer validated by its own reader proves only that the two agree;
if the format was misread, both are wrong together and the round trip is
silent about it. So every offset and field below is transcribed from
``nanovdb/NanoVDB.h`` (OpenVDB master) rather than from whatever the writer
happens to emit, and the sizes the header asserts are asserted here too:

    FileHeader    16 B     FileMetaData  176 B
    GridData     672 B     TreeData       64 B

If a transcription is wrong, one of those almost certainly stops matching.

**Why a verifier at all.** A subtly wrong VDB still opens. It shows geometry
that is plausibly shaped and wrong — a tree whose child offsets are off, or a
value mask off by a bit, yields a surface, just not the surface that was
computed. That is the same failure mode as the inverted depth map and the
10-image sparse fragment earlier in this project: output that looks fine and
is not. Reading the bytes back and comparing values against what went in is
the only check that catches it.

Layout, as written to disk (see ``nanovdb/io/IO.h``):

    FileHeader, FileMetaData_0, name_0, ... FileMetaData_N, name_N,
    grid_0, ... grid_N

and each grid buffer begins with GridData, then TreeData, then the root node,
the upper (32^3) and lower (16^3) internal nodes, then the 8^3 leaves.

Standard library and numpy only — nothing here links OpenVDB, which is the
point: the format is a documented, fixed binary layout and does not need a
C++ toolchain to read.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# From NanoVDB.h. The "0" variant is the legacy file magic and the "2" variant
# the current one; isValid() accepts either, so this does too.
MAGIC_NUMB = 0x304244566F6E614E  # "NanoVDB0"
MAGIC_GRID = 0x314244566F6E614E  # "NanoVDB1"
MAGIC_FILE = 0x324244566F6E614E  # "NanoVDB2"

FILE_HEADER_SIZE = 16
FILE_META_SIZE = 176
GRID_DATA_SIZE = 672
TREE_DATA_SIZE = 64

# Node extents: the tree is root -> 32^3 upper -> 16^3 lower -> 8^3 leaf.
UPPER_LOG2DIM, LOWER_LOG2DIM, LEAF_LOG2DIM = 5, 4, 3
LEAF_VOXELS = 1 << (3 * LEAF_LOG2DIM)  # 512

CODECS = {0: "NONE", 1: "ZIP", 2: "BLOSC"}
GRID_CLASSES = {0: "Unknown", 1: "LevelSet", 2: "FogVolume", 3: "Staggered",
                4: "PointIndex", 5: "PointData", 6: "Topology", 7: "VoxelVolume"}
GRID_TYPES = {0: "Unknown", 1: "Float", 2: "Double", 3: "Int16", 4: "Int32",
              5: "Int64", 6: "Vec3f", 7: "Vec3d", 8: "Mask", 9: "Half",
              10: "UInt32", 11: "Boolean"}


def decode_version(raw: int) -> tuple[int, int, int]:
    """Version packs major/minor/patch into 11 + 11 + 10 bits."""
    return (raw >> 21) & 0x7FF, (raw >> 10) & 0x7FF, raw & 0x3FF


@dataclass(frozen=True)
class FileMeta:
    """One grid's entry in the file's metadata table."""

    grid_size: int
    file_size: int
    name_key: int
    voxel_count: int
    grid_type: int
    grid_class: int
    world_bbox: tuple[float, ...]  # (minx, miny, minz, maxx, maxy, maxz)
    index_bbox: tuple[int, ...]
    voxel_size: tuple[float, float, float]
    name: str
    node_count: tuple[int, int, int, int]  # leaf, lower, upper, root
    tile_count: tuple[int, int, int]
    codec: int
    version: tuple[int, int, int]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "grid_type": GRID_TYPES.get(self.grid_type, self.grid_type),
            "grid_class": GRID_CLASSES.get(self.grid_class, self.grid_class),
            "voxel_count": self.voxel_count,
            "voxel_size": [round(v, 6) for v in self.voxel_size],
            "index_bbox": list(self.index_bbox),
            "world_bbox": [round(v, 4) for v in self.world_bbox],
            "nodes(leaf,lower,upper,root)": list(self.node_count),
            "codec": CODECS.get(self.codec, self.codec),
            "version": ".".join(str(v) for v in self.version),
        }


class NanoVDBError(ValueError):
    """The file is not a NanoVDB, or its internal bookkeeping disagrees."""


@dataclass
class Grid:
    """A parsed float grid: the header fields plus the raw buffer behind them."""

    meta: FileMeta
    buffer: bytes
    grid_name: str
    grid_class: int
    grid_type: int
    voxel_size: tuple[float, float, float]
    translation: tuple[float, float, float]
    tree_node_offsets: tuple[int, int, int, int]
    tree_node_counts: tuple[int, int, int]
    tree_voxel_count: int

    # ---- tree walk -------------------------------------------------------

    def _leaf_records(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Every leaf as ``(origin, value_mask_bits, values)``.

        Leaves are stored contiguously and breadth-first, so they can be read
        directly from ``mNodeOffset[0]`` without walking the tree at all. That
        is a property of how NanoVDB serialises rather than an assumption: the
        grid flags carry ``IsBreadthFirst`` and the offsets are explicit.
        """
        if self.grid_type != 1:
            raise NanoVDBError(
                f"only Float grids can be sampled here, got "
                f"{GRID_TYPES.get(self.grid_type, self.grid_type)}"
            )
        leaf_count = self.tree_node_counts[0]
        base = GRID_DATA_SIZE + self.tree_node_offsets[0]
        # LeafData<float, 3>: bbox_min 12 + bbox_dif 3 + flags 1 + mask 64
        #                   + min 4 + max 4 + avg 4 + stddev 4 + 512 * 4
        stride = 12 + 3 + 1 + 64 + 16 + LEAF_VOXELS * 4
        records = []
        for index in range(leaf_count):
            off = base + index * stride
            chunk = self.buffer[off:off + stride]
            if len(chunk) < stride:
                raise NanoVDBError(
                    f"leaf {index} runs past the end of the grid buffer — "
                    f"node offsets and node counts disagree"
                )
            origin = np.frombuffer(chunk, dtype="<i4", count=3)
            mask = np.frombuffer(chunk, dtype="<u8", count=8, offset=16)
            values = np.frombuffer(chunk, dtype="<f4", count=LEAF_VOXELS,
                                   offset=12 + 3 + 1 + 64 + 16)
            records.append((origin, mask, values))
        return records

    def active_voxels(self) -> tuple[np.ndarray, np.ndarray]:
        """All active voxels as ``(ijk int32 (N,3), values float32 (N,))``.

        A voxel is active when its bit is set in the leaf's value mask. Reading
        the values without consulting the mask returns the inactive background
        entries too, which is a common way to "verify" a grid that is actually
        half empty.
        """
        coords, values = [], []
        for origin, mask, leaf_values in self._leaf_records():
            bits = np.unpackbits(mask.view(np.uint8), bitorder="little")
            active = np.nonzero(bits)[0]
            if not len(active):
                continue
            # NanoVDB orders a leaf's voxels x-major: n = x<<6 | y<<3 | z.
            x = (active >> (2 * LEAF_LOG2DIM)) & ((1 << LEAF_LOG2DIM) - 1)
            y = (active >> LEAF_LOG2DIM) & ((1 << LEAF_LOG2DIM) - 1)
            z = active & ((1 << LEAF_LOG2DIM) - 1)
            coords.append(np.stack([x, y, z], axis=1).astype(np.int32) + origin)
            values.append(leaf_values[active])
        if not coords:
            return np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
        return np.concatenate(coords), np.concatenate(values)

    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        return np.asarray(ijk, dtype=np.float64) * self.voxel_size + self.translation


def read(path: Path | str) -> list[Grid]:
    """Parse every grid in a ``.nvdb`` file."""
    raw = Path(path).read_bytes()
    if len(raw) < FILE_HEADER_SIZE:
        raise NanoVDBError(f"{path}: {len(raw)} bytes is too short to be a NanoVDB file")

    magic, version_raw, grid_count, codec = struct.unpack_from("<QIHH", raw, 0)
    if magic not in (MAGIC_NUMB, MAGIC_FILE):
        raise NanoVDBError(
            f"{path}: magic 0x{magic:016x} is not NanoVDB "
            f"(expected 0x{MAGIC_NUMB:016x} or 0x{MAGIC_FILE:016x})"
        )
    if codec != 0:
        raise NanoVDBError(
            f"{path}: codec is {CODECS.get(codec, codec)}; only uncompressed "
            "files are read here (ZIP/BLOSC would need those libraries)"
        )

    grids: list[Grid] = []
    cursor = FILE_HEADER_SIZE
    metas: list[FileMeta] = []
    for _ in range(grid_count):
        fields = struct.unpack_from("<QQQQ" "II" "6d" "6i" "3d" "I" "4I" "3I" "HHI",
                                    raw, cursor)
        # Index map for the format string above, which calcsize()s to exactly
        # the 176 bytes NanoVDB.h asserts: 0-3 sizes, 4 gridType, 5 gridClass,
        # 6-11 worldBBox, 12-17 indexBBox, 18-20 voxelSize, 21 nameSize,
        # 22-25 nodeCount, 26-28 tileCount, 29 codec, 30 blindDataCount,
        # 31 version.
        name_size = fields[21]
        name_off = cursor + FILE_META_SIZE
        name = raw[name_off:name_off + name_size].split(b"\x00")[0].decode("utf-8")
        metas.append(FileMeta(
            grid_size=fields[0], file_size=fields[1], name_key=fields[2],
            voxel_count=fields[3], grid_type=fields[4], grid_class=fields[5],
            world_bbox=fields[6:12], index_bbox=fields[12:18],
            voxel_size=fields[18:21], name=name,
            node_count=fields[22:26], tile_count=fields[26:29],
            codec=fields[29], version=decode_version(fields[31]),
        ))
        cursor = name_off + name_size

    for meta in metas:
        buffer = raw[cursor:cursor + meta.file_size]
        if len(buffer) != meta.file_size:
            raise NanoVDBError(
                f"{path}: grid '{meta.name}' claims {meta.file_size} bytes but only "
                f"{len(buffer)} remain"
            )
        grids.append(_parse_grid(meta, buffer))
        cursor += meta.file_size
    return grids


def _parse_grid(meta: FileMeta, buffer: bytes) -> Grid:
    # Every offset below is a fixed position inside GridData, and the tree read
    # at the end starts at GRID_DATA_SIZE itself. meta.file_size came out of the
    # file, so a truncated or hostile header can point here with far less than
    # that, and struct.unpack_from would raise struct.error from six different
    # places. Say what is wrong once, in terms of the grid.
    if len(buffer) < GRID_DATA_SIZE:
        raise NanoVDBError(
            f"grid '{meta.name}': {len(buffer)} bytes is shorter than GridData's "
            f"{GRID_DATA_SIZE}"
        )
    grid_magic = struct.unpack_from("<Q", buffer, 0)[0]
    if grid_magic not in (MAGIC_NUMB, MAGIC_GRID):
        raise NanoVDBError(
            f"grid '{meta.name}': buffer magic 0x{grid_magic:016x} is not a NanoVDB grid"
        )
    grid_size = struct.unpack_from("<Q", buffer, 32)[0]
    if grid_size != meta.grid_size:
        raise NanoVDBError(
            f"grid '{meta.name}': GridData.mGridSize {grid_size} disagrees with the "
            f"metadata's {meta.grid_size}"
        )
    name = buffer[40:40 + 256].split(b"\x00")[0].decode("utf-8")

    # Map is at offset 296: matf[9], invMatf[9], vecf[3], taperf, matd[9],
    # invMatd[9], vecd[3], taperd. The double translation is what places the
    # grid in world space, at 296 + 36+36+12+4 + 72+72 = 528.
    translation = struct.unpack_from("<3d", buffer, 296 + 36 + 36 + 12 + 4 + 72 + 72)
    voxel_size = struct.unpack_from("<3d", buffer, 608)
    grid_class, grid_type = struct.unpack_from("<II", buffer, 632)

    tree = struct.unpack_from("<4q3I3IQ", buffer, GRID_DATA_SIZE)
    return Grid(
        meta=meta, buffer=buffer, grid_name=name,
        grid_class=grid_class, grid_type=grid_type,
        voxel_size=voxel_size, translation=translation,
        tree_node_offsets=tree[0:4], tree_node_counts=tree[4:7],
        tree_voxel_count=tree[10],
    )


def verify(path: Path | str, *, expect: dict | None = None) -> dict:
    """Parse a file and check it against itself, and optionally against expectations.

    ``expect`` may carry ``voxel_count``, ``voxel_size``, ``grid_class`` and
    ``name``. Everything else is consistency between the file's own claims:
    metadata versus grid header versus tree.

    Returns a report. Raises on anything that makes the file wrong rather than
    merely surprising.
    """
    grids = read(path)
    if not grids:
        raise NanoVDBError(f"{path}: no grids")
    report = {"path": str(path), "grids": []}
    for grid in grids:
        ijk, values = grid.active_voxels()
        entry = dict(grid.meta.as_dict())
        entry["active_voxels_walked"] = int(len(ijk))

        if grid.grid_name != grid.meta.name:
            raise NanoVDBError(
                f"grid name disagrees: metadata says '{grid.meta.name}', "
                f"GridData says '{grid.grid_name}'"
            )
        if grid.tree_voxel_count != grid.meta.voxel_count:
            raise NanoVDBError(
                f"'{grid.meta.name}': tree says {grid.tree_voxel_count} active voxels, "
                f"metadata says {grid.meta.voxel_count}"
            )
        # The strongest internal check available: the count claimed by the
        # bookkeeping against the count actually reachable by walking leaf
        # masks. A tree with wrong offsets or masks fails here and nowhere else.
        if len(ijk) != grid.tree_voxel_count:
            raise NanoVDBError(
                f"'{grid.meta.name}': walked {len(ijk)} active voxels but the tree "
                f"claims {grid.tree_voxel_count} — masks or node offsets are wrong"
            )
        if len(ijk):
            lo, hi = ijk.min(axis=0), ijk.max(axis=0)
            claimed_lo = np.array(grid.meta.index_bbox[:3])
            claimed_hi = np.array(grid.meta.index_bbox[3:])
            if (lo < claimed_lo).any() or (hi > claimed_hi).any():
                raise NanoVDBError(
                    f"'{grid.meta.name}': active voxels span {lo.tolist()}..{hi.tolist()} "
                    f"outside the declared index bbox "
                    f"{claimed_lo.tolist()}..{claimed_hi.tolist()}"
                )
            entry["value_range"] = [float(values.min()), float(values.max())]

        if expect:
            for key, getter in (
                ("voxel_count", lambda g: g.tree_voxel_count),
                ("grid_class", lambda g: g.grid_class),
                ("name", lambda g: g.grid_name),
            ):
                if key in expect and getter(grid) != expect[key]:
                    raise NanoVDBError(
                        f"'{grid.meta.name}': {key} is {getter(grid)!r}, expected "
                        f"{expect[key]!r}"
                    )
            if "voxel_size" in expect:
                if not np.allclose(grid.voxel_size, expect["voxel_size"], rtol=1e-9):
                    raise NanoVDBError(
                        f"'{grid.meta.name}': voxel size {grid.voxel_size} != "
                        f"{expect['voxel_size']}"
                    )
        report["grids"].append(entry)
    return report
