"""Extract the zero isosurface of a ``.nvdb`` level set as a PLY, for looking at.

Omniverse renders the volume through the USD prim; nothing else on this
workstation draws NanoVDB. usdview's Storm ignores volumes, Blender wants
OpenVDB. So when the question is "what is in that file", this answers it with
a mesh any viewer opens, built from the file itself rather than from the TSDF
that produced it — which is the point: it shows what a consumer of the .nvdb
will get, offsets and all.

The grid is read with ``nanovdb_read`` (written independently of the writer),
densified over its index bounding box with the background value outside the
band, and run through marching cubes at level 0. Vertices come out in index
space and are mapped through the grid's own transform, so the mesh lands where
the volume does — in the volume's own coordinates. For a capture that is
DA3 world, not the scene frame: ``reconstruction/mesh.ply`` sits in the scene
frame (trainer scale and gravity rotation applied), this one does not.

    python tools/nanovdb_to_mesh.py data/jobs/<job>/reconstruction/volume.nvdb
    # -> volume_isosurface.ply beside it; --out to choose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.nanovdb_read import read  # noqa: E402

MAX_DENSE_CELLS = 400_000_000  # ~1.6 GB float32; refuse beyond this rather than swap


def isosurface(nvdb: Path, *, level: float = 0.0, step: int = 1) -> tuple[np.ndarray, np.ndarray, dict]:
    grids = read(nvdb)
    if not grids:
        raise ValueError(f"{nvdb} holds no grids")
    grid = grids[0]
    ijk, values = grid.active_voxels()
    if not len(ijk):
        raise ValueError(f"{nvdb} has no active voxels")
    lo, hi = ijk.min(axis=0), ijk.max(axis=0)
    shape = (hi - lo + 1)
    cells = int(np.prod(shape.astype(np.int64)))
    if cells > MAX_DENSE_CELLS:
        raise ValueError(f"dense extent {tuple(shape)} is {cells:,} cells; use --step")
    # The writer stores the truncation distance as the background; the band's
    # largest magnitude is the same number, and is what the file itself says.
    background = float(np.abs(values).max())
    # Inactive cells take the sign of the nearest active voxel, at background
    # magnitude. Filling them all positive was tried first and puts a phantom
    # zero crossing one band-width behind every surface, where the band's
    # negative inner edge meets the positive fill — a doubled wall in the
    # export and a sphere with a second shell inside it in the test.
    local = ijk - lo
    active = np.zeros(shape, dtype=bool)
    active[local[:, 0], local[:, 1], local[:, 2]] = True
    dense = np.zeros(shape, dtype=np.float32)
    dense[local[:, 0], local[:, 1], local[:, 2]] = values
    from scipy import ndimage

    nearest = ndimage.distance_transform_edt(~active, return_distances=False, return_indices=True)
    sign = np.sign(dense[nearest[0], nearest[1], nearest[2]])
    sign[sign == 0] = 1.0
    dense = np.where(active, dense, sign * abs(background)).astype(np.float32)
    if step > 1:
        dense = dense[::step, ::step, ::step]
        active = active[::step, ::step, ::step]

    # And only cubes whose eight corners were all observed produce triangles,
    # which is what Open3D's TSDF extraction does too. With the signed fill
    # alone, unobserved space between a wall's back and the next wall's front
    # still grows a seam where the two fills meet.
    observed = np.zeros_like(active)
    core = np.ones(tuple(s - 1 for s in active.shape), dtype=bool)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                core &= active[di:di + core.shape[0], dj:dj + core.shape[1], dk:dk + core.shape[2]]
    observed[:-1, :-1, :-1] = core

    from skimage import measure

    verts, faces, _normals, _vals = measure.marching_cubes(dense, level=level, mask=observed)
    verts = verts * step + lo  # back to index space
    world = grid.index_to_world(verts)
    report = {"grid": grid.grid_name, "active_voxels": int(len(ijk)),
              "dense_shape": [int(s) for s in shape], "vertices": int(len(world)),
              "faces": int(len(faces)), "voxel_size": float(grid.voxel_size[0]),
              "background": background}
    return world.astype(np.float32), faces.astype(np.int32), report


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nvdb", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--step", type=int, default=1, help="subsample the dense grid by this factor")
    args = parser.parse_args(argv)
    out = args.out or args.nvdb.with_name(args.nvdb.stem + "_isosurface.ply")
    started = time.perf_counter()
    vertices, faces, report = isosurface(args.nvdb, level=args.level, step=args.step)
    write_ply(out, vertices, faces)
    report["seconds"] = round(time.perf_counter() - started, 1)
    report["out"] = str(out)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
