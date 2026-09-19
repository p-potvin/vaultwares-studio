"""The isosurface export reads the file, not the writer's memory.

A sphere is the level set whose zero crossing is known exactly, so the mesh
that comes back has to sit at that radius in world units, through the grid's
own voxel size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from vaultwares_studio.nanovdb_write import write_float_grid  # noqa: E402
from nanovdb_to_mesh import isosurface  # noqa: E402


def test_sphere_level_set_comes_back_at_its_radius(tmp_path):
    voxel, radius, band = 0.05, 1.0, 0.2
    r = np.arange(-26, 27)
    i, j, k = np.meshgrid(r, r, r, indexing="ij")
    ijk = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(np.int32)
    # Integer index is the voxel centre in NanoVDB: index_to_world(i) = i * voxel.
    distance = np.linalg.norm(ijk * voxel, axis=1) - radius
    keep = np.abs(distance) <= band
    path = tmp_path / "sphere.nvdb"
    write_float_grid(path, ijk[keep], distance[keep].astype(np.float32), voxel_size=voxel,
                     name="sphere", grid_class=1, background=band)

    vertices, faces, report = isosurface(path)
    assert report["grid"] == "sphere"
    assert len(faces) > 1000
    radii = np.linalg.norm(vertices, axis=1)
    # Marching cubes interpolates within a voxel; half a voxel is generous.
    assert abs(np.median(radii) - radius) < voxel / 2
    assert np.percentile(radii, 99) - np.percentile(radii, 1) < voxel
