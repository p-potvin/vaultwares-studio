"""volume_asset and the level-set occupancy feed.

Both failure modes here are silent. A USD stage missing the field relationship
opens and renders nothing; an occupancy grid built without the scene transform
is a perfectly ordinary grid of the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaultwares_studio.nanovdb_write import write_float_grid
from vaultwares_studio.robot_lab.occupancy import FREE, OCCUPIED, grid_from_level_set
from vaultwares_studio.volume_asset import volume_to_usd


def _slab(tmp_path, name="surface", voxel=0.05):
    """A horizontal slab with a bump, as a level set: something with a floor."""
    xs, zs = np.meshgrid(np.arange(-20, 21), np.arange(-20, 21), indexing="ij")
    coords, values = [], []
    for dy in range(-2, 3):
        ijk = np.stack([xs.ravel(), np.full(xs.size, dy), zs.ravel()], axis=1)
        coords.append(ijk)
        values.append(np.full(xs.size, dy * voxel, dtype=np.float32))
    # A block standing on the slab, to give the body band an obstacle.
    bx, by, bz = np.meshgrid(np.arange(4, 9), np.arange(3, 12), np.arange(4, 9),
                             indexing="ij")
    coords.append(np.stack([bx.ravel(), by.ravel(), bz.ravel()], axis=1))
    values.append(np.zeros(bx.size, dtype=np.float32))

    ijk = np.concatenate(coords).astype(np.int32)
    values = np.concatenate(values).astype(np.float32)
    _, first = np.unique((ijk.astype(np.int64) + 1000) @ np.array([10**8, 10**4, 1]),
                         return_index=True)
    ijk, values = ijk[np.sort(first)], values[np.sort(first)]
    path = tmp_path / "slab.nvdb"
    write_float_grid(path, ijk, values, voxel_size=voxel, name=name)
    return path


def test_usd_layer_wires_the_field_relationship(tmp_path):
    """The relationship is what connects volume to field. Without it the stage
    is valid, opens cleanly, and renders nothing at all."""
    from pxr import Usd, UsdVol

    nvdb = _slab(tmp_path)
    usd = volume_to_usd(nvdb, tmp_path / "volume.usda", field_name="surface")
    stage = Usd.Stage.Open(str(usd))
    volume = UsdVol.Volume(stage.GetPrimAtPath("/World/Volume"))
    assert volume, "no Volume prim"
    fields = volume.GetFieldPaths()
    assert "surface" in fields
    assert str(fields["surface"]) == "/World/Volume/surface"


def test_the_asset_path_is_relative(tmp_path):
    """An absolute path works on the machine that wrote it and nowhere else."""
    nvdb = _slab(tmp_path)
    usd = volume_to_usd(nvdb, tmp_path / "volume.usda")
    text = usd.read_text()
    assert "@slab.nvdb@" in text
    assert str(tmp_path) not in text


def test_the_field_is_labelled_a_level_set(tmp_path):
    """fieldClass tells a consumer how to read the values. Labelled as fog
    density, the negative interior renders as nothing."""
    nvdb = _slab(tmp_path)
    usd = volume_to_usd(nvdb, tmp_path / "volume.usda")
    text = usd.read_text()
    assert 'fieldClass = "levelSet"' in text
    assert 'fieldDataType = "float"' in text


def test_field_name_must_match_the_grid(tmp_path):
    """The name is the lookup key; a mismatch yields an empty volume, not an
    error, so it is worth being able to see what was written."""
    nvdb = _slab(tmp_path, name="density")
    usd = volume_to_usd(nvdb, tmp_path / "volume.usda", field_name="density")
    assert 'fieldName = "density"' in usd.read_text()


def test_occupancy_from_a_level_set_finds_floor_and_obstacle(tmp_path):
    """The slab spans y -0.10..0.10 and the block sits at 0.15..0.55, so a body
    band starting above the slab leaves the slab as floor and the block as the
    only obstacle. A band that dips into the slab marks every column occupied,
    which is a property of the band rather than of the geometry."""
    nvdb = _slab(tmp_path)
    grid = grid_from_level_set(nvdb, cell_size=0.05, body_band=(0.22, 0.70),
                               min_support=1)
    assert (grid.cells == FREE).any(), "no floor found"
    assert (grid.cells == OCCUPIED).any(), "the block was not seen as an obstacle"


def test_the_scene_transform_is_applied(tmp_path):
    """Without it the grid is a perfectly ordinary grid of the wrong place."""
    nvdb = _slab(tmp_path)
    plain = grid_from_level_set(nvdb, cell_size=0.05, min_support=1)
    shifted = np.eye(4)
    shifted[:3, 3] = (10.0, 0.0, 0.0)
    moved = grid_from_level_set(nvdb, scene_transform=shifted, cell_size=0.05,
                                min_support=1)
    assert moved.origin[0] - plain.origin[0] == pytest.approx(10.0, abs=0.1)


def test_a_band_that_selects_nothing_is_refused(tmp_path):
    """Silently returning an empty grid would look like an empty scene.

    The slab contains exact zeros, so even an absurdly tight band finds
    something there. This grid deliberately has none — every value is a full
    voxel from the surface — which is the case that would otherwise pass an
    empty point set to grid_from_points.
    """
    ijk = np.stack(np.meshgrid(np.arange(8), np.arange(8), np.arange(8),
                               indexing="ij"), axis=-1).reshape(-1, 3).astype(np.int32)
    values = np.full(len(ijk), 0.05, dtype=np.float32)
    path = tmp_path / "offset.nvdb"
    write_float_grid(path, ijk, values, voxel_size=0.05, name="surface")
    with pytest.raises(ValueError, match="no voxel within"):
        grid_from_level_set(path, surface_band=1e-9)


def test_the_band_selects_the_surface_not_the_whole_volume(tmp_path):
    """surface_band is why this beats a point cloud: it means 'within N metres
    of a real surface', which a cloud cannot express."""
    nvdb = _slab(tmp_path, voxel=0.05)
    tight = grid_from_level_set(nvdb, surface_band=0.02, cell_size=0.05, min_support=1)
    loose = grid_from_level_set(nvdb, surface_band=0.20, cell_size=0.05, min_support=1)
    assert int((loose.cells != 0).sum()) >= int((tight.cells != 0).sum())
