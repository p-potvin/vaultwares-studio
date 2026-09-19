"""Free space from depth rays, filling what the level set cannot.

A TSDF stores nothing far from a surface, so an occupancy grid fed from it
leaves every cell no surface came near as UNKNOWN. The rays know better: each
depth sample is a segment the camera saw straight through. These tests pin the
geometry of that carve on a scene small enough to reason about by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaultwares_studio.robot_lab.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid,
    carve_free_space,
    cells_free_fraction,
    grid_from_depth_frames,
    ray_pass_counts,
)

CELL = 0.05
BAND = (0.10, 0.60)


def _grid() -> OccupancyGrid:
    """20x20 cells over world x, z in [0, 1], floor at y = 0, all unknown."""
    return OccupancyGrid(cells=np.full((20, 20), UNKNOWN, dtype=np.uint8),
                         origin=(0.0, 0.0), cell_size=CELL, floor_y=0.0)


def _frame(height: float = 0.4, depth: float = 1.0, size: int = 8) -> dict:
    """One camera at (0.5, height, 0.05), OpenCV axes aligned with the world,
    looking down +z with a 90-degree field of view and a flat depth map."""
    c2w = np.eye(4)
    c2w[:3, 3] = [0.5, height, 0.05]
    f = size / 2
    intrinsics = np.array([[f, 0, f], [0, f, f], [0, 0, 1]], dtype=np.float64)
    return {"depth": np.full((size, size), depth, dtype=np.float32),
            "intrinsics": intrinsics, "c2w": c2w}


def test_rays_count_ahead_of_the_camera_and_not_behind():
    grid = _grid()
    passes = ray_pass_counts(grid, [_frame()], body_band=BAND, pixel_stride=1)
    ahead = passes[5, 10]           # z = 0.275, straight ahead of x = 0.5
    assert ahead >= 3, passes[:8, 8:13]
    assert passes[0, :].sum() == 0  # z < 0.05 is behind the camera


def test_a_ray_counts_once_per_cell():
    """Sampling at half a cell puts several samples in every cell a ray crosses;
    the count must not scale with the sampling."""
    grid = _grid()
    coarse = ray_pass_counts(grid, [_frame()], body_band=BAND, pixel_stride=1, max_steps=16)
    fine = ray_pass_counts(grid, [_frame()], body_band=BAND, pixel_stride=1, max_steps=256)
    rays = 8 * 8
    # No cell can be crossed by more rays than there are, whatever the sampling.
    assert fine.max() <= rays and coarse.max() <= rays
    # Straight ahead every row's ray is in the band: all eight, counted once each.
    assert fine[5, 10] == coarse[5, 10] == 8
    # Denser sampling finds more grazed corners; it must never inflate a cell
    # past the rays that cross it (that is the max check above), and the cells
    # it adds are the marginal ones. On this 20x20 toy the gap is ~15%.
    assert coarse.sum() <= fine.sum() <= 1.3 * coarse.sum()


def test_carve_promotes_unknown_only_and_leaves_obstacles_alone():
    grid = _grid()
    grid.cells[5, 10] = OCCUPIED
    passes = ray_pass_counts(grid, [_frame()], body_band=BAND, pixel_stride=1)
    assert passes[5, 10] >= 3, "the obstacle cell is crossed by rays, which is the point"
    carved = carve_free_space(grid, passes, min_passes=3)
    assert carved.cells[5, 10] == OCCUPIED
    # Column 11 is the +0.25 ray of the fan (x = 0.556 at that row); column 9
    # sits between rays of an 8-wide fan and is legitimately still unknown.
    assert passes[5, 11] >= 3
    assert carved.cells[5, 11] == FREE
    assert carved.cells[0, 10] == UNKNOWN
    # The input is untouched.
    assert grid.cells[5, 11] == UNKNOWN


def test_min_passes_is_a_real_threshold():
    grid = _grid()
    passes = ray_pass_counts(grid, [_frame()], body_band=BAND, pixel_stride=1)
    assert (carve_free_space(grid, passes, min_passes=10_000).cells == UNKNOWN).all()
    assert (carve_free_space(grid, passes, min_passes=1).cells == FREE).sum() > \
        (carve_free_space(grid, passes, min_passes=3).cells == FREE).sum()


def test_rays_outside_the_body_band_carve_nothing():
    """A camera two metres up looking level never puts a ray through the band."""
    grid = _grid()
    passes = ray_pass_counts(grid, [_frame(height=2.0)], body_band=BAND, pixel_stride=1)
    assert passes.sum() == 0


def test_margin_stops_short_of_the_surface():
    grid = _grid()
    frame = _frame(depth=0.5)      # surface straight ahead at z = 0.55, row 11
    full = ray_pass_counts(grid, [frame], body_band=BAND, pixel_stride=1)
    short = ray_pass_counts(grid, [frame], body_band=BAND, pixel_stride=1, margin=0.2)
    assert full[10, 10] > 0
    assert short[10, 10] == 0, "the last 0.2 before the surface must not be carved"
    assert short[5, 10] == full[5, 10]


def test_the_scene_transform_moves_the_carve():
    grid = _grid()
    shift = np.eye(4)
    shift[0, 3] = 0.25                # x += 0.25: five cells to the right
    plain = ray_pass_counts(grid, [_frame()], body_band=BAND, pixel_stride=1)
    moved = ray_pass_counts(grid, [_frame()], body_band=BAND, pixel_stride=1,
                            scene_transform=shift)
    assert moved[5, 15] == plain[5, 10]
    assert moved[5, 5] == 0 or moved[5, 5] < plain[5, 5]


def test_pass_counts_must_match_the_grid():
    with pytest.raises(ValueError):
        carve_free_space(_grid(), np.zeros((3, 3), dtype=np.int64))


def test_free_fraction_reads_the_grid():
    grid = _grid()
    grid.cells[2, 2] = FREE
    points = np.array([[0.125, 0.125], [0.9, 0.9], [5.0, 5.0]])   # in, unknown, outside
    assert cells_free_fraction(grid, points) == pytest.approx(1 / 3)
    assert np.isnan(cells_free_fraction(grid, np.zeros((0, 2))))


# --- frame-local classification -------------------------------------------

def _ground_frame(scale: float = 1.0, size: int = 16) -> dict:
    """A camera at (0.5, 0.4, 0.05) over a flat ground with a wall 0.8 ahead.

    Rays that point down hit the ground plane y = 0; the others hit a wall at
    camera depth 0.8. ``scale`` multiplies every depth, which is exactly what
    a chunk with the wrong SIM3 scale does to a frame: the ground it sees
    moves up towards the camera, and every distance shrinks with it.
    """
    c2w = np.eye(4)
    c2w[:3, 3] = [0.5, 0.4, 0.05]
    f = size / 2
    intrinsics = np.array([[f, 0, f], [0, f, f], [0, 0, 1]], dtype=np.float64)
    r, c = np.mgrid[0:size, 0:size]
    dy = (r - f) / f          # cam y per unit depth; identity pose so +y is up
    depth = np.full((size, size), 0.8, dtype=np.float32)
    down = dy < -1e-6
    depth[down] = np.minimum(0.8, 0.4 / -dy[down])
    return {"depth": depth * scale, "intrinsics": intrinsics, "c2w": c2w}


def test_frame_local_grid_sees_ground_as_free_and_wall_as_occupied():
    template = _grid()
    grid, report = grid_from_depth_frames(
        template, [_ground_frame()], camera_height_m=0.4, body_band_m=BAND,
        pixel_stride=1, min_votes=1, min_passes=1, margin_m=0.05,
    )
    assert report["frames_used"] == 1
    assert report["metres_per_unit_p10_50_90"][1] == pytest.approx(1.0, abs=0.05)
    assert grid.cells[17, 10] == OCCUPIED      # the wall, z = 0.85, straight ahead
    # Ground straight ahead: the ray at dy = -0.5 lands 0.8 ahead, z = 0.85 —
    # that is the wall row; the one at dy = -1 lands 0.4 ahead, z = 0.45.
    assert grid.cells[9, 10] == FREE
    assert grid.cells[0, 10] == UNKNOWN        # behind the camera


def test_frame_local_grid_is_invariant_to_a_chunk_scale_error():
    """Halve every depth: the frame's ground rises to y = 0.2, its wall comes
    to 0.4 ahead. Against a global floor at 0 that ground is 0.2 up — inside
    the body band — and would read as an obstacle. Measured against the
    frame's own ground it is still the floor."""
    template = _grid()
    grid, report = grid_from_depth_frames(
        template, [_ground_frame(scale=0.5)], camera_height_m=0.4, body_band_m=BAND,
        pixel_stride=1, min_votes=1, min_passes=1, margin_m=0.05,
    )
    assert report["metres_per_unit_p10_50_90"][1] == pytest.approx(2.0, abs=0.1)
    assert grid.cells[9, 10] == OCCUPIED       # the wall, now at z = 0.45
    assert grid.cells[5, 10] == FREE           # ground 0.2 ahead, at y = 0.2
    assert (grid.cells == OCCUPIED).sum() < (grid.cells == FREE).sum()


def test_frame_local_grid_skips_frames_without_ground_below_the_camera():
    template = _grid()
    sky = _frame(height=2.0)                   # flat depth, every point above the ground
    # All points sit at the camera's own height: the frame's "ground" is the
    # camera, so it cannot be a ruler and is skipped.
    grid, report = grid_from_depth_frames(template, [sky], camera_height_m=0.4,
                                          body_band_m=BAND, pixel_stride=1)
    assert report["frames_used"] + report["frames_skipped"] == 1
    assert (grid.cells == UNKNOWN).all() or report["frames_used"] == 1


def test_frame_local_votes_are_per_frame_not_per_sample():
    """One frame with a dense obstacle is one vote; min_votes=2 needs a second
    frame to agree."""
    template = _grid()
    one, _ = grid_from_depth_frames(template, [_ground_frame()], camera_height_m=0.4,
                                    body_band_m=BAND, pixel_stride=1, min_votes=2, min_passes=2)
    two, _ = grid_from_depth_frames(template, [_ground_frame(), _ground_frame()],
                                    camera_height_m=0.4, body_band_m=BAND, pixel_stride=1,
                                    min_votes=2, min_passes=2)
    assert (one.cells == OCCUPIED).sum() == 0
    assert two.cells[17, 10] == OCCUPIED
