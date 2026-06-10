import math

import numpy as np
import pytest

from vaultwares_studio.robot_lab import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Episode,
    GridNavBackend,
    OccupancyGrid,
    geodesic_field,
    grid_from_points,
    make_gym_env,
    run_episode,
)
from vaultwares_studio.robot_lab.gridnav import FORWARD, STOP, TURN_LEFT


def synthetic_room_points(count: int = 30_000, seed: int = 3) -> np.ndarray:
    """A 2x2 'room': dense floor at y=0, a wall slab across the middle."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack(
        [rng.uniform(0, 2, count), rng.normal(0.02, 0.01, count), rng.uniform(0, 2, count)]
    )
    wall_count = count // 4
    wall = np.column_stack(
        [
            rng.uniform(0.9, 1.1, wall_count),
            rng.uniform(0.15, 0.55, wall_count),
            rng.uniform(0.0, 1.4, wall_count),  # gap at z in (1.4, 2.0)
        ]
    )
    return np.vstack([floor, wall])


@pytest.fixture(scope="module")
def room_grid() -> OccupancyGrid:
    return grid_from_points(synthetic_room_points(), cell_size=0.05)


def test_grid_classifies_floor_wall_unknown(room_grid):
    cells = room_grid.cells
    assert (cells == FREE).sum() > 0.4 * cells.size
    wall_row, wall_col = room_grid.world_to_cell(1.0, 0.7)
    assert cells[wall_row, wall_col] == OCCUPIED
    outside = cells[0, 0]
    assert outside in (UNKNOWN, FREE, OCCUPIED)  # boundary cell, any is legal
    # The wall splits x<0.9 from x>1.1 except through the gap.
    field = geodesic_field(room_grid, *room_grid.world_to_cell(1.8, 0.5))
    left_distance = field[room_grid.world_to_cell(0.3, 0.5)]
    assert np.isfinite(left_distance)  # reachable, but only around the gap
    direct_cells = (1.5 / room_grid.cell_size)
    assert left_distance > direct_cells  # forced detour


def test_grid_round_trip(tmp_path, room_grid):
    path = tmp_path / "occupancy.npz"
    room_grid.save(path)
    loaded = OccupancyGrid.load(path)
    assert np.array_equal(loaded.cells, room_grid.cells)
    assert loaded.cell_size == room_grid.cell_size


def test_episode_sampling_reachable(room_grid):
    backend = GridNavBackend(room_grid)
    episodes = backend.sample_episodes(5, seed=11)
    assert len(episodes) == 5
    for episode in episodes:
        assert room_grid.is_free_world(*episode.start)
        assert room_grid.is_free_world(*episode.goal)


def test_scripted_navigation_reaches_goal(room_grid):
    backend = GridNavBackend(room_grid)
    start = room_grid.cell_to_world(*room_grid.world_to_cell(0.3, 1.7))
    goal = room_grid.cell_to_world(*room_grid.world_to_cell(0.65, 1.7))
    episode = Episode(start=list(start), goal=list(goal), start_heading=0.0)

    def greedy_policy(observation) -> int:
        distance, sin_a, cos_a = observation["goal"]
        if distance <= 0.12:
            return STOP
        if abs(math.atan2(sin_a, cos_a)) > 0.3:
            return TURN_LEFT
        return FORWARD

    record = run_episode(backend, episode, greedy_policy)
    assert record.success, f"failed: trajectory tail {record.trajectory[-3:]}"
    assert 0.0 < record.spl <= 1.0
    assert len(record.trajectory) == record.steps + 1


def test_gym_env_contract(room_grid):
    env = make_gym_env(room_grid, seed=5)
    observation, _info = env.reset(seed=5)
    assert observation.shape == env.observation_space.shape
    observation, reward, terminated, truncated, info = env.step(FORWARD)
    assert observation.dtype == np.float32
    assert isinstance(reward, float)
    assert truncated is False
    assert "success" in info
