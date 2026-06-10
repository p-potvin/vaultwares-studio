"""Grid-nav backend: PointNav over the occupancy grid, native and free.

A gymnasium Env wrapping the NavSimBackend so stable-baselines3 PPO trains
directly on it. Observations follow the standard PointNav recipe: goal
vector (distance + heading-relative angle) plus a 16-ray depth scan. Reward
is geodesic-progress shaping with a success bonus and per-step slack.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Episode, EpisodeRecord, NavSimBackend
from .occupancy import FREE, OccupancyGrid, geodesic_field

FORWARD, TURN_LEFT, TURN_RIGHT, STOP = 0, 1, 2, 3
N_RAYS = 16
MAX_RAY = 2.0  # scene units
FORWARD_STEP = 0.08
TURN_STEP = math.radians(15)
SUCCESS_RADIUS = 0.15
MAX_STEPS = 300


class GridNavBackend(NavSimBackend):
    name = "gridnav"

    def __init__(self, grid: OccupancyGrid) -> None:
        self.grid = grid
        self._pose = (0.0, 0.0, 0.0)
        self._episode: Episode | None = None
        self._geodesic: np.ndarray | None = None
        self._steps = 0
        self._shortest_start: float = 1.0
        self._path_length = 0.0

    # -- episodes ---------------------------------------------------------------

    def sample_episodes(self, count: int, seed: int | None = None) -> list[Episode]:
        rng = np.random.default_rng(seed)
        free_rows, free_cols = np.nonzero(self.grid.cells == FREE)
        episodes: list[Episode] = []
        attempts = 0
        while len(episodes) < count and attempts < count * 200:
            attempts += 1
            si, gi = rng.integers(0, len(free_rows), size=2)
            field = geodesic_field(self.grid, int(free_rows[gi]), int(free_cols[gi]))
            distance = field[free_rows[si], free_cols[si]]
            if not np.isfinite(distance) or distance < 6:
                continue  # unreachable or trivially close
            start = self.grid.cell_to_world(int(free_rows[si]), int(free_cols[si]))
            goal = self.grid.cell_to_world(int(free_rows[gi]), int(free_cols[gi]))
            episodes.append(
                Episode(start=list(start), goal=list(goal), start_heading=float(rng.uniform(0, 2 * math.pi)))
            )
        return episodes

    # -- simulation ---------------------------------------------------------------

    def reset(self, episode: Episode) -> dict:
        self._episode = episode
        self._pose = (episode.start[0], episode.start[1], episode.start_heading)
        goal_row, goal_col = self.grid.world_to_cell(episode.goal[0], episode.goal[1])
        self._geodesic = geodesic_field(self.grid, goal_row, goal_col)
        self._steps = 0
        self._path_length = 0.0
        self._shortest_start = max(self._geodesic_at(*self._pose[:2]) * self.grid.cell_size, 1e-6)
        return self._observation()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        assert self._episode is not None, "call reset() first"
        x, z, heading = self._pose
        previous_distance = self._geodesic_at(x, z)
        collided = False

        if action == FORWARD:
            nx = x + FORWARD_STEP * math.cos(heading)
            nz = z + FORWARD_STEP * math.sin(heading)
            if self.grid.is_free_world(nx, nz):
                self._path_length += FORWARD_STEP
                x, z = nx, nz
            else:
                collided = True
        elif action == TURN_LEFT:
            heading = (heading + TURN_STEP) % (2 * math.pi)
        elif action == TURN_RIGHT:
            heading = (heading - TURN_STEP) % (2 * math.pi)

        self._pose = (x, z, heading)
        self._steps += 1

        euclid_goal = math.hypot(self._episode.goal[0] - x, self._episode.goal[1] - z)
        done = False
        success = False
        if action == STOP:
            done = True
            success = euclid_goal <= SUCCESS_RADIUS
        elif self._steps >= MAX_STEPS:
            done = True

        progress = (previous_distance - self._geodesic_at(x, z)) * self.grid.cell_size
        reward = progress - 0.005 - (0.05 if collided else 0.0)
        if done and success:
            reward += 2.5

        info = {"success": success, "collided": collided, "distance_to_goal": euclid_goal}
        return self._observation(), float(reward), done, info

    def agent_pose(self) -> tuple[float, float, float]:
        return self._pose

    def spl(self, success: bool) -> float:
        if not success:
            return 0.0
        return self._shortest_start / max(self._path_length, self._shortest_start)

    # -- internals ----------------------------------------------------------------

    def _geodesic_at(self, x: float, z: float) -> float:
        row, col = self.grid.world_to_cell(x, z)
        if not self.grid.in_bounds(row, col):
            return float(self.grid.shape[0] + self.grid.shape[1])
        value = self._geodesic[row, col]
        return float(value) if np.isfinite(value) else float(self.grid.shape[0] + self.grid.shape[1])

    def _ray_distances(self) -> np.ndarray:
        x, z, heading = self._pose
        distances = np.empty(N_RAYS, dtype=np.float32)
        step = self.grid.cell_size * 0.9
        for ray in range(N_RAYS):
            angle = heading + 2 * math.pi * ray / N_RAYS
            dx, dz = math.cos(angle) * step, math.sin(angle) * step
            distance = 0.0
            px, pz = x, z
            while distance < MAX_RAY:
                px += dx
                pz += dz
                distance += step
                if not self.grid.is_free_world(px, pz):
                    break
            distances[ray] = distance
        return distances

    def _observation(self) -> dict:
        x, z, heading = self._pose
        gx, gz = self._episode.goal
        distance = math.hypot(gx - x, gz - z)
        angle = math.atan2(gz - z, gx - x) - heading
        return {
            "goal": np.array([distance, math.sin(angle), math.cos(angle)], dtype=np.float32),
            "rays": self._ray_distances(),
        }


def make_gym_env(grid: OccupancyGrid, seed: int = 0):
    """Gymnasium wrapper: flat Box observation, episodes resampled per reset."""
    import gymnasium as gym
    from gymnasium import spaces

    class GridNavEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.backend = GridNavBackend(grid)
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3 + N_RAYS,), dtype=np.float32
            )
            self._rng_seed = seed
            self._episode_count = 0

        def _flatten(self, observation: dict) -> np.ndarray:
            return np.concatenate([observation["goal"], observation["rays"]]).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._episode_count += 1
            episodes = self.backend.sample_episodes(1, seed=(seed or self._rng_seed) + self._episode_count)
            if not episodes:
                raise RuntimeError("No reachable episodes in this occupancy grid.")
            observation = self.backend.reset(episodes[0])
            return self._flatten(observation), {}

        def step(self, action):
            observation, reward, done, info = self.backend.step(int(action))
            return self._flatten(observation), reward, done, False, info

    return GridNavEnv()


def run_episode(backend: GridNavBackend, episode: Episode, policy) -> EpisodeRecord:
    """Roll one episode with policy(obs)->action; records the trajectory."""
    observation = backend.reset(episode)
    record = EpisodeRecord(episode=episode)
    record.trajectory.append(list(backend.agent_pose()))
    for _ in range(MAX_STEPS + 1):
        action = policy(observation)
        observation, _reward, done, info = backend.step(action)
        record.trajectory.append(list(backend.agent_pose()))
        if done:
            record.success = bool(info["success"])
            break
    record.steps = len(record.trajectory) - 1
    record.spl = backend.spl(record.success)
    return record
