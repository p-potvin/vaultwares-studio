"""2.5D occupancy grid from a reconstructed scene's point cloud.

Until the sim_export mesh stage lands, the splat preview cloud is a workable
source: estimate the floor height, then mark cells occupied where points sit
in the robot's body band above the floor. Cells with no points at all are
unknown (treated as untraversable).

Grid encoding (uint8): 0 = free, 1 = occupied, 2 = unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 1, 2


@dataclass
class OccupancyGrid:
    cells: np.ndarray  # (rows, cols) uint8 — rows index z, cols index x
    origin: tuple[float, float]  # world (x, z) of cell [0, 0]'s corner
    cell_size: float
    floor_y: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.cells.shape

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        col = int((x - self.origin[0]) / self.cell_size)
        row = int((z - self.origin[1]) / self.cell_size)
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin[0] + (col + 0.5) * self.cell_size,
            self.origin[1] + (row + 0.5) * self.cell_size,
        )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.cells.shape[0] and 0 <= col < self.cells.shape[1]

    def is_free_world(self, x: float, z: float) -> bool:
        row, col = self.world_to_cell(x, z)
        return self.in_bounds(row, col) and self.cells[row, col] == FREE

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            cells=self.cells,
            origin=np.array(self.origin),
            cell_size=self.cell_size,
            floor_y=self.floor_y,
        )

    @classmethod
    def load(cls, path: Path) -> "OccupancyGrid":
        data = np.load(path)
        return cls(
            cells=data["cells"],
            origin=(float(data["origin"][0]), float(data["origin"][1])),
            cell_size=float(data["cell_size"]),
            floor_y=float(data["floor_y"]),
        )


def grid_from_points(
    points: np.ndarray,
    cell_size: float = 0.05,
    body_band: tuple[float, float] = (0.10, 0.60),
    min_support: int = 2,
) -> OccupancyGrid:
    """Build the grid from (N, 3) points in the scene's (x, y-up, z) frame.

    body_band is the height window above the estimated floor that the robot's
    body sweeps: points there are obstacles; points only below it are floor.
    Heights are in scene units (nerfstudio-normalized scenes are roughly
    unit-scale; tune body_band per scene if needed).
    """
    floor_y = float(np.percentile(points[:, 1], 8))
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = np.percentile(xs, [1, 99])
    z_min, z_max = np.percentile(zs, [1, 99])
    cols = max(8, int(np.ceil((x_max - x_min) / cell_size)))
    rows = max(8, int(np.ceil((z_max - z_min) / cell_size)))

    col_index = np.clip(((xs - x_min) / cell_size).astype(int), 0, cols - 1)
    row_index = np.clip(((zs - z_min) / cell_size).astype(int), 0, rows - 1)
    flat = row_index * cols + col_index

    height = ys - floor_y
    in_band = (height >= body_band[0]) & (height <= body_band[1])
    below_band = height < body_band[0]

    obstacle_counts = np.bincount(flat[in_band], minlength=rows * cols).reshape(rows, cols)
    support_counts = np.bincount(flat[below_band], minlength=rows * cols).reshape(rows, cols)

    cells = np.full((rows, cols), UNKNOWN, dtype=np.uint8)
    cells[support_counts >= min_support] = FREE
    cells[obstacle_counts >= min_support] = OCCUPIED
    return OccupancyGrid(cells=cells, origin=(float(x_min), float(z_min)), cell_size=cell_size, floor_y=floor_y)


def grid_from_preview_ply(preview_ply: Path, **kwargs) -> OccupancyGrid:
    from plyfile import PlyData

    vertex = PlyData.read(str(preview_ply))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    return grid_from_points(points, **kwargs)


def geodesic_field(grid: OccupancyGrid, goal_row: int, goal_col: int) -> np.ndarray:
    """BFS distance (in cells) from every free cell to the goal; inf elsewhere."""
    from collections import deque

    rows, cols = grid.shape
    distances = np.full((rows, cols), np.inf)
    if not grid.in_bounds(goal_row, goal_col) or grid.cells[goal_row, goal_col] != FREE:
        return distances
    distances[goal_row, goal_col] = 0.0
    queue = deque([(goal_row, goal_col)])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if (
                grid.in_bounds(nr, nc)
                and grid.cells[nr, nc] == FREE
                and distances[nr, nc] == np.inf
            ):
                distances[nr, nc] = distances[row, col] + 1.0
                queue.append((nr, nc))
    return distances
