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


def grid_from_level_set(
    nvdb_path: Path,
    *,
    surface_band: float | None = None,
    scene_transform: np.ndarray | None = None,
    **kwargs,
) -> OccupancyGrid:
    """Build the grid from a NanoVDB level set instead of the splat preview.

    This *feeds* the existing point-based path rather than replacing it: the
    voxels near the zero crossing are the surface, and their centres are handed
    to ``grid_from_points`` unchanged. Everything about floor estimation, the
    body band and the support threshold stays where it is and keeps working the
    same way.

    Why that is worth doing at all, given the preview cloud already exists: the
    splat preview is 200k gaussians subsampled for display, weighted towards
    whatever the renderer found interesting. The level set is every voxel a
    depth ray actually crossed, at a known spacing, with a known distance to the
    surface — so ``surface_band`` selects "within N metres of a real surface"
    rather than "a point happened to land here".

    ``scene_transform`` must be ``camera_scene.scene_frame_transform(job_dir)``
    — trainer normalisation then the gravity rotation, the same chain
    ``depth_fusion`` applies to the mesh. The grid is fused in DA3 world
    coordinates where "up" is wherever the phone was pointing, and the body band
    is measured against a floor, so without it the band cuts the scene at an
    angle. Passing the gravity rotation alone is not enough and lands the floor
    metres away: the trainer's normalisation carries a scale.

    ``surface_band`` defaults to one voxel, which is the thinnest shell that is
    still closed. Widening it thickens obstacles; narrowing it below a voxel
    starts punching holes that the geodesic field will happily route through.

    What this does NOT use is the free space, and it cannot: a TSDF only writes
    voxels within the truncation band of a surface, so "empty and far from
    anything" is not stored. That evidence is in the depth rays, and
    ``ray_pass_counts`` + ``carve_free_space`` recover it from them.
    """
    from ..nanovdb_read import read

    grids = read(nvdb_path)
    if not grids:
        raise ValueError(f"{nvdb_path} holds no grids")
    grid = grids[0]
    ijk, values = grid.active_voxels()
    if not len(ijk):
        raise ValueError(f"{nvdb_path} has no active voxels")

    band = surface_band if surface_band is not None else float(grid.voxel_size[0])
    near = np.abs(values) <= band
    if not near.any():
        raise ValueError(
            f"no voxel within {band} of the surface; the band is narrower than "
            f"the voxel size {grid.voxel_size[0]}"
        )
    points = grid.index_to_world(ijk[near])
    if scene_transform is not None:
        transform = np.asarray(scene_transform, dtype=np.float64)
        points = points @ transform[:3, :3].T + transform[:3, 3]
    return grid_from_points(points, **kwargs)


def ray_pass_counts(
    grid: OccupancyGrid,
    frames,
    *,
    body_band: tuple[float, float] = (0.10, 0.60),
    scene_transform: np.ndarray | None = None,
    pixel_stride: int = 4,
    depth_trunc: float = 1e9,
    margin: float = 0.0,
    max_steps: int = 128,
) -> np.ndarray:
    """How many depth rays crossed each cell's body band on the way to a surface.

    This is the free-space evidence the level set does not hold. A TSDF only
    ever writes voxels within the truncation band of a surface, so "empty" far
    from any surface is not stored anywhere — the grid's UNKNOWN cells are
    exactly the cells no surface came near. What does say those cells are empty
    is the ray: every valid depth sample is a segment from the camera to the
    surface that passed through nothing, and the part of that segment inside
    the body band is a body-height observation of empty space.

    ``frames`` is what ``tsdf_volume.load_streaming_frames`` yields: ``depth``
    in world units, ``intrinsics`` for that resolution, ``c2w`` OpenCV camera
    to world. ``scene_transform`` is the same matrix ``grid_from_level_set``
    took, and must be, or the rays are carved through the wrong place; it
    carries the trainer's scale, which is why ``margin`` (in scene units) is
    not simply the TSDF truncation.

    Each ray is sampled at half a cell along the part of it that lies inside
    the band, and counted at most once per cell, so a ray grazing along a cell
    boundary does not count for more than a ray crossing it cleanly.
    """
    rows, cols = grid.shape
    counts = np.zeros(rows * cols, dtype=np.int64)
    lo = grid.floor_y + body_band[0]
    hi = grid.floor_y + body_band[1]
    spacing = grid.cell_size * 0.5
    if scene_transform is not None:
        transform = np.asarray(scene_transform, dtype=np.float64)
    else:
        transform = np.eye(4)

    for frame in frames:
        placed = _frame_in_scene(frame, transform, pixel_stride, depth_trunc)
        if placed is None:
            continue
        surface, centre = placed
        counts += _ray_cells(grid, centre, surface, lo, hi, margin=margin, spacing=spacing,
                             max_steps=max_steps)

    return counts.reshape(rows, cols)


def _frame_in_scene(frame, transform: np.ndarray, pixel_stride: int, depth_trunc: float,
                    conf_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Unproject one depth map into the scene frame: ``(surface (N, 3), centre (3,))``."""
    depth = np.asarray(frame["depth"])[::pixel_stride, ::pixel_stride]
    valid = np.isfinite(depth) & (depth > 0) & (depth < depth_trunc)
    conf = frame.get("conf") if isinstance(frame, dict) else None
    if conf is not None and conf_floor > 0:
        valid &= np.asarray(conf)[::pixel_stride, ::pixel_stride] > conf_floor
    if not valid.any():
        return None
    r, c = np.nonzero(valid)
    z = depth[r, c].astype(np.float64)
    intr = np.asarray(frame["intrinsics"], dtype=np.float64)
    cam = np.stack([
        (c * pixel_stride - intr[0, 2]) / intr[0, 0] * z,
        (r * pixel_stride - intr[1, 2]) / intr[1, 1] * z,
        z,
    ], axis=1)
    c2w = np.asarray(frame["c2w"], dtype=np.float64)
    a, b = transform[:3, :3], transform[:3, 3]
    surface = (cam @ c2w[:3, :3].T + c2w[:3, 3]) @ a.T + b
    centre = c2w[:3, 3] @ a.T + b
    return surface, centre


def _ray_cells(grid: OccupancyGrid, centre: np.ndarray, surface: np.ndarray,
               lo: float, hi: float, *, margin: float, spacing: float,
               max_steps: int) -> np.ndarray:
    """Flat per-cell count of rays from ``centre`` to each ``surface`` point
    whose segment crosses the height slab ``[lo, hi]`` inside that cell. One
    count per ray per cell, whatever the sampling."""
    rows, cols = grid.shape
    counts = np.zeros(rows * cols, dtype=np.int64)
    d = surface - centre
    length = np.linalg.norm(d, axis=1)
    ok = length > 1e-9
    d, length = d[ok], length[ok]
    if not len(d):
        return counts
    t_max = np.clip(1.0 - margin / length, 0.0, 1.0)

    # Where along the ray is the slab? y(t) is linear, so it is an interval —
    # the whole ray when the ray is level and inside, nothing when level and
    # outside.
    dy = d[:, 1]
    level = np.abs(dy) < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t_a = (lo - centre[1]) / dy
        t_b = (hi - centre[1]) / dy
    t0 = np.where(level, 0.0, np.minimum(t_a, t_b))
    t1 = np.where(level, np.where((centre[1] >= lo) & (centre[1] <= hi), 1.0, -1.0),
                  np.maximum(t_a, t_b))
    t0 = np.maximum(t0, 0.0)
    t1 = np.minimum(t1, t_max)
    crosses = t1 > t0
    if not crosses.any():
        return counts
    d, length, t0, t1 = d[crosses], length[crosses], t0[crosses], t1[crosses]

    steps = int(np.ceil((length * (t1 - t0)).max() / spacing)) + 1
    steps = int(min(max(steps, 2), max_steps))
    fractions = np.linspace(0.0, 1.0, steps)
    t = t0[:, None] + (t1 - t0)[:, None] * fractions[None, :]
    points = centre[None, None, :] + d[:, None, :] * t[:, :, None]
    col = np.floor((points[:, :, 0] - grid.origin[0]) / grid.cell_size).astype(np.int64)
    row = np.floor((points[:, :, 2] - grid.origin[1]) / grid.cell_size).astype(np.int64)
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    ray_id = np.broadcast_to(np.arange(len(d))[:, None], row.shape)
    flat = row * cols + col
    pairs = np.unique((ray_id[inside].astype(np.int64) * (rows * cols)) + flat[inside])
    counts += np.bincount(pairs % (rows * cols), minlength=rows * cols)
    return counts


def _cell_index(grid: OccupancyGrid, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat cell index for (N, 3) scene points, and the mask of those in bounds."""
    rows, cols = grid.shape
    col = np.floor((points[:, 0] - grid.origin[0]) / grid.cell_size).astype(np.int64)
    row = np.floor((points[:, 2] - grid.origin[1]) / grid.cell_size).astype(np.int64)
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return (row * cols + col), inside


def _ground_plane(surface: np.ndarray, centre: np.ndarray,
                  low_percentile: float = 5.0) -> tuple[float, float, float] | None:
    """Least-squares plane ``y = a_x * x + a_z * z + c`` through a frame's ground.

    Why a plane and not a height: DA3's ground is not flat relative to the
    camera. Measured on the 13 Sep backyard, the ground a frame sees curves
    and tilts by 10-20 cm over three metres — enough that, against a single
    reference height, the far ground sits inside a body band that starts at
    10 cm and every frame votes the walked path an obstacle. A plane through
    the low points absorbs the tilt; what it leaves is the curvature and the
    real objects.

    The low points are those in the bottom slice by height relative to the
    lowest percentile — a sixth of the camera's height above it, about 23 cm
    on a handheld capture, which is the ground plus its own noise and little
    else. A third was tried first and let the base of every wall in, tilting
    the plane up towards it. One reweighting pass drops the outliers a first
    fit let in. Returns None when there is not enough ground to fit, or the
    camera is not above the result.
    """
    y = surface[:, 1]
    p_low = float(np.percentile(y, low_percentile))
    above = centre[1] - p_low
    if above <= 1e-9:
        return None
    low = y < p_low + above / 6.0
    if low.sum() < 16:
        return None

    def fit(mask: np.ndarray) -> np.ndarray:
        design = np.stack([surface[mask, 0], surface[mask, 2], np.ones(int(mask.sum()))], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        return coefficients

    coefficients = fit(low)
    residual = y - (surface[:, 0] * coefficients[0] + surface[:, 2] * coefficients[1] + coefficients[2])
    scale = 1.4826 * float(np.median(np.abs(residual[low] - np.median(residual[low])))) + 1e-12
    inlier = low & (np.abs(residual) < 2.5 * scale)
    if inlier.sum() >= 16:
        coefficients = fit(inlier)
    # Refuse a plane the camera is not above, or one tilted past 30 degrees:
    # both are a frame that saw no ground, not a floor.
    tilt = np.hypot(coefficients[0], coefficients[1])
    if tilt > np.tan(np.radians(30)):
        return None
    if centre[1] - (coefficients[0] * centre[0] + coefficients[1] * centre[2] + coefficients[2]) <= 1e-9:
        return None
    return float(coefficients[0]), float(coefficients[1]), float(coefficients[2])


def grid_from_depth_frames(
    template: OccupancyGrid,
    frames,
    *,
    scene_transform: np.ndarray | None = None,
    camera_height_m: float = 1.4,
    body_band_m: tuple[float, float] = (0.10, 0.60),
    margin_m: float = 0.2,
    pixel_stride: int = 4,
    depth_trunc: float = 1e9,
    conf_floor: float = 0.0,
    ground_percentile: float = 5.0,
    min_samples: int = 2,
    min_votes: int = 2,
    min_passes: int = 3,
    obstacle_fraction: float = 0.5,
    max_steps: int = 128,
) -> tuple[OccupancyGrid, dict]:
    """Occupancy voted frame by frame, each frame measured against its own ground.

    Why this exists, after ``grid_from_level_set``: the fused volume from a
    chunked SfM is only as vertically consistent as the chunks. On the 13 Sep
    backyard the camera's height above the ground *it sees in the same frame*
    steps between 0.7 and 1.9 apparent metres from one 45-frame block to the
    next, while the person's arm did no such thing. The level set inherits that
    — the ground near the walked path is smeared over nearly a metre — and no
    single floor plane and body band can be laid across it: 72% of the camera
    positions came out OCCUPIED.

    A single frame does not have that problem. Its depth map and its pose come
    from the same chunk, so within the frame the ground is where the ground
    is, and the camera's height above it is a ruler: ``camera_height_m`` is
    what the phone was held at, and the frame's own camera-to-ground distance
    says how many scene units that is *for this frame*. The ground is a plane
    fitted to the frame's low points (``_ground_plane``), because a single
    reference height leaves DA3's tilted, curving ground inside the band.
    Heights are then in metres above that plane, the body band is applied there, and
    the frame votes once per cell — obstacle where it saw at least
    ``min_samples`` body-height points, support where it saw ground, and a ray
    pass where a depth ray crossed the band. A cell is OCCUPIED when at least
    ``obstacle_fraction`` of the frames that saw it voted obstacle. Cell placement in x, z still uses
    the global poses; that drift is the trajectory's and is what a grid is for.

    ``template`` gives the extent, cell size and origin (the level-set grid is
    the natural source: the extent is what the volume covers). Its floor_y is
    kept as a label; the classification does not use it.

    Returns the grid and a report with the frames used and the per-frame scale
    spread, which is the number that says whether this was needed.
    """
    transform = np.eye(4) if scene_transform is None else np.asarray(scene_transform, dtype=np.float64)
    rows, cols = template.shape
    size = rows * cols
    obstacle_votes = np.zeros(size, dtype=np.int64)
    support_votes = np.zeros(size, dtype=np.int64)
    passes = np.zeros(size, dtype=np.int64)
    used, skipped, scales = 0, 0, []
    spacing = template.cell_size * 0.5

    for frame in frames:
        placed = _frame_in_scene(frame, transform, pixel_stride, depth_trunc, conf_floor)
        if placed is None:
            skipped += 1
            continue
        surface, centre = placed
        plane = _ground_plane(surface, centre, ground_percentile)
        if plane is None:
            skipped += 1
            continue
        # Shear the frame so its ground plane is y = 0. A shear is affine, so
        # rays stay straight and the carve below can keep using a flat slab.
        # Cells are indexed from the unsheared x, z, which the shear leaves
        # alone.
        a_x, a_z, c = plane
        rectified = surface.copy()
        rectified[:, 1] -= a_x * surface[:, 0] + a_z * surface[:, 2] + c
        centre_r = centre.copy()
        centre_r[1] -= a_x * centre[0] + a_z * centre[2] + c
        above = centre_r[1]
        if above <= 1e-9:
            skipped += 1
            continue
        metres_per_unit = camera_height_m / above
        scales.append(metres_per_unit)
        height_m = rectified[:, 1] * metres_per_unit
        lo = body_band_m[0] / metres_per_unit
        hi = body_band_m[1] / metres_per_unit

        flat, inside = _cell_index(template, surface)
        in_band = inside & (height_m >= body_band_m[0]) & (height_m <= body_band_m[1])
        # Support is ground: below the band but not far below it, since a
        # point a metre under the ground is a depth error, not a floor.
        on_ground = inside & (height_m < body_band_m[0]) & (height_m > -body_band_m[0])
        obstacle_votes += np.bincount(flat[in_band], minlength=size) >= min_samples
        support_votes += np.bincount(flat[on_ground], minlength=size) >= min_samples
        passes += _ray_cells(template, centre_r, rectified, lo, hi,
                             margin=margin_m / metres_per_unit, spacing=spacing,
                             max_steps=max_steps)
        used += 1

    # An obstacle is what most of the frames that looked at a cell saw there.
    # Frames vote support or obstacle independently, so a real wall collects
    # obstacle votes from every frame that sees the cell, while a depth error
    # is one frame's opinion against the others' ground. Without this,
    # ``min_votes`` alone lets any two noisy frames close a cell for good.
    seen = obstacle_votes + support_votes
    cells = np.full(size, UNKNOWN, dtype=np.uint8)
    cells[(support_votes >= min_votes) | (passes >= min_passes)] = FREE
    cells[(obstacle_votes >= min_votes) & (obstacle_votes >= obstacle_fraction * seen)] = OCCUPIED
    grid = OccupancyGrid(cells=cells.reshape(rows, cols), origin=template.origin,
                         cell_size=template.cell_size, floor_y=template.floor_y)
    report = {
        "frames_used": used, "frames_skipped": skipped,
        "metres_per_unit_p10_50_90": ([round(float(v), 3) for v in np.percentile(scales, [10, 50, 90])]
                                      if scales else None),
        "obstacle_votes": obstacle_votes.reshape(rows, cols),
        "support_votes": support_votes.reshape(rows, cols),
        "passes": passes.reshape(rows, cols),
    }
    return grid, report




def carve_free_space(
    grid: OccupancyGrid,
    passes: np.ndarray,
    *,
    min_passes: int = 3,
) -> OccupancyGrid:
    """Promote UNKNOWN cells that enough rays crossed to FREE.

    OCCUPIED is never touched: a surface in the band is direct evidence, a ray
    passing is indirect, and a depth map's noise puts rays through walls far
    more often than it puts walls in empty space. A cell already FREE stays
    FREE. So the only transition is UNKNOWN -> FREE, and only when
    ``min_passes`` independent rays agree — one ray is one noisy pixel.

    This does not require floor support under the cell. It is the deliberate
    trade: the alternative leaves every cell whose floor the capture never
    looked at UNKNOWN, which on a walking capture is most of the walked path,
    because a phone held at chest height sees the ground ahead and not the
    ground underfoot.
    """
    if passes.shape != grid.cells.shape:
        raise ValueError(f"pass counts {passes.shape} do not match the grid {grid.cells.shape}")
    cells = grid.cells.copy()
    promote = (cells == UNKNOWN) & (passes >= min_passes)
    cells[promote] = FREE
    return OccupancyGrid(cells=cells, origin=grid.origin, cell_size=grid.cell_size,
                         floor_y=grid.floor_y)


def cells_free_fraction(grid: OccupancyGrid, points_xz: np.ndarray) -> float:
    """Fraction of (N, 2) world (x, z) positions that fall in FREE cells.

    Handed the camera trajectory this is the one number that checks free space
    against something that is known independently to have been free: the
    person walked there.
    """
    if not len(points_xz):
        return float("nan")
    hits = sum(grid.is_free_world(float(x), float(z)) for x, z in points_xz)
    return hits / len(points_xz)


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
