<!-- v1.0.0 -->
# Free space for the occupancy grid — Wed, 17 Sep 2026

The 17 Sep handoff named the next increment: the level set knows "empty"
from "never observed", so use it to fill the grid's UNKNOWN cells. This
session did that, and the measurement that came with it changed what the
grid is built from. Read the numbers before the code.

## The measurement: does the walked path read as free?

The person walked the camera trajectory, so every camera position is a
cell that is known to be free without consulting any grid. The fraction of
camera positions landing in FREE cells is therefore the one discriminative
check of free space this capture offers. `tools/occupancy_from_volume.py`
prints it for every grid it builds.

| grid, 13 Sep backyard, 25 cm cells, band 0.10–0.60 m | free | occupied | unknown | **cameras in FREE** |
|---|---|---|---|---|
| splat preview (incumbent) | 12.2% | 44.2% | 43.6% | 0.21 |
| level set, global floor | 14.1% | 33.9% | 52.0% | 0.22 |
| level set + ray carve | 15.7% | 33.9% | 50.4% | 0.24 |
| frame-local, constant ground | 21.7% | 31.5% | 46.9% | 0.35 |
| frame-local, ground plane | 26.6% | 27.0% | 46.4% | 0.48 |
| **frame-local, plane + majority** | **37.9%** | **15.4%** | **46.7%** | **0.79** |

The first three rows are the handoff's plan, executed. They put 72–79% of
the walked path in OCCUPIED or UNKNOWN. The ray carve, the thing the
handoff asked for, promoted 81 cells: the carve was correct and beside the
point, because the cells it could reach were not the problem.

## Why the level set cannot be the source, on this capture

Three measurements, in the order they were made:

**The band was in the wrong units.** The scene is not metric. The median
camera sits 0.165 scene units above the estimated floor; a phone held at
1.4 m makes that 8.5 m per unit, so the band `(0.10, 0.60)` was 0.85–5 m
above the ground and swept in every wall and tree. Cross-check: the path
length works out to ~67 m over 134 s, a walking pace. The tool now takes
the band and cell in metres and derives the ruler from the cameras.

**The volume is not vertically self-consistent.** With the band in metres,
the path still read 72% OCCUPIED. Unprojecting each frame's own depth map
and measuring the camera's height above *the ground it sees* — a quantity
the person's arm keeps nearly constant — gives, per 45-frame block:

```
frames   0- 44: 1.36 m      frames 270-314: 1.51 m
frames  90-134: 0.82 m      frames 315-359: 1.91 m
frames 135-179: 0.75 m      frames 360-404: 0.79 m
frames 225-269: 1.31 m      frames 450-539: 0.71-0.76 m
```

That is chunk scale drift between DA3-Streaming's SIM3-aligned windows,
inherited by every voxel the TSDF fused from them. Near the walked path the
"ground" is smeared over nearly a metre of height (voxels within 0.4 m of a
camera: 3.9k at 0–0.1 m, 3.3k at 0.1–0.2, 3.4k at 0.2–0.4, 6.3k at
0.4–0.6). No single floor plane and body band can be laid across that, and a
local floor per cell only cleared 66 of 358 path cells.

**Within one frame, the ground tilts.** Even against the frame's own ground
reference, the far ground sits 10–20 cm up over three metres — DA3's depth
is affine-accurate, not flat — and lands inside a band that starts at 10 cm.
The in-band samples on the path were high-confidence points 1–3 m ahead of
*other* frames, not noise and not the walker's feet.

## What was built

`vaultwares_studio/robot_lab/occupancy.py`:

- `ray_pass_counts` + `carve_free_space`: the increment the handoff asked
  for. Rays from each camera to each depth sample, the part inside the band
  rasterised at half a cell, counted once per ray per cell; UNKNOWN → FREE
  at `min_passes`, OCCUPIED never touched. Correct, tested, and kept —
  it is what the frame-local grid uses for its free-space evidence.
- `grid_from_depth_frames`: each frame measured against a plane fitted to
  its own low points (`_ground_plane`: bottom sixth of the camera height,
  one robust reweighting; a third let wall bases in and tilted the plane).
  The camera's height above that plane is the frame's ruler, so chunk scale
  drift cancels frame by frame. Each frame votes once per cell: obstacle,
  support, ray pass. A cell is OCCUPIED only when at least
  `obstacle_fraction` (0.5) of the frames that saw it voted obstacle — a
  wall is what most observers agree on, a depth error is one frame's
  opinion. That rule alone took the path from 0.48 to 0.79.
- The level-set path (`grid_from_level_set`) is unchanged and still the
  extent source: the frame-local grid takes its origin, cell size and shape
  from it.

`tools/occupancy_from_volume.py` builds all of them, prints the table above
as JSON, and writes `reconstruction/occupancy.{npz,png,json}` (frame-local)
plus `occupancy_levelset.*` and the per-cell votes. 3 s for 500 frames at
pixel stride 4.

`tests/test_free_space.py`: 13 tests on a 20×20 toy — ray fan geometry,
once-per-cell counting, margin, scene transform, the frame-local classifier
on a ground plane with a wall, and its invariance to a uniformly scaled
frame (the chunk-drift case). Three failed first and all three were the
test's arithmetic (the wall lands on row 17, not 16).

## What is still open

- **21% of the path is not FREE**, and 47% of the grid is UNKNOWN. The loop
  interior was never observed at all (the depth cut at 0.74 DA3 units is
  ~2.7 m; rays stop there). Raising `--depth-trunc` for the carve is the
  cheap experiment; DA3's far depth is where it is least reliable, so it
  should be measured against the same camera metric, not assumed.
- **The camera height is an assumption** (1.4 m). It sets the metre; it does
  not affect the classification's invariance. If the phone height is known
  for a capture, pass `--camera-height-m`.
- **The drift itself.** The per-frame scale spread (p10–p90 of metres per
  unit: 7.9–19.7) is the number that says how inconsistent the streaming
  volume is. It is also why the 13 Sep splat's far field was culled and why
  the hybrid seed's `scale_spread` was 12.3. Until the SfM is consistent, the
  level set is a display artefact and the occupancy comes from the frames.
