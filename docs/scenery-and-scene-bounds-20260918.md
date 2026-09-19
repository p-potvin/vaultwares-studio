<!-- v1.0.0 -->
# Why the new splats feel cramped — Fri, 18 Sep 2026

The user's observation: the 14 June splat has background houses that read as
photographs, and the September and 18 Sep splats do not. They feel like a small
room rather than a place. This is not a rendering impression. It is measurable,
it has one cause, and the cause is upstream of splatfacto.

## The measurement

Fraction of gaussians lying beyond N times the core radius, where the core
radius is the median distance from the cloud's own centre:

| run | SfM | gaussians | beyond 2x | beyond 5x | beyond 10x |
|---|---|---|---|---|---|
| **14 Jun** | **COLMAP** | 1,626,430 | 12.8% | **3.6%** | **1.1%** |
| 13 Sep | DA3 | 1,643,538 | 2.2% | 0.0% | 0.0% |
| 17 Sep | DA3 | 384,175 | 5.5% | 0.0% | 0.0% |
| 18 Sep | DA3 | 569,661 | 14.9% | 0.0% | 0.0% |

Every DA3 run is **exactly zero** past five core radii. That is not a gradual
falloff, it is a wall.

The seed clouds show where the wall comes from:

| seed | points | p95 radius | p99.9 | max | max / p95 |
|---|---|---|---|---|---|
| COLMAP, 14 Jun | 185,355 | 6.58 | 20.32 | **90.34** | **13.7** |
| DA3, 18 Sep | 4,807,560 | 6.49 | 8.27 | 9.20 | 1.42 |
| DA3, 13 Sep | 1,060,163 | 2.66 | 3.21 | 3.59 | 1.35 |

**The two agree almost exactly at p95** (6.58 against 6.49). DA3 is not worse
in the near or middle field — on 18 Sep it is denser there, and it has 26x more
points. What it does not have is the tail: the 0.1% of COLMAP points sitting
between 20 and 90 units out. Those few thousand points are the houses across
the street, the far trees, the horizon. They are a rounding error in the count
and they are the entire difference in how the scene feels.

## Why DA3 has no tail

`streaming_convert.py` already says it, in a docstring written before anyone
was looking for this:

> streaming already applies its own `depth_threshold` and confidence filtering,
> and its output has a max/p95 radius ratio around 1.4

That ratio is the wall, and it was recorded as a reason *not* to apply further
spatial filtering. It is a property of monocular depth: DA3 estimates a depth
per pixel and the Space's config truncates at `depth_threshold: 15.0`, with
`Pointcloud_Save.conf_threshold_coef: 0.75` dropping low-confidence samples on
top. Far pixels are exactly where monocular depth is least reliable, so they
are cut. COLMAP has no such limit — it triangulates whatever it can match
across the sequence, and a distant house matched across a hundred frames
triangulates fine.

So the chain is: no far seed points → splatfacto has nothing to densify in the
background → `cull_alpha_thresh` prunes the faint gaussians that do form → a
hard-bounded scene.

This is also the same root cause as the "neighbour's house collapsing onto
mine" artefact: past the truncation, DA3's depth is guesswork, and whatever
survives lands in the wrong place.

## Where culling happens, in order

Answering the direct question — every stage that removes far geometry:

1. **DA3-Streaming, `depth_threshold: 15.0`** — in the Space's `_config()`,
   `spaces/da3-zerogpu/app.py`. This is the one that matters. Exposed in our
   own code, changeable without a rebuild.
2. **`Pointcloud_Save.conf_threshold_coef: 0.75`** — drops samples below 0.75x
   the frame's mean confidence. Far pixels are disproportionately low.
3. **`splat_filter.py`** — an aggressive radial filter that is *deliberately
   skipped* on the streaming path, precisely because the cloud is already
   bounded. It is not the culprit.
4. **`--pipeline.model.cull-alpha-thresh 0.05`** — splatfacto's own pruning,
   already gentler than the 0.1 default.
5. **`fuse_nanovdb.py --depth-trunc 0.74`** — volume path only, does not affect
   the splat.

## Observation 2: haze is worse in enclosed spaces

Reported: the table area, walled by curtains and trees, is among the worst,
despite five visits and deliberate loop stops. Open areas — the front lawn, the
driveway, the garden — are consistently better, in both old and new captures.

That pattern is consistent with the per-view opacity trend already measured
(median opacity 0.811 at 40 iterations per view, 0.646 at 18.75, 0.585 at
12.5): under-converged gaussians are translucent, and translucency accumulates
along a ray. In an open area a ray hits one surface. In a three-walled nook it
passes through several layers of foliage and fabric, each contributing haze, so
the same per-gaussian deficiency is several times more visible. Revisiting does
not fix it because the problem is the iteration budget per view, not the number
of views.

**Untested alternative worth ruling out**: thin structures like curtains and
leaves are also where the intrinsics and lens distortion matter most, and where
DA3's depth is noisiest at close range. Distinguishing these needs a per-region
reprojection error, not an opinion.

## What to try, cheapest first

None of these have been run. Listed with the reason each might work.

1. **COLMAP for SfM, DA3 for density.** This is the one the evidence points at.
   COLMAP produced the tail; the 14 Jun run took 41 minutes of CPU for it. The
   `split-flavor-pipeline` note already proposes COLMAP on `cpu-basic` at about
   $0.03/hr for exactly this shape of job. The hybrid seed work
   (`docs/hybrid-colmap-da3-20260913.md`) is the unfinished half of it.
2. **Raise `depth_threshold` in the Space.** One number, free to test on
   ZeroGPU. The risk is that it admits the unreliable far depth that causes the
   collapse artefact, so it should be judged on the far field specifically, not
   overall.
3. **splatfacto MCMC** (`strategy: "mcmc"`, `max_gs_num`). Does not add
   scenery, but it bounds the model size directly, which is the allocation the
   memory gate cannot currently predict. Needs the deployed nerfstudio checked
   or the image rebuilt.
4. **A sky or background model.** Standard in unbounded-scene work. Nerfstudio's
   splatfacto has no sky model; methods that do exist elsewhere. Unresearched.

## Measurements still missing

- **VRAM was never recorded.** The 18 Sep heartbeat logged host RAM but not
  GPU memory, so the honest answer to "how much VRAM did that need?" is that we
  do not know. Now added to the heartbeat via `nvidia-smi`, so the next run
  reports it.
- Peak host RAM was 36.8 GB for 2000 frames on the 46 GB box. To fit the 30 GB
  L4 the image cache must come down to about 10 GB, which is about 1600 frames
  at full resolution — the gate's own arithmetic, and it halves the hourly rate.
