<!-- v1.1.0 -->
# The hybrid: COLMAP geometry, DA3 density

> Sun, 13 Sep 2026. Written after the resolution audit that found the near-field
> regression, and after the lens/shared-camera fix in `51dda46`.

## Why

Three runs of the *same* video (`backyard_134s_sunny.mp4`) separate the two
halves of the problem cleanly.

| | Jun 14 COLMAP | Jul 13 DA3 | Sep 13 DA3-Streaming |
|---|---|---|---|
| SfM input resolution | **1920x1080** | full-res in, DA3 internal resize | **504x280** |
| lens distortion | **solved** (k1 .0141, k2 -.0147) | none | none |
| intrinsics | one shared camera | per frame, 3.75% spread | per frame, 4.30% spread |
| seed points | 185,355 triangulated | 500,000 (capped) | 1,060,163 |
| gaussians | 1,626,430 | 1,839,412 | 1,643,538 |
| **median gaussian radius** (path extents) | **0.16** | 0.86 | 0.83 |
| p99 / p50 radius (camera-free) | 11.04 | 9.47 | **2.06** |

COLMAP put half its gaussians within 0.16 camera-path extents of the walk —
density where you actually stood. Both DA3 runs put their median gaussian in a
shell at ~0.85, with the middle thin. That is the near-field complaint, measured.

But COLMAP is sparse (185k points, from a matcher that needs texture) and slow
(2442 s of sequential matching on this video). DA3 is dense and fast and gives a
depth value for every pixel including the blank wall a matcher has nothing to say
about.

So: **poses and intrinsics from COLMAP, depth from DA3.** This is what most
current pipelines actually do, and it is proven rather than novel — which is the
reason for choosing it over the alternatives in the options table below.

## The one hard part: scale

DA3's depth is defined up to an unknown transform per frame. COLMAP's points are
in one consistent frame. Where a sparse point lands in a frame we get a pair
`(predicted depth, true depth)`; a few hundred of those solve
`true ~= a * predicted + b`.

`vaultwares_studio/hybrid_seed.py` does this, with three decisions worth knowing:

- **Robust, not least-squares.** A sparse point that projects into a frame is not
  necessarily *visible* in it — it can sit behind a wall, and an occluded point
  always reads too far. The outlier tail is one-sided, which plain least squares
  cannot survive. IRLS with a Huber loss, threshold re-derived each iteration
  from the median absolute residual (scene scale is arbitrary, so a threshold in
  metres means something different every run).
- **Affine and scale-only, both reported.** A pure scale is the right model for a
  metric predictor. If the shift is consistently doing real work
  (`shift_matters_fraction` high), DA3's depth is *relative* on this scene, not
  metric — worth knowing before trusting it anywhere else.
- **Unalignable frames are dropped, not guessed.** A frame placed at the wrong
  depth does not merely add nothing. It adds a whole surface in the wrong place,
  and splatfacto will dutifully fit gaussians to it.

Measured against the June 14 bundle: the median frame has **45,966** sparse
points projecting into it (worst 6,089), so the fit is over-constrained by three
orders of magnitude. `MIN_CORRESPONDENCES = 24` is a guard against a degenerate
frame, not a tuning knob.

### Verified end to end

Depth maps were manufactured from COLMAP's own geometry at a known scale of
**3.7** and fed through the real bundle:

```text
median_scale            3.71111     (0.3% error)
scale_spread            0.0294
frames_aligned          40 / 40
points_after_dedup      282,709     from 40 frames at stride 4
```

## Running it

All four steps exist today; the hybrid is a composition, not new plumbing.

**1. COLMAP poses.** Any `SfmMethod.COLMAP` preset. For
`backyard_134s_sunny.mp4` this is already done — June 14's bundle is on disk
with its database and sparse model retained:

```bash
ls "D:/3D Reconstruction/vaultwares-studio-jobs/data/jobs/local-run-20260614-234541/reconstruction/remote_out/processed_min.zip"
```

**2. DA3 depth** on the same frames. `--sfm-only` already writes `depths.zip`
(per-frame `.npy` named by source stem) alongside `processed_min.zip`. Only the
depth is used; DA3's poses and intrinsics are discarded.

**3. Fuse, locally, on the CPU.** Minutes, free, and every parameter sweep is
free too — which is the point of doing it outside the job:

```bash
python tools/build_hybrid_seed.py --colmap <colmap_processed_min.zip> --depths <depths_dir> --out data/jobs/<job>/hybrid --stride 2 --voxel 0.02
```

**4. Train** against the hybrid bundle with `tools/queue_train_only.py`.

### Read the report before training

`scale_spread` is the decision. Near zero means DA3's depth is consistent against
COLMAP's geometry and the fused cloud is sound. Above ~0.25 the tool warns: the
per-frame scale is drifting, and the fused cloud is only as good as its worst
frames. **That number is also the cheapest available test of the streaming
pipeline's suspected global-scale problem** — the thing most likely to explain
"only two good standing spots" — and it costs no GPU time.

## Deferred, deliberately

Both of these are real and probably worth doing; they are parked so the hybrid
gets tested without confounds.

### A — turn on pose refinement

`splatfacto.py:213` defaults `camera_optimizer` to `mode="off"`. Correct for
COLMAP, whose poses are already bundle-adjusted. Wrong for a feed-forward model
that never globally optimised anything. Every published DUSt3R/VGGT/DA3 → 3DGS
pipeline refines poses during training; we inherited "off" from the COLMAP era
and never revisited it.

```text
--pipeline.model.camera-optimizer.mode SO3xR3
```

One flag, one job, ~$0.35. Note it becomes *less* useful on the hybrid path —
COLMAP's poses are the good ones — so its real value is as a rescue for the
pure-DA3 path, and it should be measured there.

### B — MCMC densification

splatfacto hard-codes `DefaultStrategy` (`splatfacto.py:315`), which only splits
and clones gaussians that already exist. That is exactly the dead end the 45k
refine hit: it came back *smaller*. gsplat 1.4.0 already ships `MCMCStrategy`,
which teleports low-opacity gaussians into high-opacity regions and holds a
constant budget — the direct answer to "nothing to split and no gradient to
drive it". It is sitting unused in our own venv.

Needs a `SplatfactoModel` subclass in this repo (~50 lines) swapping the strategy
and registering the method. Worth doing after the hybrid, because a denser and
better-placed seed cloud may make the densification question moot.

### Also noted

- `use_bilateral_grid=False` and no per-image appearance embedding — the actual
  reason sunny and overcast captures cannot be mixed in one splat.
- splatfacto ignores DA3's depth as *supervision*; the hybrid uses it for seeding
  only. Depth-supervised splatting (DN-Splatter and relatives) is the further
  step, and is option C's main argument.
- `local-run-20260630-193814` has a principal point at **cy = 304** where it
  should be ~540. That bundle is broken; do not use it as a reference.
