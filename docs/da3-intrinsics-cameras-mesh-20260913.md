<!-- v1.0.0 -->
# DA3 intrinsics fix, capture cameras in USD, and a fused mesh

> **Sun, 13 Sep 2026** — One backyard video through the ZeroGPU console and
> an L4 training job, with three changes to the pipeline on the way: the
> intrinsics bug that skewed every September splat, the camera staging stage
> finished with the reconstruction's real cameras, and a TSDF mesh from DA3's
> own depth. Companion to `plan-da3-streaming-20260801.md`.

---

## The regression: intrinsics scaled from the wrong resolution

The ZeroGPU console's *High* preset resizes frames to **672×378** before
DA3-Streaming. DA3 does not care: it resizes its input to its own working
resolution (**504×280** for DA3-LARGE) and reports `intrinsic.txt` in that
frame. The retained `results_output/*.npz` prove it — their `image` arrays
are 280×504 for a 672×378 input, and `cx, cy` are exactly (252, 140).

`tools/import_zerogpu_artifact.py` scaled those intrinsics by
`1920/672 = 2.857` instead of `1920/504 = 3.810`. On IMG_1274 that produced
`fl_x = 1231, cx = 720, cy = 400` on a 1920×1080 image: focal length 25 %
short, principal point 240 px left and 140 px up of centre. The
`splatfacto-img1274-l4-20260910` run trained against that. Every "the September
splat looks worse than July" observation has this underneath it; July's runs
fed 504-wide frames, so the assumption happened to hold.

Fix, in `streaming_convert.py`:

- `infer_stream_size()` reads the retained npz image shape, then the
  principal point (`2*cx, 2*cy`), then a fallback — in that order.
- `validate_principal_point()` refuses a `stream_size` whose centre is more
  than 10 % away from `(cx, cy)`. The old code path now raises instead of
  silently producing a skewed camera.
- `write_processed_bundle(stream_size=...)` treats the argument as a fallback.

Verified on the backyard video against July's non-streaming DA3 run of the
same file: **882 px** now vs **898 px** then (a 94° horizontal field of view;
it is a wide lens), principal point (960, 540) both times.

The console's 672×378 preset therefore buys nothing over 504×280 — same
tokens, same VRAM, larger upload. It was left alone this session because a
Space rebuild was not worth the remaining ZeroGPU budget; see *Not done*.

## The run

Source: `backyard_134s_sunny.mp4` (134 s, 1920×1080, 59.94 fps), the same
file as the July `da3-standard` benchmark.

| Stage | Where | Time | Cost |
|---|---|---|---|
| Frame selection (940 candidates at 7 fps, 500 kept, one per bucket) | Space CPU + local replay | ~1 min | free |
| DA3-Streaming, chunk 90 / overlap 45, loop closure on | ZeroGPU (RTX PRO 6000 MIG 48 GB) | **102 GPU s** | free |
| Import with the intrinsics fix, training bundle, upload (250 MB + 266 MB) | local | ~4 min | free |
| splatfacto, 20 000 iterations, scale regularisation | HF Jobs `l4x1` | see below | see below |
| TSDF mesh from 500 depth fields | local CPU | **9.8 s** | free |
| Camera staging: 500 capture cameras, retrace path, mesh reference | local | seconds | free |

Streaming output: 500/500 frames posed. Trajectory (reconstruction units):
path length 18.65, extent 5.46, **start-to-end distance 0.33, a closure ratio
of 0.06** — the walk came back to where it started. Loop detection at the
console's fixed similarity threshold of 0.85 reported **2 pairs**, both
near-neighbours (400↔380, 243↔232), and nothing joining the end to the start.
The threshold is too strict for a real revisit and is hard-coded in the Space;
IMG_1274 got zero pairs at the same setting.

### Training result

| | July `da3-standard` | Aug `da3-stream` | **Today** |
|---|---|---|---|
| Frames posed | 80 | 500 | 500 (loop closure on, intrinsics fixed) |
| Iterations | 15 000 | 15 000 | 20 000 + scale regularisation |
| Training | 1 372 s | 1 317 s | **1 824 s** + 48 s export, `l4x1` |
| Gaussians | 1 839 412 | 956 304 | **1 036 140** |
| Cost | ~$0.36 | ~$0.36 | **~$0.43** (SfM was free) |
| Wall clock, submit to splat on disk | | | 36 min (07:55 -> 08:31); 44 min from the first ZeroGPU probe |

Gravity alignment rotated the trained scene 89.4 degrees to bring +Y up (the
DA3 world had the yard on its side). The mesh was re-fused after alignment
through `scene_frame_transform`, and mesh, splat and trajectory share one
frame — see `review/sep13/splat_overview.png`. A quality judgement on the
render is the user's; what this run establishes is that the cameras are
right, the pipeline is repeatable end to end without a hand-built submission,
and structure now comes out beside appearance.

## Camera staging, finished

`vaultwares_studio/capture_cameras.py` + `stages/camera_staging.py`:

- **Every registered frame** becomes a `CaptureFrame` (pose in the viewer
  world — trainer normalisation and gravity rotation applied through the
  existing `prepare_retrace_transforms` — plus `fx, fy, cx, cy, w, h` and a
  timestamp) and is written to `usd/capture_cameras.json`. That file is the
  Cosmos hand-off: the conventions are stated in it, and `cosmos_output` now
  points at it.
- **USD**: `/World/Capture/CaptureCamera` is one animated `UsdGeomCamera`
  with per-frame transform *and* per-frame lens time samples. Pixels map to
  a 36 mm filmback: `focalLength = fx·36/w`, the vertical aperture absorbs
  `fx≠fy`, and the aperture offsets carry an off-centre principal point (with
  the y sign flipped, image down vs USD up). `/World/Capture/Trajectory` is
  the path as a polyline; `/World/Capture/Keyframes/Frame_NNNNN` are static
  cameras every 25 frames plus the last, tagged with `vw:frameIndex`,
  `vw:sourceFile`, `vw:time`.
- **Render path**: the default is now a *retrace* of the real trajectory
  (thinned to ~60 keyframes, capped at 30 s) rather than the synthetic orbit,
  whenever poses exist. Viewport choices and captured walkthroughs still win.
- **Presets** are scaled to the scene's bounds instead of the fixed 5 m room
  they were authored for.
- `save_active_camera()` re-composes with the capture cameras and the mesh, so
  picking a path in the viewport no longer throws them away.
- `camera_scene.scene_frame_transform()` exposes the normalisation +
  gravity chain as one matrix; the mesh uses it.

## Structure: a mesh from DA3's depth

`vaultwares_studio/depth_fusion.py`, `tools/fuse_streaming_mesh.py`.
Streaming retains, per frame, depth and confidence at 504×280, the working
intrinsics, and the SIM3 `(s, R, T)` that placed the frame's chunk in the
global map; `camera_poses.txt` has the global camera-to-world. Global depth is
`depth · s` from the global pose. Open3D's scalable TSDF fuses the 500 frames
(confidence below 0.75× the frame mean masked, depth truncated at 1.5× the
90th percentile, voxel = reach/350) into `reconstruction/mesh.ply` and a
`UsdGeomMesh` layer `reconstruction/mesh.usda`, in the scene frame, referenced
at `/World/DigitalTwin/Surface`.

Backyard: 301 k raw triangles, 285 k after dropping crumbs, 155 k vertices,
9.8 s on the CPU. Scale is DA3's unit; nothing here is metric.

## Not done, and why

- **Space changes** (drop the 672×378 preset, expose the loop similarity
  threshold, keep results by default) need a rebuild and a verification run;
  with 17 minutes of ZeroGPU left this month that verification was not worth
  it. `spaces/da3-zerogpu/core.py` is where both live.
- **Depth-supervised training** (dn-splatter: depth + normal losses, mesh
  export) is the next real quality lever and needs a worker image rebuild.
  The DA3 depth fields it would consume are now retained and already fused.
- **Per-view training budget**: 20 k iterations over 500 views is 40 per
  view against July's 188 for 80 views. Still a placeholder, not a choice.

## Files

| File | Change |
|---|---|
| `vaultwares_studio/streaming_convert.py` | resolution inference, principal-point guard |
| `tools/import_zerogpu_artifact.py` | uses it; keeps `results_output`; records sizes, loop pairs, candidates |
| `vaultwares_studio/capture_cameras.py` | new: CaptureFrame, JSON hand-off, USD authoring |
| `vaultwares_studio/depth_fusion.py`, `tools/fuse_streaming_mesh.py` | new: TSDF mesh + USD |
| `vaultwares_studio/camera_scene.py` | `compose_scene(capture_frames=, mesh=)`, `scene_frame_transform()` |
| `vaultwares_studio/stages/camera_staging.py` | capture cameras, retrace default, scaled presets, mesh |
| `vaultwares_studio/stages/cosmos_output.py` | annotations point at the hand-off files |
| `tools/prepare_zerogpu_training.py` | new: replay the console's frame selection, submit Job B with the current entrypoint overlaid |
| `tests/test_capture_cameras.py`, `tests/test_depth_fusion.py`, `tests/test_streaming_convert.py` | 13 new tests; suite at 189 passing |

---

# Part two: why the splat narrowed, Cosmos Reason wired, multi-capture

> Later the same day, after looking at the trained splat in the viewport.

## The splat got narrow, and it was a one-flag mistake

User's read: *"only ~2 spots you can stand that isn't surrounded by blur… much
more concentrated instead of being all over the place, a little too much even…
there are places that should be mapped that are cut off."* Measured, that is
exactly right:

| | July `da3-standard` | **13 Sep, 20k iters** |
|---|---|---|
| Seed cloud extent / camera-path extent | 1.19x | 1.14x |
| **Trained splat extent / camera-path extent** | **3.84x** | **0.51x** |
| Gaussians within 0.2 path-extents of the walk | 52% | **100%** |
| Gaussians beyond 0.4 path-extents | 11% | **0%** |

The two runs were handed **seed clouds of the same shape**. July's splatfacto
grew 3.2x beyond its seeds and reached out to 6.2 path-extents; the 13 Sep run
*shrank* to under half its seeds. Nothing was ever built out there — it was
built and then thrown away.

**Cause.** splatfacto's `stop_split_at` defaults to **15000**. The 13 Sep run
asked for 20000 iterations and left that default, so its last 5000 iterations
densified nothing and only culled, with the periodic alpha reset (every
`reset_alpha_every * refine_every` = 3000 steps) knocking opacities down at
15000 and 18000 for the pruner to collect. Far-field gaussians are the faint,
weakly-supported ones; they went first. July ran 15000 iterations against the
same 15000 default, so it densified to its final step and never had a
culling-only tail.

Fixed in `tools/prepare_zerogpu_training.py`: `stop_split_at` is now derived as
90% of the requested iterations, in one place, with the reasoning in a comment
so nobody re-introduces it by raising `--iterations`. Scale regularisation,
added the same morning, is dropped — it was unproven, and August's run without
it came out just as narrow, so it is not what separates the two.

## Incremental training works, once a real bug is out of the way

Yes, more training on the same backyard is incremental. `--refine-mode` already
existed in the worker and had **never once run**. Its first run failed:

```text
ValueError: invalid literal for int() with base 10: 'nerfstudio_model'
```

`ns-train --load-dir` wants the directory that holds the `.ckpt` files.
The entrypoint passed the *run* directory, which also contains `config.yml`
and `nerfstudio_models/`; nerfstudio's `_load_checkpoint` does a bare
`os.listdir()` and parses a step number out of every entry. Now
`find_checkpoint_dir()` finds the newest directory that actually contains
checkpoints, and `tests/test_refine_checkpoint.py` pins it.

The refine costs **nothing to stage**: `frames.zip`, `processed_min.zip` and
`model.zip` are already in the artifact dataset from the first run, so
`--refine` pulls them Hub-side and uploads only the 19 KB worker overlay.
`--iterations` is the new *total*, and `stop_split_at` at 0.9x lands well past
the resume point — without that a resumed run would only cull, which is the
failure this whole section is about.

## Cosmos Reason, wired

Six months blocked, unblocked. `vaultwares_studio/cosmos_reason.py` +
`stages/cosmos_output.py`.

A vision-language model looks at the frames the capture cameras actually saw,
names what is there, and **every observation is anchored in 3D** by
unprojecting its image position through that frame's DA3 depth (times the
chunk scale, through the global pose, through `scene_frame_transform`) into the
same world as the splat. Repeat sightings cluster into one annotation. Output:

- `cosmos/cosmos_annotations.json` — label, kind, static/movable, confidence,
  3D position, spread, contributing frames, aliases.
- `/World/Annotations/<slug>` Xform prims in the USD stage. A named place is
  an ObjectNav goal, which is what roadmap M4 wanted.

**Providers**, same interface: `nvidia` (API catalog), `ollama` (local GGUF),
`none`. The intended model is `nvidia/cosmos-reason2-8b` and it is **one flag
away** — but it answers `404 Function not found for account` for this key, an
NVCF entitlement rather than a code problem. So does `nvidia/vila` and
`microsoft/phi-3-vision`. Reachable today and therefore the default:
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`. `meta/llama-3.2-11b-vision-instruct`
also answers; the 90B one times out.

First real run, backyard, 12 views, ~40 s per view, free tier:

```text
19 objects, 13 anchored in 3D
blue car [-1.17, -0.17, 0.44]   silver jeep [-0.77, -0.05, -0.13]
blue screen door [0.70, 0.02, -0.31]   stacked firewood [0.80, -0.09, -0.37]
grass lawn, paved driveway, asphalt road, yellow house wall, red latch …
```

The cars land at the street end of the walk and the firewood by the shed, which
is where they are. One flaw the real output exposed and the tests now pin:
clustering matched labels by substring, so *"Blue door"* and *"blue screen
door"* stayed two objects 0.15 apart. It matches word sets now — one label's
words being a subset of the other's makes them one thing, while *"large green
hedge"* and *"medium green bush"* share only a colour and stay apart.

## Several videos, one map

`tools/compose_multi_video.py`. Frames from every capture go into **one ordered
sequence**, clip after clip, and DA3-Streaming poses them together so its SIM3
chunk alignment and loop closure stitch the overlaps.

**splatfacto is never told where the seams are, because by the time it runs
there are none** — the poses are already in one world. That is the whole
mechanism, and it is why this belongs in the SfM leg rather than in training.

Two things that decide whether it works:

- **Order.** Streaming assumes consecutive frames are close. Each video
  boundary is one hard cut, which it survives only if loop closure finds the
  revisit. So loop closure is **on by default** here, unlike the single-video
  path.
- **Lighting, and this one matters more.** splatfacto has no per-image
  appearance model. Mixing a sunny capture with an overcast one puts two
  different colours on one surface and the optimiser splits the difference.
  Group captures by weather. The tool allows mixing and records it in the
  manifest; the splat is where it shows up.

First run: `cloudyday1_june14_194sec` + `cloudyday2_june14_348sec`, both
overcast, same day — 400 frames each, 800 total, loop closure on, `a10g-small`.
The open question is whether the two captures actually register into one map;
that is what this job answers.

