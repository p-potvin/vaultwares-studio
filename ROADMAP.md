# Roadmap: VaultWares Studio v1.0

Approved plan (2026-06-09, detailed): `docs/plans/plan-v1-remote-first-20260609.md`

Direction: **remote-first compute**. The local PC (12 GB VRAM, also hosts the VaultWares API) stays light: GUI, WebGL splat viewport, ffmpeg frame extraction, CPU grid-nav sim. Heavy stages (COLMAP/splat training, Habitat RL, Cosmos, renders) run on rented GPU compute — Hugging Face Jobs primary (PRO, per-minute billing, custom Docker), generic SSH GPU-server backend second (future 48 GB L40S box). No WSL2. Every paid remote job requires explicit cost confirmation.

## M0: Remote Execution Foundation (~8 d)
StageRunner abstraction (`runners/base.py`): `LocalStageRunner` (Popen streaming + cancel), `HfJobsStageRunner` (run_job + bucket Volumes, poll/cancel, log→progress), `vw-studio-worker` Docker image v1, manifest schema v2 (placement/runner/cost per stage), cost-confirm dialog + spend ledger.

## M1: Real Reconstruction, Remote (~8 d)
Worker entrypoint: `ns-process-data` (sequential matching) → `ns-train splatfacto` → `ns-export gaussian-splat`. Quality presets pick flavor (Draft t4/l4 <$1, Standard l4/a10g ~$1, High a10g/a100). Full-attribute splat PLY (`splat_io.py`, stop flattening through Open3D), proper PLY→USD (26.03 schema probe, UsdGeomPoints+primvars fallback).

## M2: Interactive Viewport + Camera Staging (~18 d, local)
QWebEngineView + vendored GaussianSplats3D (three.js) splat viewer; fly controls; place/aim cameras; keyframe timeline (Catmull-Rom + slerp in `camera_paths.py`); `CameraEntity` → manifest + USD time-sampled xformOps; remote `ns-render camera-path` walkthroughs. Includes `gui/` package split of gui_app.py. Stage rename `usd_cameras` → `camera_staging` (uses NEEDS_USER_INPUT).

## M3: Robot Lab (~16 d)
`NavSimBackend` interface. Grid-nav backend local (gymnasium + SB3 PPO over occupancy grid, CPU, free). Habitat-Sim backend remote (native Linux in the worker image — no Windows port needed). `sim_export` stage: poisson mesh → trimesh cleanup → occupancy grid + navmesh. Episode replay animated in the splat viewport; Robot Lab GUI tab (train/evaluate/watch).

## M4: Cosmos AI Layer (~5 d)
Cosmos Reason 2-2B annotation as a remote stage (fp16 on 24 GB; local INT4 opt-in) → real `cosmos_annotations.json`; labels become ObjectNav goals. Transfer 2.5 stays scope-boxed to the future 48 GB SSH box.

## M5: Polish & Packaging (~8 d)
Onboarding wizard (HF token, demo job), actionable error cards, spend dashboard, slim EXE install (heavy venv now optional), docs refresh. EN/QC translation explicitly deferred (lowest priority); strings stay behind `_STRINGS`/`_t()`.

## M6 (optional, deferred): Isaac Sim Bridge
Remote headless Isaac (48 GB box or a100); USD + physics colliders; Isaac Lab RL out of scope.

## Watch items
- cuTile / CUDA Tile (`cuda-tile`, CUDA 13.2): kernel-authoring DSL — relevant only if custom splat kernels ever land here.
- **ARKit-pose ingestion** (user-requested 2026-06-10): accept captures that carry device poses (ARKit/ARCore visual-inertial tracking — e.g. Polycam/Record3D/Stray Scanner exports, or a small companion capture app). Poses seed or replace COLMAP SfM, give absolute scale (IMU), and survive feature-poor scenes (bare walls, dim rooms) where pure-pixels SfM fails — exactly the failure mode of the first test captures. Candidate slot: M1.5 ingestion option after the camera-staging milestone; nerfstudio already accepts `transforms.json` with known poses, so the integration seam is the worker's `ns-process-data` step.

---

### Legacy milestones (superseded by the plan above)
M1 Working Local App (done) · M2 Gaussian Splat Pipeline (→ new M1) · M3 Native OpenUSD (→ new M1) · M4 Isaac Sim (→ new M6) · M5 Cosmos (→ new M4)
