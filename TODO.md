# Project: VaultWares Studio

Plan of record: `docs/plans/plan-v1-remote-first-20260609.md` (M0–M6). Legacy phases below are kept for history.

## M0: Remote Execution Foundation

- [x] `StageRunner` abstraction (`vaultwares_studio/runners/base.py`): StageContext, StageResult, CancelToken, cost estimates
- [x] `LocalStageRunner`: Popen line-streamed output, cancel (psutil process-tree kill), timeout
- [x] `HfJobsStageRunner`: consent gate before any network call, dataset-repo artifact transport (`jobs/<id>/<stage>/{in,out}`), launch/poll/cancel (≥10 s poll), log fetch, output download
- [x] Manifest schema v2: `schema_version`, per-stage `placement`/`runner`/`params`/`cost`, `spend_ledger`, v1 migration on load
- [x] `record_spend()` ledger helper
- [x] `pipeline._run_command` delegates to LocalStageRunner (live logs + cancel)
- [x] Stage placement defaults (reconstruction → remote; honored from M1)
- [x] Worker image v1: `docker/worker/Dockerfile` (COLMAP + nerfstudio + hub) + `vw_stage.py` + `tools/build_worker_image.ps1`
- [x] GUI Settings: HF token (OS keyring), artifact repo, default flavor, cost-confirm dialog, echo-test button
- [x] Tests: runner streaming/cancel/timeout, cost-denial, config round-trip, manifest v2 migration, spend ledger (19 passing)
- [x] Live verification: echo job round-trip on `cpu-basic` (needs HF token; run from Settings → "Run Echo Test Job")
- [x] Build & push `vw-studio-worker:0.1` to Docker Hub (needed before M1 remote reconstruction)

## M1: Real Reconstruction, Remote

**Quality Policy (2026-06-10):** Reject bad footage rather than engineer workarounds. Frame selection improved with variance-of-Laplacian blur awareness (~280 sharp frames from ~600 extracted). Capture guidelines: slow walk, good light, single lens, 30-60s landscape.

- [x] Worker recon entrypoint (`docker/worker/recon_entrypoint.py`): ns-process-data (sequential) → ns-train splatfacto (flag probe drops unsupported args) → ns-export gaussian-splat; structured error.json (e.g. too_few_registered_images) + summary.json + model.zip checkpoint for M2 renders
- [x] Quality presets (`presets.py`): Draft/Standard/High + local-debug; flavor + cost per preset; VRAM-saver train args
- [x] Full-attribute splat PLY (`splat_io.py`): read/write 3DGS PLY, decimated `cloud_preview.ply`, no more Open3D flattening
- [x] PLY→USD: native 26.03 schema probe; lossless `UsdGeomPoints` + `primvars:gsplat:*` fallback (verified round-trip in tests)
- [x] Remote-stage wiring: `reconstruction` honors placement, zips frames, runs the worker entrypoint via HfJobsStageRunner, records spend; CostDenied/failure falls back to local quick path then placeholders
- [x] Remote log streaming with nerfstudio % progress parsing in HfJobsStageRunner
- [x] GUI: quality-preset dropdown on the Studio tab; pre-run cost-confirm dialog for remote reconstruction; viewer prefers `cloud_preview.ply`
- [x] Tests: splat round-trip/USD, presets, fake-remote reconstruction wiring (30 passing)
- [x] Live verification (2026-06-10): backyard_99s_cloudy.mp4 → 280/280 registered (CPU SIFT), 477,731 gaussians (draft, l4x1, $0.36), cloud.ply 107MB + preview + USD + model.zip checkpoint. Bugs found and fixed along the way: container GPU-SIFT garbage, --vis none removal, HfHubHTTPError fallback bypass, private-Space image 500
- [x] Standard run COLMAP mapper non-determinism fix (2026-06-10): degenerate RANSAC init pair in incremental mapper (2/280 registered vs expected 280/280). Fix: retry_mapper() re-runs only colmap mapper on existing database with relaxed init (tri-angle 8 then 4, lower inlier floors, multiple_models 0), picks largest model, regenerates transforms via ns-process-data --skip-colmap. Worker rebuilding + Standard rerun chained in background.
- [ ] OOM auto-retry at next-lower preset (deferred — needs mid-run confirm UX; failures currently suggest a lower preset)

## M2: Interactive Viewport + Camera Staging (in progress)

- [x] Day-1 spike: QtWebEngineWidgets verified in the venv (PySide6 6.7.2)
- [x] Vendored viewer: three.js 0.160.0 + GaussianSplats3D 0.4.7 under `vaultwares_studio/webviewer/vendor/`
- [x] `vw://` URL scheme handler (app assets + job artifacts, no web server, path-traversal guarded)
- [x] Viewport tab v1 (`gui/viewport.py`): loads the job's `cloud.ply` (progressive), orbit/WASD fly, QWebChannel bridge, "Capture Camera" saves poses to `usd/captured_cameras.json`
- [x] Blender-style axis gizmo (2026-06-10): corner widget with transparent WebGL canvas, click axis balls to snap views, drag to orbit, pole snap avoids singularity
- [x] SuperSplat-style infinite zoom (2026-06-10): scroll-zoom within 0.4 units pushes pivot forward for continuous scene traversal
- [x] Embedded-viewport fix (2026-06-10): Mica disabled on FluentWindow + WA_NativeWindow on web view
- [x] Camera authoring (2026-06-10): walk pattern recreates capture trajectory with collision-free path smoothing and spiral fallback
- [ ] CameraEntity integration: captured poses → camera_director entities → manifest + USD
- [ ] Keyframe timeline + Catmull-Rom/slerp camera paths (`camera_paths.py`)
- [ ] Offline walkthrough via remote `ns-render camera-path` (reuses model.zip from M1)
- [ ] `usd_cameras` → `camera_staging` stage rename with NEEDS_USER_INPUT flow
- [ ] gui_app.py split into `gui/` package (theme, strings, widgets, main_window)
- [ ] .ksplat conversion for faster viewport loads

## M3: Robot Lab (in progress)

- [x] M3 slice 1 shipped (2026-06-10): robot_lab package with NavSimBackend contract, 2.5D occupancy grid from splat preview cloud (floor estimate + body-band obstacles + BFS geodesic field), grid-nav PointNav simulator (forward/turn/stop, goal vector + 16-ray scan, geodesic-progress reward, SPL, reachable-episode sampling), gymnasium wrapper for SB3 PPO, run_episode trajectory recorder for viewport replay. 48 tests passing. Dependencies: gymnasium + stable-baselines3.
- [ ] Walk-pattern cameras staged after Standard recon lands
- [ ] Rendering approval from user
- [ ] PPO training demo on backyard grid
- [ ] Timeline UI
- [ ] GUI split
- [ ] Cost retest

## M3+: see plan file (Cosmos, packaging)

## Cost optimization (after first green run)

- [ ] Test GPU *matching* with CPU *extraction* (`--no-gpu` currently disables both; the container bug may be extraction-only). If matching works on GPU, COLMAP drops from ~20 min to ~2 min on the L4 → run cost ~USD 0.15, no orchestration changes.
- [ ] Else: split reconstruction into two jobs — SfM on `cpu-upgrade` (~USD 0.04) + training on `l4x1` (~USD 0.13), processed dataset handed off through the artifact dataset (user-proposed; boot latency acceptable for batch work). ZeroGPU evaluated and rejected (2-min GPU slices can't hold a training run); free Space CPU tier rejected (2 vCPU, no job semantics, ToS-gray).

## Sun, 13 Sep 2026 — intrinsics fix, capture cameras, fused mesh

- [x] ZeroGPU import scaled DA3 intrinsics from the *fed* resolution (672x378) instead of DA3's working one (504x280): focal 25% short, principal point off-centre on every September splat. `streaming_convert.infer_stream_size` + a principal-point guard. See [docs/da3-intrinsics-cameras-mesh-20260913.md](docs/da3-intrinsics-cameras-mesh-20260913.md).
- [x] Camera staging authors the reconstruction's real cameras (`/World/Capture`, `usd/capture_cameras.json`), retraces the trajectory as the default render path, scales presets to scene bounds, references the fused mesh.
- [x] TSDF mesh from DA3-Streaming depth/confidence (`tools/fuse_streaming_mesh.py`), 10 s on the CPU for 500 frames.
- [ ] Space: retire the 672x378 preset (DA3 downsizes anyway) and expose the loop similarity threshold (0.85 found 2 near-neighbour pairs on a walk that closed to 6% of its extent).
- [ ] dn-splatter in the worker image: depth + normal supervision from the now-retained DA3 fields.

## Infra notes

- Worker image is built server-side on HF (local Docker/WSL unavailable): `tools/push_worker_space.py` + `tools/monitor_worker_space.py`; image ref `hf.co/spaces/clopeux/vw-studio-worker` (set in data/remote_compute.json)
- HF Jobs requires pre-paid credits: echo test + remote reconstruction blocked until the account balance is topped up (402 Payment Required on run_job)
- [DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) — ByteDance's own answer to the 80-frame `--max-sfm-frames` cap in `da3_entrypoint.py`. Sliding-window chunking (30/60/90/120 frames, 50% overlap) with loop-closure alignment (similarity threshold 0.85) fusing chunks into one `camera_poses.txt` + `combined_pcd.ply` — i.e. it already does the "fix the seams between runs" work we were about to build by hand. 8.51 FPS on an A100 across 11,373 frames; 120-frame chunks peak ~15.9 GB VRAM (fits an L4's 22 GB). Candidate replacement for the current subsample-to-80 logic once direct-3DGS (`--gs-only`, see gsplat/nvcc below) is proven — evaluate before hand-rolling any chunk-merge tooling.
- `da3-draft` (`--gs-only`, direct 3DGS) **works as of 2026-07-30** — 4,503,479 gaussians in 76 s for ~$0.02 on `a10g-small`. It had never run once before that: five stacked failures, each hidden by the one above (silent Dockerfile fallback → no nvcc in the nerfstudio base → `e3nn` ImportError swallowed by DA3 as a `NameError` → CUDA OOM → a preview-video render discarding a completed run). Full write-up, incl. the abandoned local-Windows-build side quest and the ZeroGPU finding: [docs/da3-direct-3dgs-debugging-20260730.md](docs/da3-direct-3dgs-debugging-20260730.md). Quality is soft-but-correct — feed-forward, no per-scene optimisation; use it for preview/layout, `da3-standard` when geometry must hold up.
- [ ] Repoint `da3-draft`/`da3-incremental` from `vw-studio-da3-gs` back to `vw-studio-da3` (drop the `-gs` override in `presets.py`) — the images now differ only by the nvcc/wheel/e3nn additions, which are safe for the pose+depth presets. Means rebuilding the image `da3-standard` depends on, so not done blind.
- [ ] Clamp DA3's `opacity` (max comes back `+inf`) before trusting `da3-incremental` — sigmoid-safe when rendering, but `nan`-poisons any mean/sum incl. `gaussian_merge.py`'s quality scoring.
- [ ] Percentile-clip outliers in `convert_splat_outputs` — 96% of DA3 gaussians sit within radius 1.58 but the bbox is ±600 (skewness 32.7 vs 1.07 trained), which breaks viewer auto-framing and skews the gravity estimate.
- [ ] `da3-incremental` end-to-end: shares the now-fixed `--gs-only` path, but the ICP merge itself has never run.
- [x] **DA3-Streaming integration** — phases 1–4 green. `da3-stream` preset poses **all 500 frames** (vs 80 for `da3-standard`), trains unchanged, renders with visibly sharper architecture. ~$0.38 on HF, $0.00 for the local phases. Spec: [docs/plans/plan-da3-streaming-20260801.md](docs/plans/plan-da3-streaming-20260801.md). Lifts the 80-frame cap on the *SfM* path (feeds splatfacto; does not replace `--gs-only`) and natively fixes seams between separate captures. Key findings: `da3_streaming/` is **not** pip-installed so it must be vendored; needs faiss-gpu/numba/pypose plus local weight files incl. a separate SALAD checkpoint; and the config we sketched has three keys that don't exist upstream while the proposed 65-chunk @ 672×378 is ~1.44× the tokens of the largest benchmarked config and won't fit 24GB.

---

### Legacy phases (pre-plan history)

Phase 0 Infrastructure — complete. Phase 1 Capture & Reconstruction — frames/fallback PLY done; COLMAP SfM + gsplat training move to M1 (remote). Phase 2 USD — composition + smoke tests done; PLY→USD 26.03 moves to M1. Phase 3 Isaac Sim — deferred to M6. Phase 4 Cosmos — moves to M4.

- [ ] **Settle the `da3-stream` gaussian-count drop** — 500 posed frames produced 956k gaussians vs 1.84M from 80. Two competing explanations: stronger multi-view culling (good) or under-training (15k iterations over 500 views is 30 iters/view, against 188 for the baseline). One ~$1.50 run at ~90k iterations gives 500 views a matched per-view budget and separates them. `iterations=15_000` on `da3-stream` is currently a placeholder, not a considered choice.
- [ ] DA3-Streaming phase 5: loop closure (`faiss-gpu` + DINO-SALAD weights). Raised from optional to likely-required by phase 2's tail-drift finding.
- [ ] `--downscale` is passed to `da3_entrypoint.py` from three call sites but never read — pre-existing no-op that makes `downscale_factor` look meaningful for DA3 runs.
