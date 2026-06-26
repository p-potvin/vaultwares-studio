# vaultwares-studio

`vaultwares-studio` is a local-first OpenUSD desktop app for turning room video into a digital-twin job: COLMAP feature extraction, Gaussian splat training (via Nerfstudio splatfacto), and USD composition — with remote GPU compute handled through Hugging Face Jobs (HF PRO). The GUI runs locally; heavy stages run on rented L4/A10G hardware and stream logs back in real time.

If you are just trying to answer, "Does this repo work on my machine?", start with the smoke test. It produces a real `.usda` file you can inspect. If you want the working app, run the desktop studio.

## What You Can Run Today

- `python gui_app.py` — Desktop Digital Twin Studio app (GUI).
- `python tools/headless_remote_run.py --preset draft --yes` — Headless end-to-end run, dispatches a real HF Job for reconstruction.
- `python tools/headless_remote_run.py --resume-job <job-id> --yes` — Re-queue a failed job reusing already-uploaded HF frames (no re-upload cost).
- `python tools/recover_remote_recon.py --job <job-id>` — Pull outputs from a completed HF Job whose local poller died.
- `pytest` — 83 tests covering pipeline stages, runners, splat I/O, presets, camera paths, robot lab, and resume-job wiring.
- `python usd_smoke.py` — Minimal USD artifact generator (no heavy deps).

## Quick Start

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install usd-core pytest redis PySide6 PySide6-Fluent-Widgets Pillow
python -m pytest -s
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install usd-core pytest redis PySide6 PySide6-Fluent-Widgets Pillow
python -m pytest -s
```

If the test passes, you should see a line like this:

```text
Generated USD artifact: .../data/test_outputs/smoke_scene.usda
```

## Remote Compute (HF Jobs)

Heavy reconstruction stages run on Hugging Face Jobs (PRO account required, pre-paid credits). The worker image is `hf.co/spaces/clopeux/vw-studio-worker`.

### Presets

| Preset | Flavor(s) | Est. time | Est. cost | Notes |
|--------|-----------|-----------|-----------|-------|
| `draft` | l4x1 | ~15 min | ~$0.20 | 7k iterations, 4× downscale, ≤300k gaussians. Single job. |
| `standard` | cpu-upgrade → l4x1 | ~60 min | ~$0.19 | 15k iterations, full-res. **Split:** SfM 35 min @ cpu-upgrade ($0.06) + training 25 min @ l4x1 ($0.33). Was $0.80 on a single l4x1. |
| `refine` | cpu-upgrade → l4x1 | ~110 min | ~$0.42 | Fine-tune on existing checkpoint + COLMAP joint registration. **Split:** SfM 90 min @ cpu-upgrade ($0.15) + training 20 min @ l4x1 ($0.27). Was $1.60 on a single l4x1. |
| `high` | a10g-large | ~75 min | ~$1.88 | 30k iterations, 2× downscale, ≤1.5M gaussians, antialiased. Single job. |

> **How split jobs work:** Job A runs COLMAP (feature extraction + vocab-tree matching + mapping) on `cpu-upgrade` and exits, uploading `processed_min.zip`. Job B pulls that artifact via HF xet and runs only `ns-train + ns-export` on the GPU instance. Both jobs are sequential — the GPU is idle during SfM.
>
> **Note:** Estimates are billed-per-minute; the actual HF invoice is the authoritative price. The `refine` estimate assumes ~1,500 combined frames — smaller new-frame sets will finish faster.

### Run a new job

```powershell
# Headless, approves cost automatically:
.venv\Scripts\python.exe tools\headless_remote_run.py --preset standard --yes

# Multi-video with refine from a prior completed job:
.venv\Scripts\python.exe tools\headless_remote_run.py `
  --video inputs/cloudyday1_june14_194sec.MOV `
  --video inputs/cloudyday2_june14_348sec.MOV `
  --preset refine `
  --refine-from local-run-20260614-234541 `
  --yes
```

### Resume a credit-killed job (reuse HF frames)

If a job dies mid-run (402 credit exhaustion, network drop), re-queue without re-running frame extraction:

```powershell
.venv\Scripts\python.exe tools\headless_remote_run.py `
  --resume-job local-run-20260615-065732 `
  --yes
```

This reads the prior manifest (videos, preset, refine-from) automatically, marks intake and frame extraction as already-complete, and launches a new HF Job pointing at the existing `jobs/<prior-job-id>/reconstruction/in/frames.zip`. HF xet storage makes upload time negligible regardless.

### Recover outputs from a completed HF Job

If the local poller died but the job finished and uploaded outputs:

```powershell
.venv\Scripts\python.exe tools\recover_remote_recon.py --job local-run-20260613-211202
```

## Verified Test Output

The smoke test writes this file:

- `data/test_outputs/smoke_scene.usda`

You can also generate it directly:

```powershell
.\.venv\Scripts\python.exe .\usd_smoke.py
```

## Run The Desktop App

The main app entrypoint is:

```powershell
.\.venv\Scripts\python.exe .\gui_app.py
```

The app workflow is:

1. choose a video, or use the bundled room video
2. click `Run Full Job`
3. inspect each stage from the step rail
4. open the output folder, USD artifacts, or walkthrough video
5. use `Open Latest Job` or `Open Job Manifest` to resume previous work

Jobs are stored under:

- `data/jobs/<job-id>/manifest.json`
- `data/jobs/<job-id>/frames/`
- `data/jobs/<job-id>/reconstruction/`
- `data/jobs/<job-id>/usd/`
- `data/jobs/<job-id>/camera_previews/`
- `data/jobs/<job-id>/deliverables/`

The normal app mode is fallback-safe. If Nerfstudio or COLMAP is missing, the reconstruction stage writes deterministic placeholder-safe USD/PLY artifacts so the rest of the job can still complete. Enable strict mode in Settings when you want missing heavy tools to fail the stage.

For a non-UI app verification run:

```powershell
.\.venv\Scripts\python.exe .\demo_launcher.py --headless
```

## What `pytest` Actually Covers

Default `pytest` is intentionally scoped to the tests that belong to `vaultwares-studio` itself:

- `tests/test_usd_smoke.py`
- `vaultwares-adk/omx_integration/tests/`

It does **not** recurse into:

- `.venv/`
- `cosmos-reason2/`
- `vaultwares-themes/`

That is deliberate. `cosmos-reason2` is an upstream submodule with extra platform-specific dependencies and should not be treated as the default local smoke-test target for this repo.

## Full Pipeline Reality Check

The repo also contains an agent-driven pipeline:

- `worker_runner.py`
- `manager_runner.py`
- `run_pipeline_demo.py`

That flow is now the legacy advanced orchestration path. It depends on extra tooling such as Redis, ffmpeg, Nerfstudio/COLMAP, and the vendored `vaultwares-adk` framework. The everyday app path is `gui_app.py`, which uses `vaultwares_studio` directly and does not require Redis.

The smoke test is the reliable starting point because it proves all of the following with minimal setup:

- OpenUSD imports correctly.
- A USD stage can be authored locally.
- The generated file can be reopened and validated.

## Run The Demo Locally

The local demo is the Redis-backed orchestration flow driven by `run_pipeline_demo.py`.

It dispatches three steps:

1. extract frames from `test_input.mp4`
2. run reconstruction into `data/reconstruction`
3. compose `data/digital_twin_scene.usda`

### Prerequisites

- initialize submodules first
- use a Python environment with at least:
  - `usd-core`
  - `redis`
  - `pytest`
  - `ffmpeg` available on `PATH`
- run a local Redis server on `localhost:6379`

If you want the heavier reconstruction step to use Nerfstudio/COLMAP instead of the built-in placeholder fallback, also install those dependencies from `requirements.txt`.

### One-time setup

```powershell
git submodule update --init --recursive
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install usd-core redis pytest
```

Optional heavier setup:

```powershell
.\setup_env.ps1
```

### Start Redis

Use whatever local Redis workflow you prefer, but the demo expects:

- host: `localhost`
- port: `6379`
- database: `0`

### Start the demo processes

Open separate terminals in the repo root.

Terminal 1, start the workers:

```powershell
.\.venv\Scripts\python.exe .\worker_runner.py
```

Terminal 2, optional manager/alerts:

```powershell
.\.venv\Scripts\python.exe .\manager_runner.py
```

Terminal 3, run the orchestrator:

```powershell
.\.venv\Scripts\python.exe .\run_pipeline_demo.py
```

### Expected outputs

If the demo completes, you should see:

- extracted frames under `data/extracted_frames/`
- reconstruction output under `data/reconstruction/`
- final USD scene at `data/digital_twin_scene.usda`

Important detail:

- if `ns-process-data` / COLMAP is available, the reconstruction agent uses it
- if it is missing or fails, the agent creates a placeholder `data/reconstruction/cloud.usda` so the demo can still finish and emit `data/digital_twin_scene.usda`

## Single Launcher And `.exe`

If you do not want separate worker, manager, and orchestrator terminals, use the single launcher:

```powershell
.\.venv\Scripts\python.exe .\demo_launcher.py
```

What it does:

- runs the local pipeline in one process
- extracts frames from the selected source video
- attempts Nerfstudio/COLMAP reconstruction when available
- falls back to a placeholder reconstruction when those tools are missing
- composes the final USD stage
- writes outputs under `data/jobs/`

For a non-UI verification run:

```powershell
.\.venv\Scripts\python.exe .\demo_launcher.py --headless
```

To build a Windows executable:

```powershell
.\build_demo_exe.ps1
```

That produces:

- `dist/vaultwares-studio-demo.exe`

Notes:

- when run from source, the launcher writes output under `data/demo_outputs/`
- the packaged app writes output next to the `.exe` in its local `data/` folder
- the packaged app includes `test_input.mp4`
- `ffmpeg` still needs to be available on `PATH`
- Nerfstudio/COLMAP are optional; when absent, the launcher writes a valid placeholder reconstruction and still emits the final `.usd` scene

## More Detail

See [TESTING.md](TESTING.md) for step-by-step testing instructions and troubleshooting.

## Repository Contents

| File | Purpose |
| ------ | --------- |
| [`usd_smoke.py`](usd_smoke.py) | Minimal USD artifact generator used by the smoke test |
| [`tests/test_usd_smoke.py`](tests/test_usd_smoke.py) | Smoke test that creates and validates a real `.usda` file |
| [`REPORT.md`](REPORT.md) | Research notes and architectural direction |
| [`TECHNICAL_SPECS.md`](TECHNICAL_SPECS.md) | Detailed dependency and hardware notes |
| [`requirements.txt`](requirements.txt) | Broad dependency list for the larger research pipeline |

## Features

- **Remote GPU Reconstruction:** COLMAP SfM + Nerfstudio splatfacto training dispatched as HF Jobs; logs stream in real time; cost requires explicit confirmation before any network call.
- **Resume Failed Jobs:** `--resume-job` reuses an already-uploaded `frames.zip` so a credit-killed run restarts without re-uploading.
- **Quality Presets:** Draft / Standard / Refine / High, each with a fixed flavor + iteration budget + cost estimate.
- **Interactive Splat Viewport:** QWebEngineView + GaussianSplats3D, orbit/fly controls, axis gizmo, infinite scroll-zoom.
- **Camera Authoring:** Walk-pattern trajectories, captured camera poses → USD time-sampled xformOps.
- **Robot Lab (M3):** 2.5D occupancy grid from splat preview cloud, BFS geodesic field, gymnasium PointNav wrapper for SB3 PPO.
- **Local-first OpenUSD Generation:** Every stage writes deterministic placeholder-safe USD/PLY artifacts so the pipeline can complete even without heavy tools.
- **Fallback-safe Reconstruction:** If the remote runner fails or is unconfigured, the stage falls back to a local quick path then placeholder outputs — the job never hard-fails.
