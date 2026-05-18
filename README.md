# usd-playground

`usd-playground` is a local-first OpenUSD desktop app for turning a room video into a resumable digital-twin job with generated USD, camera previews, a walkthrough video, and optional VaultWares workflow export.

If you are just trying to answer, "Does this repo work on my machine?", start with the smoke test. It produces a real `.usda` file you can inspect. If you want the working app, run the desktop studio.

## What You Can Run Today

- `python gui_app.py` opens the desktop Digital Twin Studio app.
- The app creates and resumes jobs under `data/jobs/`.
- The full app run writes a manifest, extracted frames, USD stage, camera previews, and walkthrough MP4.
- `pytest` runs the repo's supported tests.
- The smoke test writes `data/test_outputs/smoke_scene.usda`.
- `python usd_smoke.py` generates the same USD artifact without pytest.

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

Default `pytest` is intentionally scoped to the tests that belong to `usd-playground` itself:

- `tests/test_usd_smoke.py`
- `vaultwares_agentciation/omx_integration/tests/`

It does **not** recurse into:

- `.venv/`
- `cosmos-reason2/`
- `vault-themes/`

That is deliberate. `cosmos-reason2` is an upstream submodule with extra platform-specific dependencies and should not be treated as the default local smoke-test target for this repo.

## Full Pipeline Reality Check

The repo also contains an agent-driven pipeline:

- `worker_runner.py`
- `manager_runner.py`
- `run_pipeline_demo.py`

That flow is now the legacy advanced orchestration path. It depends on extra tooling such as Redis, ffmpeg, Nerfstudio/COLMAP, and the vendored `vaultwares_agentciation` framework. The everyday app path is `gui_app.py`, which uses `studio_core` directly and does not require Redis.

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

- `dist/usd-playground-demo.exe`

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

- **Local-first OpenUSD Generation:** Turn room videos into resumable digital-twin jobs emitting valid `.usda` files.
- **Robust App Mode:** Desktop studio app using fallback-safe reconstruction that writes deterministic placeholders when heavy tools are missing.
- **Redis-backed Orchestration Pipeline:** Agent-driven tasks (via ExtrovertAgent base class) to extract frames, reconstruct 3D spaces, and compose USDs.
- **Video Processing Pipeline:** Integrated FFmpeg support for sampling, trimming, resizing, and frame-level processing.
- **Fallback-safe Reconstruction:** Sparse reconstruction (COLMAP) and Gaussian splatting (gsplat / Nerfstudio) with built-in placeholder fallback mechanism.
