# Testing vaultwares-studio

This repo has three testing layers:

1. A fast, local pytest suite (83 tests) covering pipeline stages, runners, splat I/O, presets, camera paths, robot lab, and resume-job wiring. No HF token required.
2. The desktop Digital Twin Studio app and headless pipeline runner.
3. Live HF Jobs verification (requires HF PRO token with credit balance).

## Recommended Test

Run this from the repo root.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install usd-core pytest redis PySide6 PySide6-Fluent-Widgets Pillow
python -m pytest -s
```

### What success looks like

- `pytest` exits with code `0`
- the test summary reports **83 tests passed**
- `data/test_outputs/smoke_scene.usda` exists
- the console prints the generated USD path

## Remote Compute Verification (HF Jobs)

Requires an HF PRO account with credit balance and the token set in the OS keyring (or `HF_TOKEN` env var).

### Headless new run

```powershell
.venv\Scripts\python.exe tools\headless_remote_run.py --preset draft --yes
```

Expected output: `[run] reconstruction degraded: False, gaussians: <N>, runner: hf-jobs`

### Resume a credit-killed job

```powershell
# Reuse already-uploaded frames.zip — no re-upload, no frame extraction:
.venv\Scripts\python.exe tools\headless_remote_run.py --resume-job local-run-20260615-065732 --yes
```

Expected: `[resume] resuming local-run-20260615-065732 | preset=refine | reusing HF frames`

### Recover outputs from a completed job

```powershell
.venv\Scripts\python.exe tools\recover_remote_recon.py --job local-run-20260613-211202
```

## Generated Artifact

The smoke test writes:

- `data/test_outputs/smoke_scene.usda`

You can generate the same file without pytest:

```powershell
.\.venv\Scripts\python.exe .\usd_smoke.py
```

## App Pipeline Verification

Run the local app pipeline without opening the GUI:

```powershell
.\.venv\Scripts\python.exe .\demo_launcher.py --headless
```

What success looks like:

- the command exits with code `0`
- the console prints a `data/jobs/<job-id>/manifest.json` path
- the console prints a walkthrough video path
- the manifest state is `complete`
- outputs exist under `frames/`, `reconstruction/`, `usd/`, `camera_previews/`, and `deliverables/`

The GUI entrypoint is:

```powershell
.\.venv\Scripts\python.exe .\gui_app.py
```

Use `Open Latest Job` in the app to reopen the most recent verified run.

## Why The Default Test Scope Is Limited

`pytest.ini` intentionally limits discovery to:

- `tests/`
- `vaultwares-adk/omx_integration/tests/`

It skips:

- `.venv/` because that contains third-party package tests
- `cosmos-reason2/` because it is an upstream submodule with additional platform-specific requirements
- `vaultwares-themes/` because it is not part of the Python test surface for this repo

Without that scoping, a normal `pytest` run tries to execute unrelated vendored tests and fails for reasons that have nothing to do with `vaultwares-studio`.

## About The Bigger Pipeline

These files are still present for the advanced Redis-backed orchestration path:

- `run_pipeline_demo.py`
- `worker_runner.py`
- `manager_runner.py`

They are better treated as advanced demos than as the default test story. They may require:

- a running Redis instance
- ffmpeg
- Nerfstudio
- COLMAP
- extra GPU-heavy dependencies

If you only want to verify that the repo can author OpenUSD output locally, the smoke test is the right tool. If you want to verify the app, use `demo_launcher.py --headless`.

## Troubleshooting

If `from pxr import Usd` fails:

- install `usd-core` into the active environment
- confirm you are using the same Python interpreter for both install and test commands

If `pytest` tries to run submodule or `.venv` tests anyway:

- make sure you are running from the repo root
- confirm `pytest.ini` is present

If the USD file is missing after a green test run:

- check whether the repo root is writable
- rerun with `python -m pytest -s tests/test_usd_smoke.py`
