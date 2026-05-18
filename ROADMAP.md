# Roadmap: USD Digital Twin Studio

## Milestone 1: Working Local App (current)
Goal: Provide a desktop workflow that can create, resume, run, and inspect local digital-twin jobs without Redis.

Current baseline:

- PySide desktop shell in `gui_app.py`
- reusable job execution in `vaultwares_studio/`
- resumable manifests under `data/jobs/`
- fallback-safe reconstruction outputs when heavyweight tools are missing
- exported USD, camera previews, and walkthrough MP4

## Milestone 2: Gaussian Splat Pipeline
Goal: Successfully reconstruct a 3D scene from video and export to PLY with COLMAP / Nerfstudio / gsplat when those tools are installed.

## Milestone 3: Native OpenUSD Assets
Goal: Convert PLY to OpenUSD using the 26.03 schema and establish an asset library.

## Milestone 4: Real-time Simulation & Robotics
Goal: Ingest USD assets into Isaac Sim and validate physics/camera navigation.

## Milestone 5: Cosmos AI Enhancement
Goal: Integrate Cosmos models for automated annotation and photorealistic domain transfer.
