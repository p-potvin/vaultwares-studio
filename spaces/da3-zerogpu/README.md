---
title: DA3 Reconstruction Console
emoji: 🧭
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: 3.10
app_file: app.py
fullWidth: true
startup_duration_timeout: 30m
---

# DA3 Reconstruction Console

Private, experimental console for markerless cellphone-video pose and depth
reconstruction. Upload one video; CPU prepares it, a bounded ZeroGPU call runs
DA3 Streaming, then CPU creates a downloadable artifact bundle. The initial
build downloads public model files at Space startup; build-time preloading is
re-enabled after the runtime is verified.

The default high-quality preset requests a 48 GB ZeroGPU allocation for at most
30 minutes. It runs DA3-LARGE at 672×378 with 90-frame windows and 45-frame
overlap. It is experimental: the console reports observed timing and memory so
the preset can be calibrated from real captures.

This Space deliberately excludes Splatfacto training, USD authoring, 3D viewing,
camera paths, mesh work, Cosmos, history, and multi-user workflows.
