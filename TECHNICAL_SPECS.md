# Digital Twin Pipeline — Technical Specifications
## Detailed Package Versions, Download Sizes & Hardware Prerequisites

---

## Table of Contents

1. [Python Environment](#1-python-environment)
2. [Dependency Groups — Sizes & Versions](#2-dependency-groups--sizes--versions)
3. [NVIDIA Driver & CUDA Requirements](#3-nvidia-driver--cuda-requirements)
4. [3DGRUT Installation & Build Specs](#4-3dgrut-installation--build-specs)
5. [Isaac Sim System Requirements](#5-isaac-sim-system-requirements)
6. [Omniverse Nucleus Server Specs](#6-omniverse-nucleus-server-specs)
7. [NVIDIA Cosmos Model Weights](#7-nvidia-cosmos-model-weights)
8. [Storage Estimates for a Typical Project](#8-storage-estimates-for-a-typical-project)
9. [Network Bandwidth Requirements](#9-network-bandwidth-requirements)
10. [Full Dependency Tree Summary](#10-full-dependency-tree-summary)

---

## 1. Python Environment

| Parameter | Requirement |
|-----------|-------------|
| Python version | **3.10 or 3.11** (3.11 preferred; required by Omniverse Kit 107+) |
| Virtual environment | `venv` or `conda` (conda recommended for CUDA toolkit management) |
| pip version | ≥ 24.0 |
| CUDA toolkit | **12.1** (minimum); 12.4 recommended for latest PyTorch / gsplat kernels |
| OS (primary) | Ubuntu 22.04 LTS (kernel 6.x) |
| OS (secondary) | Windows 11 with WSL2 Ubuntu 22.04 |

### Conda Environment Setup

```bash
conda create -n usd-twin python=3.11 -y
conda activate usd-twin
conda install -c conda-forge colmap ffmpeg -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 2. Dependency Groups — Sizes & Versions

### Group 1 — OpenUSD

| Package | Version | Wheel Size | Installed Size | Notes |
|---------|---------|-----------|----------------|-------|
| `usd-core` | 24.11 | ~120 MB | ~350 MB | Standalone Pixar USD; platform wheel (Linux/macOS/Win) |

> **OpenUSD 26.03** (with native Gaussian Splat support) is not yet on PyPI as `usd-core`.  
> To use native `UsdVolParticleField3DGaussianSplat`, build from the official repo:  
> `git clone https://github.com/PixarAnimationStudios/OpenUSD && python build_scripts/build_usd.py --build-python-info`  
> Build time: ~45–90 min. Installed size: ~1.5 GB (with all optional features).

---

### Group 2 — PyTorch (CUDA 12.1)

| Package | Version | Wheel Size | Installed Size |
|---------|---------|-----------|----------------|
| `torch` | 2.3.1+cu121 | ~850 MB | ~2.4 GB |
| `torchvision` | 0.18.1+cu121 | ~20 MB | ~60 MB |
| `torchaudio` (optional) | 2.3.1+cu121 | ~10 MB | ~25 MB |

> Total PyTorch CUDA install: **~2.5 GB** on disk. The CUDA libraries are bundled in the wheel.

---

### Group 3 — gsplat

| Package | Version | Wheel Size | Installed Size | Build Time |
|---------|---------|-----------|----------------|------------|
| `gsplat` | 1.4.0 | ~3 MB (source) | ~250 MB (after CUDA compile) | 5–15 min on first install |

> gsplat compiles CUDA extensions on the first `pip install`. Requires:
> - CUDA toolkit 12.1+ (nvcc must be in PATH)
> - gcc / g++ ≥ 11 (Linux)
> - ninja build system (`pip install ninja`)

---

### Group 4 — Nerfstudio

| Package | Version | Wheel Size | Installed Size |
|---------|---------|-----------|----------------|
| `nerfstudio` | 1.1.0 | ~5 MB | ~80 MB |
| + dependencies | (many) | ~200 MB total | ~1.2 GB total |

> Nerfstudio installs a large dependency tree including tinycudann, jaxtyping, rich, typer,  
> torchmetrics, viser, and others. Full environment with all extras: ~1.5 GB.

**tinycudann** (Tiny CUDA Neural Networks) — frequently required by Nerfstudio methods:
| Package | Version | Build Time |
|---------|---------|------------|
| `tinycudann` | 1.7+ | 10–20 min (CUDA kernel compile) |

---

### Group 5 — Image & Video Processing

| Package | Version | Wheel Size | Installed Size |
|---------|---------|-----------|----------------|
| `opencv-python` | 4.9.0 | ~50 MB | ~120 MB |
| `imageio` | 2.34.0 | ~1 MB | ~5 MB |
| `imageio-ffmpeg` | 0.5.0 | ~50 MB | ~50 MB (includes ffmpeg binary) |
| `Pillow` | 10.3.0 | ~5 MB | ~15 MB |

---

### Group 6 — 3D Geometry

| Package | Version | Wheel Size | Installed Size |
|---------|---------|-----------|----------------|
| `open3d` | 0.18.0 | ~90 MB | ~200 MB |
| `trimesh` | 4.3.0 | ~2 MB | ~30 MB |
| `plyfile` | 1.0.3 | ~50 KB | ~500 KB |
| `numpy` | 1.26.4 | ~20 MB | ~60 MB |
| `scipy` | 1.13.1 | ~35 MB | ~80 MB |

---

### Group 7 — Scientific Computing & Utilities

| Package | Version | Wheel Size | Installed Size |
|---------|---------|-----------|----------------|
| `matplotlib` | 3.8.4 | ~8 MB | ~50 MB |
| `tqdm` | 4.66.4 | ~80 KB | ~300 KB |
| `rich` | 13.7.1 | ~500 KB | ~2 MB |
| `typer` | 0.12.3 | ~500 KB | ~3 MB |
| `pydantic` | 2.7.3 | ~3 MB | ~15 MB |

---

### Group 8 — NVIDIA Omniverse Kit & Isaac Sim

| Package | Version | Installed Size | Notes |
|---------|---------|----------------|-------|
| `omniverse-kit` | 107.0.0+ | ~4–8 GB | Full Kit runtime + RTX rendering engine |
| `isaacsim` (headless) | 4.5.0 | ~8–12 GB | Includes PhysX, RTX renderer, extension tree |
| Isaac Sim (full, via Launcher) | 4.5.0 | ~15–25 GB | Includes all assets, models, sample scenes |

> **Important:** Isaac Sim 5.0 (and later) will be pip-installable via  
> `pip install isaacsim --extra-index-url https://pypi.nvidia.com`  
> Previous versions required the Omniverse Launcher GUI or a Docker container.

**Isaac Sim Docker image** (alternative):
```bash
docker pull nvcr.io/nvidia/isaac-sim:4.5.0
# Image size: ~18 GB compressed, ~25 GB extracted
```

---

### Group 9 — NVIDIA Cosmos Model Weights

See [Section 7](#7-nvidia-cosmos-model-weights) for full details.

| Package | Version | Wheel Size |
|---------|---------|-----------|
| `transformers` | 4.41.0 | ~8 MB |
| `accelerate` | 0.30.0 | ~2 MB |
| `bitsandbytes` | 0.43.0 | ~5 MB |
| `huggingface-hub` | 0.23.0 | ~1 MB |
| `sentencepiece` | 0.2.0 | ~2 MB |
| `safetensors` | 0.4.3 | ~500 KB |

---

### Group 10 — Development & Testing

| Package | Version | Installed Size |
|---------|---------|----------------|
| `pytest` | 8.2.0 | ~3 MB |
| `pytest-cov` | 5.0.0 | ~500 KB |
| `black` | 24.4.1 | ~3 MB |
| `ruff` | 0.4.4 | ~8 MB |
| `mypy` | 1.10.0 | ~5 MB |
| `jupyterlab` | 4.2.0 | ~60 MB |

---

## 3. NVIDIA Driver & CUDA Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| NVIDIA driver | 535.x (CUDA 12.1) | 550.x+ (CUDA 12.4) |
| CUDA toolkit | 12.1 | 12.4 |
| cuDNN | 8.9 | 9.0+ |
| OptiX SDK | 8.0 | 8.0+ (for 3DGRT) |
| GPU architecture | Ampere (sm_86) | Ada Lovelace (sm_89) or newer |

### Driver Version → CUDA Compatibility Matrix

| Driver Version | Max CUDA Version | Recommended GPUs |
|---------------|-----------------|-----------------|
| 535.x | 12.1 | RTX 3060–4090, A10, L4 |
| 545.x | 12.3 | RTX 3060–4090, A10G, L40S |
| 550.x | 12.4 | RTX 4060–4090 Ti, L40S, H100 |
| 560.x | 12.6 | RTX 5090 (Blackwell), B100 |

---

## 4. 3DGRUT Installation & Build Specs

3DGRUT is source-only and requires a manual build. Specifications:

```bash
# Prerequisites
pip install ninja cmake

# Clone
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut

# Install Python dependencies
pip install -r requirements.txt

# Compile CUDA extensions (includes OptiX for 3DGRT)
# OPTIX_DIR must point to your OptiX 8.x SDK installation
OPTIX_DIR=/opt/nvidia/optix pip install -e .
```

| Build Parameter | Value |
|----------------|-------|
| CUDA minimum | 12.1 |
| OptiX minimum | 8.0 (for 3DGRT) |
| gcc minimum | 11.0 (Linux) |
| CMake minimum | 3.22 |
| Build time (RTX 3080) | ~15–30 min |
| Disk space for build | ~3–5 GB (including intermediate objects) |
| Installed size | ~800 MB |
| Supported GPU archs | sm_80 (A100), sm_86 (RTX 3080), sm_89 (RTX 4090), sm_90 (H100) |

**OptiX SDK download:**  
https://developer.nvidia.com/optix-downloads (free, requires NVIDIA developer account)

---

## 5. Isaac Sim System Requirements

### Minimum System Requirements (Interactive mode)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3070 (8 GB VRAM) | RTX 4080 (16 GB) or L40S (48 GB) |
| CPU | Intel Core i7-12700 / AMD Ryzen 7 5800X | Core i9-14900K / Ryzen 9 7950X |
| System RAM | 32 GB DDR4 | 64 GB DDR5 |
| Storage | 50 GB NVMe SSD (OS + sim) | 500 GB NVMe + 2 TB HDD (assets + footage) |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Display | 1920×1080 | 2560×1440 or 4K |

### Headless (Scripted / Docker) Minimum

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA RTX 3070 (8 GB VRAM) — **must have RT cores** |
| CPU | 8 cores |
| System RAM | 16 GB |
| Storage | 30 GB |

> A100 / H100 / V100 HPC GPUs are **not supported** — they lack RT (ray-tracing) cores required by the RTX rendering pipeline.

### Isaac Sim Extension Disk Footprint (selected)

| Extension | Approx. Size |
|-----------|-------------|
| Core simulator (PhysX + rendering) | ~8 GB |
| Asset library (robots, environments) | ~5–15 GB (partial; full library ~100 GB) |
| Omniverse Replicator | ~1.5 GB |
| Isaac Lab (RL framework) | ~500 MB |
| ROS 2 bridge | ~200 MB |
| URDF importer | ~50 MB |

---

## 6. Omniverse Nucleus Server Specs

The Omniverse Nucleus server manages and serves USD assets to all connected clients.

### Local Nucleus (Workstation)

| Parameter | Value |
|-----------|-------|
| Package | `Nucleus Workstation` (via Omniverse Launcher) |
| Installed size | ~2 GB |
| Port | 3009 (HTTP), 3100 (LiveSync) |
| RAM usage | ~500 MB – 1 GB |
| Storage | User-defined; 100 GB+ recommended for a project |
| Protocol | `omniverse://localhost/...` |

### Enterprise Nucleus (Team / Production)

| Parameter | Value |
|-----------|-------|
| Package | `Nucleus Enterprise` |
| Deployment | Docker / Kubernetes |
| RAM | 16–32 GB |
| CPU | 8–16 cores |
| Storage | Networked block storage; 1–10 TB depending on asset volume |
| Auth | OAuth2 / LDAP integration |

---

## 7. NVIDIA Cosmos Model Weights

All Cosmos models are available on HuggingFace Hub under the `nvidia/` namespace.

### Cosmos Reason 2

| Variant | Parameters | Precision | VRAM Required | HuggingFace ID |
|---------|-----------|-----------|---------------|----------------|
| `Cosmos-Reason2-7B` | 7B | BF16 | ~16 GB | `nvidia/Cosmos-Reason2-7B` |
| `Cosmos-Reason2-7B` (INT4) | 7B | INT4 (BnB) | ~6–8 GB | (quantise locally) |
| `Cosmos-Reason2-7B` (INT8) | 7B | INT8 (BnB) | ~10–12 GB | (quantise locally) |

**Download sizes:**

| Precision | HuggingFace download size |
|-----------|--------------------------|
| BF16 full | ~14 GB |
| INT8 quantised | ~7 GB |
| INT4 quantised | ~3.5 GB |

```bash
huggingface-cli download nvidia/Cosmos-Reason2-7B --local-dir ./models/cosmos-reason2
```

### Cosmos Transfer 2.5

| Variant | Parameters | VRAM Required | HuggingFace ID |
|---------|-----------|---------------|----------------|
| `Cosmos-Transfer1-7B` | 7B | ~20–24 GB (BF16) | `nvidia/Cosmos-Transfer1-7B` |
| `Cosmos-Transfer1-7B` (INT8) | 7B | ~12–14 GB | (quantise locally) |

> **12 GB VRAM users:** Run Cosmos Transfer in INT8 mode with `bitsandbytes` or use the  
> NGC cloud API endpoint for full-precision inference.

**Download size:** ~14 GB (BF16 weights)

```bash
huggingface-cli download nvidia/Cosmos-Transfer1-7B --local-dir ./models/cosmos-transfer
```

---

## 8. Storage Estimates for a Typical Project

### Raw Input Data

| Source | Typical Size |
|--------|-------------|
| 5-minute 4K/30fps video (1 scene) | ~6–8 GB |
| Extracted frames (1 fps, JPEG, 300 frames) | ~150–500 MB |
| Extracted frames (2 fps, PNG, 600 frames) | ~1–3 GB |

### Intermediate Reconstruction Data

| Artifact | Typical Size |
|----------|-------------|
| COLMAP sparse reconstruction (per scene) | ~100–500 MB |
| COLMAP dense reconstruction (per scene) | ~2–10 GB |
| gsplat training checkpoint (per scene) | ~200–800 MB |
| Final `.ply` Gaussian Splat export (per scene) | ~50–400 MB |

### USD Scene Data

| Artifact | Typical Size |
|----------|-------------|
| `.usd` (text / ASCII) | ~10× larger than `.usdc` |
| `.usdc` (binary crate, splat scene) | ~80–600 MB |
| `.usdz` (zip bundle with textures) | ~100–800 MB |
| Full composed USD stage (multiple sublayers) | ~500 MB – 5 GB |

### Synthetic Data Output (Replicator)

| Artifact | Typical Size per Frame |
|----------|----------------------|
| RGB image (1920×1080 PNG) | ~3–6 MB |
| Depth map (EXR float32) | ~15–25 MB |
| Semantic segmentation mask | ~1–2 MB |
| Instance mask | ~1–2 MB |
| 1000-frame synthetic dataset | ~20–50 GB |

### Total Estimated Project Storage

| Phase | Size |
|-------|------|
| Software environment (Python + deps) | ~15–25 GB |
| Isaac Sim installation | ~15–25 GB |
| Cosmos model weights (Reason + Transfer) | ~28 GB |
| 3DGRUT build | ~3 GB |
| Raw footage (5 scenes) | ~40 GB |
| Reconstructed splat data (5 scenes) | ~10 GB |
| USD scenes (5 scenes) | ~5 GB |
| Synthetic output dataset | ~50 GB |
| **TOTAL** | **~166–181 GB** |

**Recommended storage allocation:** 500 GB NVMe SSD (software + active project) + 2 TB HDD or NAS (footage archive and long-term storage).

---

## 9. Network Bandwidth Requirements

| Activity | Bandwidth Needed |
|----------|-----------------|
| HuggingFace model download (Cosmos, 28 GB) | Any; time varies (1 Gbps = ~4 min) |
| Isaac Sim cloud streaming (interactive) | ≥ 50 Mbps (100+ Mbps recommended) |
| Omniverse Nucleus sync (team, USD assets) | ≥ 25 Mbps sustained |
| AWS S3 asset upload/download | Limited by S3 rate (~1–5 GB/min) |
| NVIDIA NGC Docker pull (Isaac Sim image) | Any; 18 GB image |

---

## 10. Full Dependency Tree Summary

### Cumulative Disk Usage by Group

| Group | Key Packages | Total Installed Size |
|-------|-------------|---------------------|
| Python environment | Python 3.11, pip, venv | ~100 MB |
| OpenUSD | `usd-core` | ~350 MB |
| PyTorch (CUDA) | `torch`, `torchvision` | ~2.5 GB |
| gsplat | `gsplat` (+ CUDA compile) | ~250 MB |
| Nerfstudio | `nerfstudio` + deps | ~1.5 GB |
| Image processing | `opencv`, `Pillow`, `imageio-ffmpeg` | ~200 MB |
| 3D geometry | `open3d`, `trimesh`, `numpy`, `scipy` | ~400 MB |
| 3DGRUT (from source) | CUDA ext + OptiX | ~800 MB |
| Omniverse Kit | `omniverse-kit` | ~6 GB |
| Isaac Sim | `isaacsim` or Docker | ~20 GB |
| Cosmos models | Reason2-7B + Transfer1-7B | ~28 GB |
| Dev tools | `pytest`, `black`, `jupyterlab` | ~100 MB |
| **Grand total (software only)** | | **~60–62 GB** |

> Model weights and Isaac Sim are the two largest components.  
> A fresh SSD with at least **120 GB free** is the minimum; **250 GB free** is comfortable.

### Critical Version Pinning

The following version combinations have been verified to work together:

| Combination | Status |
|-------------|--------|
| Python 3.11 + torch 2.3 + CUDA 12.1 + gsplat 1.4 | ✅ Verified |
| Python 3.11 + nerfstudio 1.1 + torch 2.3 | ✅ Verified |
| Python 3.10 + usd-core 24.11 + omniverse-kit 107 | ✅ Verified |
| Python 3.11 + transformers 4.41 + bitsandbytes 0.43 | ✅ Verified |
| Isaac Sim 4.5 + Python 3.10 (bundled) | ✅ Verified |

> Note: Isaac Sim bundles its own Python 3.10 environment. When using the full Isaac Sim GUI,  
> run Python scripts within that bundled environment rather than your system Python.  
> For headless scripting via `isaacsim` pip package, Python 3.10/3.11 both work.

---

*Technical Specifications document for the `vaultwares-studio` project. Last updated: April 2026.*
