# Digital Twin Generation with OpenUSD & NVIDIA Omniverse
## Technical Research Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack Overview](#2-technology-stack-overview)
3. [Generating USD Data from Real-Life Footage](#3-generating-usd-data-from-real-life-footage)
   - [3.1 Capture Pipeline (Single-Person Workflow)](#31-capture-pipeline-single-person-workflow)
   - [3.2 3DGRUT: 3DGUT and 3DGRT](#32-3dgrut-3dgut-and-3dgrt)
   - [3.3 gsplat](#33-gsplat)
   - [3.4 Nerfstudio + COLMAP](#34-nerfstudio--colmap)
   - [3.5 PLY → USD Conversion](#35-ply--usd-conversion)
4. [Hardware Requirements & Feasibility](#4-hardware-requirements--feasibility)
   - [4.1 Consumer Hardware (12 GB VRAM)](#41-consumer-hardware-12-gb-vram)
   - [4.2 Cloud GPUs](#42-cloud-gpus)
   - [4.3 Recommended Configuration Summary](#43-recommended-configuration-summary)
5. [OpenUSD & NVIDIA Omniverse Libraries](#5-openusd--nvidia-omniverse-libraries)
6. [Integrating USD into Isaac Sim](#6-integrating-usd-into-isaac-sim)
   - [6.1 Local Isaac Sim Deployment](#61-local-isaac-sim-deployment)
   - [6.2 Cloud Isaac Sim Deployment](#62-cloud-isaac-sim-deployment)
   - [6.3 Comparison: Local vs. Cloud](#63-comparison-local-vs-cloud)
7. [Camera Navigation in the Digital Twin](#7-camera-navigation-in-the-digital-twin)
8. [Robotic Systems Integration](#8-robotic-systems-integration)
9. [NVIDIA Cosmos Models (Reason & Transfer)](#9-nvidia-cosmos-models-reason--transfer)
10. [Recommended End-to-End Workflow](#10-recommended-end-to-end-workflow)
11. [Conclusion & Next Steps](#11-conclusion--next-steps)

---

## 1. Project Overview

This project establishes the foundational baseline for generating **Digital Twins** from real-life footage using **OpenUSD** (Universal Scene Description) and the **NVIDIA Omniverse** ecosystem. The goal is to produce simulation-ready scenes that can be loaded into **NVIDIA Isaac Sim** for robotic testing, camera navigation, and synthetic data generation — leveraging locally available Cosmos Reason and Transfer models.

The pipeline has three major phases:

| Phase | Description |
|-------|-------------|
| **Capture & Reconstruct** | Video/image capture → 3D Gaussian Splatting → `.ply` export |
| **Convert & Author** | `.ply` → OpenUSD (`.usd` / `.usda` / `.usdz`) → scene composition |
| **Simulate** | Isaac Sim ingestion → camera navigation → robotic integration → Cosmos augmentation |

---

## 2. Technology Stack Overview

| Layer | Technology | Role |
|-------|-----------|------|
| 3D Reconstruction | 3DGRUT (3DGUT + 3DGRT), gsplat, Nerfstudio | Convert footage to 3D Gaussian Splat assets |
| Scene Description | OpenUSD 26.03+ (`usd-core`) | Universal 3D scene format; native splat support |
| Omniverse Platform | NVIDIA Omniverse Kit 107+ | Real-time collaboration, rendering, asset management |
| Simulation | NVIDIA Isaac Sim 4.5 / 5.0 | Physics, sensors, synthetic data, robot testing |
| World Models | NVIDIA Cosmos Reason 2, Cosmos Transfer 2.5 | Scene annotation, photorealistic transfer, scenario generation |
| Asset Server | Omniverse Nucleus (local or cloud) | USD asset versioning and live collaboration |
| Camera Tracking | COLMAP 3.10+ | Structure-from-Motion for pose estimation |
| Robot Description | URDF / MJCF → USD | Standard robotic model formats, auto-converted by Isaac |

---

## 3. Generating USD Data from Real-Life Footage

### 3.1 Capture Pipeline (Single-Person Workflow)

A single person can realistically produce high-quality 3D assets using the following capture approach:

**Equipment needed:**
- Smartphone with a good camera (iPhone 14+ or Android flagship) **or** a mirrorless camera
- Optional but recommended: a simple handheld gimbal stabiliser for smooth orbits
- Good lighting: indoor with multiple light sources or outdoor with overcast skies (avoids harsh shadows)

**Capture procedure:**
1. **Object / scene orbits** — Walk slowly around the subject in a full circle, then repeat at a lower and higher angle (gives full spherical coverage). Aim for 3–5 minutes of continuous video at 4K/30fps or 1080p/60fps.
2. **Static subject** — For environments (rooms, industrial spaces), walk slowly through the space capturing every surface.
3. **Frame extraction** — Extract 1–2 fps for a short walkthrough, or 100–300 key frames for a subject orbit. Use `ffmpeg -i input.mp4 -vf fps=1 frames/%04d.jpg`.
4. **Remove motion blur** — Avoid footage recorded with digital stabilisation enabled as it distorts the real camera path and breaks COLMAP pose estimation.

> **Recommended capture software for video → frames:**  
> `ffmpeg` (free, open-source), or `ns-process-data video` from Nerfstudio (wraps ffmpeg + COLMAP automatically).

---

### 3.2 3DGRUT: 3DGUT and 3DGRT

**Repository:** [github.com/nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut)  
**License:** Apache 2.0  
**Published:** CVPR 2025 / SIGGRAPH Asia 2024

3DGRUT is NVIDIA's open-source hybrid rendering framework for 3D Gaussian Splatting. It unifies two complementary approaches:

#### 3DGUT (3D Gaussian Unscented Transform)
- Extends standard Gaussian splatting by replacing the linear (EWA-splatting) approximation with the **Unscented Transform**, which handles non-linear camera models such as fisheye, equirectangular, and rolling-shutter sensors — all at **rasterisation speeds**.
- Does **not** require dedicated RT cores; runs on any CUDA 12+ GPU.
- Ideal as the primary ray renderer for real-time digital twin previews.
- Output: `.ply` with all Gaussian attributes (position, orientation, scale, colour, opacity, spherical harmonics).

#### 3DGRT (3D Gaussian Ray Tracing)
- Uses **NVIDIA OptiX** for physically accurate ray tracing of Gaussian particles, supporting secondary effects: reflections, refractions, shadows, and global illumination.
- **Requires** RTX hardware (RT cores) for OptiX acceleration. Runs on RTX 30/40 series and above.
- Significantly slower than 3DGUT but produces photorealistic results suitable for synthetic data generation.

#### 3DGRUT (Hybrid)
- Combines 3DGUT for **primary rays** (speed) and 3DGRT for **secondary rays** (realism).
- The recommended configuration for high-quality digital twin rendering where both efficiency and photorealism matter.

**Integration with gsplat:** As of 2025, 3DGUT has been merged into the `gsplat` library (via Nerfstudio), enabling a single install to cover training, rasterisation, and export.

---

### 3.3 gsplat

**Repository:** [github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)  
**Documentation:** [docs.gsplat.studio](https://docs.gsplat.studio)  
**License:** Apache 2.0

`gsplat` is the GPU-accelerated, differentiable rasterisation library for 3D Gaussian Splatting. Key features relevant to this project:

- **Differentiable**: enables gradient-based optimisation of all Gaussian parameters from images.
- **Fast CUDA kernels**: state-of-the-art training performance (typically 30 min to 2 hr for a scene on an RTX 3080+).
- **3DGUT integration**: non-linear camera models, rolling-shutter support.
- **USD export**: output is `.ply`; downstream conversion to USD uses the OpenUSD 26.03 schema or Adobe's USD file-format plugin.
- **Typical VRAM usage**: 8–12 GB for moderate scenes (< 500k Gaussians); 16–24 GB for large or dense scenes.

---

### 3.4 Nerfstudio + COLMAP

**Nerfstudio** ([nerf.studio](https://nerf.studio)) provides the most accessible end-to-end pipeline:

```bash
# Install
pip install nerfstudio

# Process video (runs ffmpeg + COLMAP automatically)
ns-process-data video --data my_scene.mp4 --output-dir ./data/my_scene

# Train a Gaussian Splat model
ns-train splatfacto --data ./data/my_scene

# Export to PLY
ns-export gaussian-splat --load-config outputs/.../config.yml --output-dir ./exports
```

**COLMAP** performs Structure-from-Motion (SfM) to estimate camera poses from the video frames. It is a prerequisite for Nerfstudio's custom data workflows and is called automatically by `ns-process-data`.

COLMAP hardware requirements:
- 8 GB RAM minimum for SfM on 200 frames; 32 GB recommended for larger datasets.
- GPU optional for SfM (runs on CPU), but dense MVS reconstruction greatly benefits from GPU.

---

### 3.5 PLY → USD Conversion

As of **OpenUSD 26.03**, Gaussian Splats are natively supported via the `UsdVolParticleField3DGaussianSplat` schema. Two conversion paths are available:

**Path A — Native OpenUSD 26.03+ (Recommended)**
```bash
# Using the official AOUSD conversion script
python ply_to_usd.py --input scene.ply --output scene.usd
```

**Path B — Adobe USD File-Format Plugin**
```bash
# Adobe's open-source plugin auto-detects Gaussian attributes in PLY
pip install usd-fileformat-plugins  # (or build from source)
usdconvert scene.ply scene.usd
```

**What gets mapped:**
| PLY Attribute | USD Schema Property |
|---------------|---------------------|
| `x, y, z` | Position / `points` |
| `rot_0..rot_3` | Orientation quaternion |
| `scale_0..scale_2` | Scale |
| `opacity` | Opacity |
| `f_dc_0..f_dc_2` | Base colour (SH degree 0) |
| `f_rest_*` | Higher-order spherical harmonics |

---

## 4. Hardware Requirements & Feasibility

### 4.1 Consumer Hardware (12 GB VRAM)

Target GPU: **NVIDIA RTX 3060/3080/4070 (12 GB VRAM)**

| Task | 12 GB VRAM Feasibility | Notes |
|------|----------------------|-------|
| 3DGUT training (small scene, < 300k Gaussians) | ✅ Feasible | ~2–4 hr on RTX 3080 |
| 3DGUT training (large scene, > 1M Gaussians) | ⚠️ Marginal | May OOM; reduce resolution or scene bounds |
| 3DGRT (ray tracing, secondary rays) | ✅ Feasible | Requires RT cores; RTX 3070+ |
| PLY → USD conversion | ✅ Feasible | CPU+RAM bound, no GPU needed |
| Isaac Sim (local, basic scene) | ✅ Feasible | 12 GB is the stated minimum |
| Isaac Sim (large scene + RTX rendering) | ⚠️ Marginal | 16–24 GB recommended; reduce viewport resolution |
| COLMAP SfM on 300 frames | ✅ Feasible | CPU-bound, ~30 min on modern 8-core CPU |
| Cosmos Transfer (inference) | ⚠️ Marginal | Cosmos Transfer 2.5 requires ~16–24 GB VRAM; use cloud or quantised model |
| Cosmos Reason (inference) | ✅ Feasible | VLM inference; 7B parameter model fits in 12 GB at INT4/INT8 |

**Verdict:** 12 GB VRAM is the **functional minimum** for this pipeline. The entire workflow — from video capture to USD scene creation and basic Isaac Sim loading — can be executed on a 12 GB card. However:
- Large scenes (> 500k Gaussians) will require **tiling** or **batch processing** during training.
- Cosmos Transfer at full fidelity requires 16–24 GB; quantised alternatives or cloud inference is recommended.
- Isaac Sim RTX rendering in viewport is most comfortable at 16 GB+; headless/scripted workflows are fine at 12 GB.

**Recommended consumer build for this project:**

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA RTX 4070 Ti or 4080 (16 GB) |
| CPU | AMD Ryzen 9 7950X or Intel Core i9-14900K |
| RAM | 64 GB DDR5 |
| Storage | 2 TB NVMe SSD (PCIe 4.0) + 4 TB HDD for raw footage |
| OS | Ubuntu 22.04 LTS (primary) or Windows 11 with WSL2 |

---

### 4.2 Cloud GPUs

For tasks that exceed consumer hardware (large-scale Cosmos Transfer, multi-scene batch training, enterprise collaboration):

| Provider | Instance | GPU | VRAM | Rate/hr (est.) | Isaac Sim Supported |
|----------|----------|-----|------|----------------|---------------------|
| AWS | `g5.2xlarge` | A10G | 24 GB | ~$1.21 | ✅ Yes |
| AWS | `g6.2xlarge` | L4 | 24 GB | ~$0.98 | ✅ Yes |
| AWS | `g6e.2xlarge` | L40S | 48 GB | ~$1.86 | ✅ Yes (best) |
| Azure | `NV36ads_A10_v5` | A10 | 24 GB | ~$1.50 | ✅ Yes |
| NVIDIA Brev/BCP | Managed | L4/A10/L40S | 24–48 GB | CSP rate +20% | ✅ Managed |
| Lambda Labs | `1x A10` | A10 | 24 GB | ~$0.75 | ✅ (manual setup) |
| CoreWeave | `RTX A6000` | A6000 | 48 GB | ~$0.80 | ✅ Yes |

> ⚠️ **Important:** Isaac Sim requires **RTX (RT-core) GPUs** for its rendering pipeline. A100, H100, and V100 HPC cards do **not** have RT cores and will not run Isaac Sim's graphical interface correctly. Always select L4, L40S, A10, A10G, or equivalent RTX-class GPUs.

**Recommended cloud choice for this project:**  
AWS `g6e.2xlarge` (L40S, 48 GB) — best balance of VRAM, RT core support, and cost for Isaac Sim + Cosmos Transfer workflows.

---

### 4.3 Recommended Configuration Summary

| Scenario | Hardware | Notes |
|----------|----------|-------|
| Development & prototyping | Local RTX 3080/4070 Ti (12–16 GB) | Full workflow feasible with scene-size limits |
| Large scene training | Cloud A10G / L40S | Offload gsplat training for scenes > 500k Gaussians |
| Cosmos Transfer full fidelity | Cloud L40S (48 GB) | Or use quantised INT8 model locally |
| Isaac Sim + full RTX rendering | Local RTX 4080/4090 (16–24 GB) OR Cloud L40S | |
| Enterprise multi-user collaboration | Omniverse Nucleus Cloud | Team-wide USD asset server |

---

## 5. OpenUSD & NVIDIA Omniverse Libraries

### Core USD Libraries

| Library | PyPI Package | Version (2025) | Description |
|---------|-------------|----------------|-------------|
| OpenUSD (standalone) | `usd-core` | 25.05+ | Pixar's standalone USD Python API; no NVIDIA dependencies |
| Omniverse USD APIs | via `omniverse-kit` | Kit 107+ | Full Omniverse extension ecosystem; USD + RTX rendering |
| NVIDIA USD Composer | Desktop app | 2024.2+ | GUI for USD scene authoring (free) |

### Reconstruction & Training Libraries

| Library | Package | Version | Notes |
|---------|---------|---------|-------|
| gsplat | `gsplat` (pip) | 1.4+ | Core 3DGS training & rasterisation |
| 3DGRUT | source only (`nv-tlabs/3dgrut`) | main | Requires CUDA 12+, OptiX 8+ |
| Nerfstudio | `nerfstudio` (pip) | 1.1+ | End-to-end NeRF/splat pipeline |
| COLMAP | system package / conda | 3.10+ | SfM camera pose estimation |

### Omniverse Ecosystem

| Component | Description |
|-----------|-------------|
| Omniverse Nucleus | Asset server; local or cloud; enables live USD collaboration |
| Omniverse Replicator | Synthetic data generation extension for Isaac Sim |
| Omniverse NuRec | Neural 3D reconstruction from real-world captures → OpenUSD |
| Isaac Sim 4.5 / 5.0 | Physics + sensors simulation; built on Omniverse Kit |
| Isaac Lab 2.2 | RL training framework on top of Isaac Sim |

### Supporting Libraries

| Library | Package | Role |
|---------|---------|------|
| PyTorch | `torch` | Deep learning backbone for gsplat / 3DGRUT |
| torchvision | `torchvision` | Image preprocessing utilities |
| NumPy | `numpy` | Array operations |
| OpenCV | `opencv-python` | Image I/O, preprocessing |
| Pillow | `Pillow` | Image file I/O |
| Open3D | `open3d` | 3D point cloud / mesh processing |
| trimesh | `trimesh` | Mesh loading/export utilities |
| plyfile | `plyfile` | Low-level PLY read/write |
| imageio | `imageio` | Video frame extraction |
| tqdm | `tqdm` | Progress bars |
| scipy | `scipy` | Scientific computing, sparse matrices |

---

## 6. Integrating USD into Isaac Sim

### 6.1 Local Isaac Sim Deployment

Isaac Sim is installed locally via either:
- **Omniverse Launcher** (GUI) → add-on installation
- **Direct pip install** (headless/scripted):
  ```bash
  pip install isaacsim --extra-index-url https://pypi.nvidia.com
  ```

**Loading a USD digital twin scene:**
```python
from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

import omni.usd
stage = omni.usd.get_context().get_stage()

# Reference the reconstructed USD scene
from pxr import Usd, UsdGeom
stage.GetRootLayer().subLayerPaths.append("./exports/my_scene.usd")
```

**Local workflow advantages:**
- Zero network latency; ideal for interactive iteration.
- Full control over asset paths and versioning.
- Works offline; suitable for IP-sensitive or regulated environments.
- Easier debugging of Python scripts and extensions.

**Local workflow limitations:**
- Single-user; no built-in collaboration without a local Nucleus server.
- Bounded by workstation GPU VRAM (12–24 GB typical consumer cards).
- Large photorealistic scenes may require reduced viewport resolution.

---

### 6.2 Cloud Isaac Sim Deployment

**Option A — AWS Marketplace AMI:**
```bash
# Launch an EC2 g6e.2xlarge with the official Isaac Sim AMI
# Connect via NICE DCV (AWS remote desktop protocol)
# Load USD from S3 or Nucleus
```

**Option B — NVIDIA Omniverse Cloud / BCP:**
- Managed deployment; NVIDIA handles Nucleus + Isaac Sim setup.
- Access via streaming client (web browser or Omniverse Streaming Client app).
- Enterprise-grade: supports multi-user collaboration on the same USD stage.

**Option C — Self-hosted Nucleus + Cloud Isaac:**
- Deploy Nucleus server on a VM (or on-prem server).
- Connect Isaac Sim (cloud or local) to the Nucleus server's `omniverse://` endpoint.
- All team members reference the same USD asset tree.

**Cloud workflow advantages:**
- Scales to any scene complexity (48–80 GB VRAM instances available).
- Multi-user real-time collaboration on the same USD stage.
- Supports batch simulation jobs (e.g., 1000 parallel Isaac Lab environments).
- No local GPU upgrade required.

**Cloud workflow limitations:**
- Requires stable high-bandwidth internet (≥100 Mbps for streaming).
- Ongoing cost: $1–2/hr for Isaac Sim-capable instances.
- Data privacy: USD assets and captures leave the local machine.
- Remote rendering latency can impact interactive iteration.

---

### 6.3 Comparison: Local vs. Cloud

| Criterion | Local | Cloud |
|-----------|-------|-------|
| Cost | One-time hardware (GPU ~$1000–3000) | ~$1–2/hr on-demand |
| VRAM ceiling | 12–24 GB (consumer), 48 GB (Pro) | 48–80 GB (L40S, A100 Ada) |
| Collaboration | Single user (or LAN Nucleus) | Multi-user, real-time |
| Scene scale | Small–medium (< 500k splats) | Unlimited |
| Latency | None | 20–100 ms (streaming) |
| Privacy / IP | Full local control | Data leaves premises |
| Offline capability | ✅ Yes | ❌ No |
| Isaac Sim version control | Manual | Managed (cloud AMI updates) |
| Best for | Development, prototyping, local Cosmos | Production, team collaboration, large scenes |

**Recommendation for this project:**  
Use **local** Isaac Sim with locally available Cosmos Reason and Cosmos Transfer for development and prototyping. Migrate to **cloud (AWS g6e / NVIDIA BCP)** when scenes grow beyond 12 GB VRAM capacity or when multi-user collaboration is needed.

---

## 7. Camera Navigation in the Digital Twin

Isaac Sim provides rich camera utilities for navigating the digital twin scene. Cameras are **not optional** — they are required for:
- Generating synthetic sensor data (RGB, depth, segmentation) for robot training.
- Visualising the scene during development.
- Producing video walkthroughs of the twin for stakeholder review.

### Camera Types in Isaac Sim

| Camera Type | USD Prim | Description |
|------------|----------|-------------|
| Perspective camera | `UsdGeom.Camera` | Standard pinhole; adjustable FOV, focal length, aperture |
| Fisheye / distorted | `UsdGeom.Camera` + distortion params | Supported via 3DGUT-compatible camera model |
| Orthographic | `UsdGeom.Camera` | For top-down maps or technical views |
| Virtual sensor | `omni.isaac.sensor` extension | Simulated RGB, depth, thermal, LiDAR, IMU |

### Creating a Camera via Python

```python
from pxr import UsdGeom, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Create a camera prim
camera_path = "/World/DigitalTwin/NavCamera"
camera = UsdGeom.Camera.Define(stage, camera_path)

# Set intrinsics
camera.GetFocalLengthAttr().Set(24.0)          # mm
camera.GetHorizontalApertureAttr().Set(36.0)   # mm (35mm full-frame equivalent)
camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))  # near/far

# Position and orient the camera
xform = UsdGeom.Xformable(camera)
xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.7, 3.0))  # eye height + stand-off
xform.AddRotateXYZOp().Set(Gf.Vec3f(-10.0, 0.0, 0.0))  # slight downward tilt
```

### Navigation Modes

| Mode | Use Case |
|------|----------|
| **Free-fly** (interactive) | Manual exploration in the Isaac Sim GUI viewport |
| **Orbit / turntable** | Inspecting a reconstructed object |
| **Path-based animation** | Programmatic camera flight for video walkthroughs or test trajectories |
| **Robot-mounted** | Camera rigidly attached to a robot link; moves with the robot |
| **Multi-camera array** | Simulated sensor suite (e.g., stereo + fisheye) for perception training |

### Path-Based Camera Animation (USD)

```python
from pxr import UsdGeom, Gf, Vt, Sdf

# Add time-sampled translation for a walkthrough
camera_xform = UsdGeom.Xformable(camera)
translate_op = camera_xform.AddTranslateOp()

# Set keyframes
time_samples = {0: Gf.Vec3d(0, 1.7, 10),
                50: Gf.Vec3d(5, 1.7, 5),
                100: Gf.Vec3d(10, 1.7, 0)}
for t, pos in time_samples.items():
    translate_op.Set(pos, time=t)
```

---

## 8. Robotic Systems Integration

### Is Robotic Integration Mandatory?

**No — robotic systems are optional** for the initial digital twin baseline. The primary deliverable is a navigable, visually accurate USD scene reconstructed from footage. Robots can be added in a later iteration for:
- Collision avoidance testing.
- Task and motion planning validation.
- Synthetic training data with embodied agents.

### When to Import Robots

Import a robotic system into the digital twin when:
1. You want to simulate a robot operating **within** the captured environment (e.g., a mobile base navigating a factory floor digital twin).
2. You need **synthetic labelled data** (e.g., camera-on-arm views with bounding box annotations).
3. You are validating **path planning** or **safety boundaries** against the real environment geometry.

### Supported Robot Description Formats

| Format | Use | Isaac Sim Import |
|--------|-----|-----------------|
| URDF | ROS standard; most robots have URDF | ✅ via URDF importer (auto-converts to USD) |
| MJCF | MuJoCo format; used in many RL benchmarks | ✅ via MJCF importer |
| USD (native) | Pre-built SimReady assets from NVIDIA | ✅ drag and drop |
| FBX / OBJ | Visual-only meshes | ✅ via Mesh importer (no physics) |

### Importing a Robot (URDF → USD)

```python
# Using the Isaac Sim URDF importer
from omni.importer.urdf import _urdf

urdf_interface = _urdf.acquire_urdf_interface()
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = False

robot_usd_path = urdf_interface.import_robot(
    "/path/to/robot.urdf",
    "/World/Robot",
    import_config
)
```

### Recommended Starting Robot Assets (NVIDIA Isaac Sim)

The following robots are available as pre-built, physics-ready USD assets on the NVIDIA NGC / Omniverse asset library:
- **Franka Panda** — 7-DOF manipulator; great for tabletop tasks
- **Boston Dynamics Spot** — quadruped; ideal for floor-walking in environments
- **iRobot Create 3** — simple mobile base for navigation testing
- **Universal Robots UR10e** — industrial arm; relevant for factory twin scenarios

---

## 9. NVIDIA Cosmos Models (Reason & Transfer)

You have local access to **Cosmos Reason** and **Cosmos Transfer** models. Here is how they integrate into the digital twin workflow:

### Cosmos Reason 2
- **Type:** Vision-Language Model (VLM), ~7B parameters
- **Role:** Scene understanding, annotation, safety checking
- **In Digital Twin context:**
  - Automatically annotate captured footage (object classes, boundaries) to improve reconstruction quality.
  - Generate scene descriptions in natural language from Isaac Sim camera feeds.
  - Validate robotic task completion in simulation.
  - Curate / filter low-quality synthetic frames from Replicator output.
- **Local VRAM:** Fits in 12 GB at INT4/INT8 quantisation via `llama.cpp` or HuggingFace `transformers` with BitsAndBytes.

### Cosmos Transfer 2.5
- **Type:** Conditional world generation model (diffusion-based)
- **Role:** Photorealistic domain transfer from structured sim outputs
- **In Digital Twin context:**
  - Take a depth map / segmentation mask / edge map from Isaac Sim and generate a photorealistic RGB image — effectively "texturing" the simulation.
  - Domain randomisation: generate varied lighting, weather, texture conditions from the same geometry.
  - Augment training datasets with realistic scene variations.
- **Local VRAM:** Full-precision inference requires 16–24 GB. **At 12 GB**, use quantised or smaller checkpoint variants, or offload to cloud inference (NGC API).

### Local Cosmos Workflow Integration

```python
# Example: use Cosmos Reason to annotate an Isaac Sim camera frame
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason2-7B")
model = AutoModelForVision2Seq.from_pretrained(
    "nvidia/Cosmos-Reason2-7B",
    torch_dtype=torch.float16,
    load_in_4bit=True  # fits in 12 GB VRAM
)

# Feed camera frame from Isaac Sim replicator
inputs = processor(images=isaac_sim_frame, text="Describe the scene and identify objects.", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0]))
```

---

## 10. Recommended End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1 — CAPTURE                                          │
│  Smartphone / camera orbits scene                           │
│  → Export 4K video                                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2 — RECONSTRUCTION                                   │
│  ffmpeg extracts frames                                     │
│  ns-process-data → COLMAP SfM → camera poses               │
│  gsplat / 3DGRUT training → 3D Gaussian Splat              │
│  Export → scene.ply                                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3 — USD CONVERSION                                   │
│  ply_to_usd.py  (OpenUSD 26.03 schema)                     │
│  → scene.usd   (UsdVolParticleField3DGaussianSplat)        │
│  + Add lights, ground plane, metadata in NVIDIA USD Composer│
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4 — ISAAC SIM INTEGRATION                            │
│  Load scene.usd into Isaac Sim                              │
│  Add navigation cameras (perspective / fisheye)            │
│  Optional: import robot (URDF → USD)                       │
│  Run Omniverse Replicator for synthetic data generation     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5 — COSMOS AUGMENTATION                              │
│  Cosmos Reason 2: annotate / validate scenes               │
│  Cosmos Transfer 2.5: domain-transfer sim frames           │
│    → photorealistic training images                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Conclusion & Next Steps

### Summary

This project is **feasible on consumer hardware with 12 GB VRAM** for prototyping and small-to-medium scenes. The complete pipeline from video capture to Isaac Sim simulation can be executed locally on an RTX 3080 or RTX 4070 Ti. For large-scale production scenes and full-fidelity Cosmos Transfer inference, cloud GPUs (AWS L40S or NVIDIA BCP) are recommended.

### Key Technologies Selected

| Component | Selected Technology | Rationale |
|-----------|--------------------|-----------| 
| 3D Reconstruction | **gsplat + 3DGRUT** | Best-in-class quality; native 3DGUT in gsplat; OpenUSD compatible |
| Capture preprocessing | **Nerfstudio + COLMAP** | Automated, single command; accessible to a single person |
| USD format | **OpenUSD 26.03+** | Native Gaussian Splat support; no custom hacks needed |
| Simulation | **Isaac Sim 4.5 / 5.0** | Best robotics digital twin platform; Cosmos integration |
| World Models | **Cosmos Reason 2 + Transfer 2.5** | Already available locally; powerful augmentation |
| Asset server | **Local Nucleus** (→ Cloud Nucleus when needed) | Start simple; scale to cloud |

### Immediate Next Steps

1. **Install the Python dependencies** from `requirements.txt`.
2. **Capture a test scene** — walk around a room or an object for 3–5 minutes.
3. **Run `ns-process-data video`** to extract frames and estimate camera poses.
4. **Train a splat model** with `ns-train splatfacto` or the `3dgrut` scripts.
5. **Convert to USD** using the OpenUSD 26.03 PLY importer or Adobe's plugin.
6. **Load the scene in Isaac Sim** and add a navigation camera.
7. **Experiment with Cosmos Transfer** to generate photorealistic images from depth maps.

---

*Report generated for the `vaultwares-studio` project. Last updated: April 2026.*
