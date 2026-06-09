# vw-studio-worker — remote execution image for VaultWares Studio stages.
# v0.1 (M0/M1 scope): COLMAP + nerfstudio/gsplat + huggingface_hub transport.
# Habitat-Sim and Cosmos layers are added in M3/M4.
#
# Build & push:  tools/build_worker_image.ps1 -Tag 0.1 -Repo <dockerhub-user>/vw-studio-worker
# HF Jobs pulls public Docker Hub images; keep model weights OUT of the image
# (they travel through HF Hub volumes/datasets instead).

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Prebuild gsplat CUDA kernels for common datacenter GPUs:
    # T4 (7.5), A10G/A100 ranges (8.0/8.6), L4/L40S (8.9)
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# COLMAP from Ubuntu repos (CPU build, sufficient for sequential matching on
# pre-extracted frames). Swap to a CUDA build in M1 if SfM becomes the
# bottleneck. ffmpeg backs ns-process-data and walkthrough renders.
RUN apt-get update && apt-get install -y --no-install-recommends \
        colmap \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "huggingface_hub>=0.26" \
        "nerfstudio>=1.1.0" \
        "gsplat>=1.4.0" \
        "plyfile>=1.0.3" \
        "trimesh>=4.3.0"

# Generic stage bootstrap: downloads jobs/<id>/<stage>/in from the artifact
# dataset, runs the stage command with VW_IN/VW_OUT, uploads out/.
COPY vw_stage.py /opt/vw/vw_stage.py
# Stage entrypoints invoked by the bootstrap's stage command.
COPY recon_entrypoint.py /opt/vw/recon_entrypoint.py

WORKDIR /workspace
CMD ["python", "/opt/vw/vw_stage.py"]
