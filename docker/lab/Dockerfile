# vw-studio-recon-lab — experimental worker image for VaultWares Studio.
# Mirrors docker/worker/Dockerfile for now; the lab Space exists so we can
# iterate on dense-frame COLMAP runs and (later) install 3DGRUT/3DGUT for
# rasterized + ray-traced gaussian splatting on the GPU leg without touching
# the production worker image.
#
# Built server-side on Hugging Face via:
#   .venv\Scripts\python.exe tools\push_lab_space.py
# Jobs run it via image="hf.co/spaces/<owner>/vw-studio-recon-lab".
#
# What differs from docker/worker/Dockerfile:
#   * VW_LAB=1 environment marker (so logs / summary.json can distinguish runs)
#   * Otherwise identical. The shared recon_entrypoint.py understands --sfm-only,
#     which is the only flag queue_lab_recon.py currently fires.

FROM ghcr.io/nerfstudio-project/nerfstudio:latest

USER root
ENV PYTHONUNBUFFERED=1
ENV VW_LAB=1

# Belt-and-braces sitecustomize.py drop — see worker Dockerfile for rationale.
# Mirrored here so the lab image behaves identically on Ampere+ GPUs if/when
# we wire the GPU leg in.
RUN python3 -c "import site; \
        print(' '.join(site.getsitepackages() + [site.getusersitepackages()]))" \
        > /tmp/sitedirs.txt \
    && for d in $(cat /tmp/sitedirs.txt) \
                /usr/local/lib/python3.10/dist-packages \
                /usr/lib/python3/dist-packages \
                /usr/lib/python3.10/dist-packages; do \
            if [ -d "$d" ]; then \
                printf '%s\n' \
                    'try:' \
                    '    import torch' \
                    "    torch.set_float32_matmul_precision('high')" \
                    'except Exception:' \
                    '    pass' \
                    > "$d/sitecustomize.py"; \
            fi; \
        done

RUN python3 -m pip install --no-cache-dir nvidia-cuda-nvrtc-cu12 \
    || pip3 install --no-cache-dir nvidia-cuda-nvrtc-cu12 \
    || echo "[lab Dockerfile] nvrtc package install skipped"

RUN ln -sf "$(command -v python3)" /usr/local/bin/python \
    && (python3 -m pip install --no-cache-dir \
            "huggingface_hub>=0.26" \
            "plyfile>=1.0.3" \
            "trimesh>=4.3.0" \
        || pip3 install --no-cache-dir \
            "huggingface_hub>=0.26" \
            "plyfile>=1.0.3" \
            "trimesh>=4.3.0")

# Shared entrypoints — uploaded by push_lab_space.py from docker/worker/.
COPY vw_stage.py /opt/vw/vw_stage.py
COPY recon_entrypoint.py /opt/vw/recon_entrypoint.py
COPY render_entrypoint.py /opt/vw/render_entrypoint.py

WORKDIR /workspace
CMD ["python", "/opt/vw/vw_stage.py"]
