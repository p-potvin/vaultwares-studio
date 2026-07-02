# vw-studio-worker — remote execution image for VaultWares Studio stages.
# v0.2: based on the official nerfstudio image, which already ships
# nerfstudio + gsplat + CUDA COLMAP + ffmpeg. Our layer adds only the HF Hub
# transport and the stage entrypoints — keeps the (HF Space) build tiny.
# Habitat-Sim and Cosmos layers are added in M3/M4.
#
# Built server-side on Hugging Face (local Docker not required):
#   .venv\Scripts\python.exe tools\push_worker_space.py
# Jobs run it via image="hf.co/spaces/<user>/vw-studio-worker".

FROM ghcr.io/nerfstudio-project/nerfstudio:latest

USER root
ENV PYTHONUNBUFFERED=1
# Drop a sitecustomize.py into every plausible site-packages location so that
# whichever one Python's site module actually imports from, it picks ours up.
# Prior version only wrote to getsitepackages()[0]; the TF32 warning kept
# appearing in training logs, meaning that path wasn't on sys.path at startup.
# Belt-and-braces approach: enumerate all candidate dirs and write to each.
# sitecustomize.py runs before any user code — sets TF32=high on Ampere+ GPUs
# (~2x FP32 matmul speedup on L4 with negligible accuracy loss for splatfacto).
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
                    > "$d/sitecustomize.py" \
                && echo "[Dockerfile] wrote sitecustomize.py to $d"; \
            fi; \
        done
# CuDNN's runtime-compiled kernel selection wants libnvrtc.so. Without it,
# splatfacto.py applies a workaround that picks slower kernel paths. The
# matching CUDA version comes from the base nerfstudio image; we install the
# python distribution package that ships the shared object.
RUN python3 -m pip install --no-cache-dir nvidia-cuda-nvrtc-cu12 \
    || pip3 install --no-cache-dir nvidia-cuda-nvrtc-cu12 \
    || echo "[Dockerfile] nvrtc package install skipped (already present or unavailable)"

# The nerfstudio image exposes python3 but neither `pip` nor a bare `python`
# on PATH; job commands use `python`, so alias it.
RUN ln -sf "$(command -v python3)" /usr/local/bin/python \
    && (python3 -m pip install --no-cache-dir \
            "huggingface_hub>=0.26" \
            "plyfile>=1.0.3" \
            "trimesh>=4.3.0" \
        || pip3 install --no-cache-dir \
            "huggingface_hub>=0.26" \
            "plyfile>=1.0.3" \
            "trimesh>=4.3.0")

# Generic stage bootstrap: downloads jobs/<id>/<stage>/in from the artifact
# dataset, runs the stage command with VW_IN/VW_OUT, uploads out/.
COPY vw_stage.py /opt/vw/vw_stage.py
# Stage entrypoints invoked by the bootstrap's stage command.
COPY recon_entrypoint.py /opt/vw/recon_entrypoint.py
COPY render_entrypoint.py /opt/vw/render_entrypoint.py

WORKDIR /workspace
CMD ["python", "/opt/vw/vw_stage.py"]
