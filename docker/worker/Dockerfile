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
