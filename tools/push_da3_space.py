"""Build the vw-studio-da3 image on Hugging Face.

Separate from the prod worker Space — DA3 adds ~2 GB of dependencies
(xformers, DA3 source, specific gsplat commit) that prod jobs don't need.
Keeping the Spaces separate avoids slowing down prod image pulls.

Uploads:
  - docker/da3/Dockerfile              (DA3-specific image)
  - docker/worker/vw_stage.py          (shared bootstrap)
  - docker/worker/da3_entrypoint.py    (DA3 stage entrypoint)
  - docker/worker/recon_entrypoint.py  (fallback COLMAP entrypoint for --train-only)
  - docker/worker/render_entrypoint.py (shared render entrypoint)

Usage:
    .venv\\Scripts\\python.exe tools\\push_da3_space.py [--space vw-studio-da3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi  # noqa: E402

from vaultwares_studio.runners import get_hf_token  # noqa: E402

DA3_DIR = ROOT / "docker" / "da3"
WORKER_DIR = ROOT / "docker" / "worker"
SHARED_ENTRYPOINTS = ("vw_stage.py", "da3_entrypoint.py", "recon_entrypoint.py", "render_entrypoint.py")
# Library modules needed by da3_entrypoint.py's --merge-splat mode.
# Copied from vaultwares_studio/ into the Docker image root.
LIBRARY_MODULES = (
    "gaussian_merge.py",
    "splat_io.py",
    "streaming_convert.py",
    # streaming_convert imports this one flat inside the container.
    "camera_calibration.py",
)

README_BY_SPACE = {
    "vw-studio-da3": """---
title: VW Studio DA3
emoji: "\U0001F50D"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# vw-studio-da3

DA3 (Depth Anything 3) execution image for VaultWares Studio. Used by
da3-standard and da3-high — DA3 pose+depth SfM feeding splatfacto training.
Proven green (docs/da3-standard-benchmark-20260713.md). No nvcc in this
image; --gs-only direct-3DGS is NOT available here — see vw-studio-da3-gs.
""",
    "vw-studio-da3-gs": """---
title: VW Studio DA3 GS
emoji: "\U0001F52C"
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
---

# vw-studio-da3-gs

Experimental sibling of vw-studio-da3, used by da3-draft and da3-incremental
(the --gs-only / direct-3DGS presets). Adds the CUDA 11.8 devel toolchain
(cuda-nvcc-11-8 + cuda-cudart-dev-11-8) so gsplat's custom commit actually
builds instead of silently no-op'ing. Unproven — repoint the two presets at
vw-studio-da3 once a run here is verified green.
""",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="vw-studio-da3")
    args = parser.parse_args()

    token = get_hf_token()
    if not token:
        print("No HF token configured (Settings > Remote Compute).")
        return 1
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{args.space}"

    # Public is mandatory for Spaces used as Job image sources.
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    try:
        api.update_repo_settings(repo_id=repo_id, private=False, repo_type="space")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not enforce public visibility: {exc}")

    readme = README_BY_SPACE.get(args.space, README_BY_SPACE["vw-studio-da3"])
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=str(DA3_DIR / "Dockerfile"),
        path_in_repo="Dockerfile",
        repo_id=repo_id,
        repo_type="space",
    )
    for name in SHARED_ENTRYPOINTS:
        source = WORKER_DIR / name
        if not source.exists():
            print(f"Missing entrypoint: {source}", file=sys.stderr)
            return 2
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="space",
        )
    # Upload library modules from vaultwares_studio/ (the package directory).
    package_dir = ROOT / "vaultwares_studio"
    for name in LIBRARY_MODULES:
        source = package_dir / name
        if not source.exists():
            print(f"Missing library module: {source}", file=sys.stderr)
            return 2
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="space",
        )
    print(f"Pushed DA3 files to https://huggingface.co/spaces/{repo_id}")
    print(f"Job image reference: hf.co/spaces/{repo_id}")
    runtime = api.get_space_runtime(repo_id)
    print(f"Space stage: {runtime.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
