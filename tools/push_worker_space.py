"""Build the vw-studio-worker image on Hugging Face instead of local Docker.

Pushes docker/worker/* to a (private) Docker-SDK Space; HF builds the image
server-side. HF Jobs can then run it via image="hf.co/spaces/<user>/<space>".
Useful when local Docker/WSL is unavailable. Re-run after editing the
Dockerfile or stage entrypoints to trigger a rebuild.

Usage:
    .venv\\Scripts\\python.exe tools\\push_worker_space.py [--space vw-studio-worker]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi  # noqa: E402

from vaultwares_studio.runners import get_hf_token  # noqa: E402

WORKER_DIR = ROOT / "docker" / "worker"
README = """---
title: VW Studio Worker
emoji: "\U0001F6E0"
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
---

# vw-studio-worker

Build-only Space: Hugging Face builds this Docker image so VaultWares Studio
jobs can run it (`image="hf.co/spaces/{owner}/{space}"`). The container idles
when started as a Space; all real work happens in HF Jobs with
`VW_STAGE_CONFIG` set. Keep the Space paused.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="vw-studio-worker")
    args = parser.parse_args()

    token = get_hf_token()
    if not token:
        print("No HF token configured (Settings > Remote Compute).")
        return 1
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{args.space}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=README.format(owner=owner, space=args.space).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    for name in ("Dockerfile", "vw_stage.py", "recon_entrypoint.py"):
        api.upload_file(
            path_or_fileobj=str(WORKER_DIR / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="space",
        )
    print(f"Pushed worker files to https://huggingface.co/spaces/{repo_id}")
    print(f"Job image reference: hf.co/spaces/{repo_id}")
    runtime = api.get_space_runtime(repo_id)
    print(f"Space stage: {runtime.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
