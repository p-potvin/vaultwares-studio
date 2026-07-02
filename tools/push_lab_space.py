"""Build the vw-studio-recon-lab image on Hugging Face.

Mirrors push_worker_space.py but targets the experimental Space. Uploads:
  - docker/lab/Dockerfile  (lab-specific image definition)
  - docker/worker/vw_stage.py            (shared bootstrap)
  - docker/worker/recon_entrypoint.py    (shared --sfm-only path)
  - docker/worker/render_entrypoint.py   (shared, unused for now but kept consistent)

Usage:
    .venv\\Scripts\\python.exe tools\\push_lab_space.py [--space vw-studio-recon-lab]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi  # noqa: E402

from vaultwares_studio.runners import get_hf_token  # noqa: E402

LAB_DIR = ROOT / "docker" / "lab"
WORKER_DIR = ROOT / "docker" / "worker"
SHARED_ENTRYPOINTS = ("vw_stage.py", "recon_entrypoint.py", "render_entrypoint.py")

README = """---
title: VW Studio Recon Lab
emoji: "\U0001F9EA"
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# vw-studio-recon-lab

Experimental sibling of `vw-studio-worker`. Used for cpu-upgrade dense-frame
COLMAP runs and (later) ray-traced gaussian-splat training via NVIDIA's
3DGRUT. Keep this Space paused — real work happens in HF Jobs invoked by
`tools/queue_lab_recon.py`. Production workloads should NOT target this image.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="vw-studio-recon-lab")
    args = parser.parse_args()

    token = get_hf_token()
    if not token:
        print("No HF token configured (Settings > Remote Compute).")
        return 1
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{args.space}"

    # HF Jobs cannot pull container images from private Spaces (the runtime
    # gets ErrImagePull / NotFound). Public is mandatory for Spaces used as
    # Job image sources -- the source code itself stays read-only to others;
    # only the built image becomes pullable. Matches vw-studio-worker.
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    # If a previous run created the Space as private, exist_ok=True won't flip
    # it. Force the visibility every push so re-runs heal a stale private flag.
    try:
        api.update_repo_settings(repo_id=repo_id, private=False, repo_type="space")
    except Exception as exc:  # noqa: BLE001 - best-effort; explicit creates below still work
        print(f"warning: could not enforce public visibility: {exc}")
    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=str(LAB_DIR / "Dockerfile"),
        path_in_repo="Dockerfile",
        repo_id=repo_id,
        repo_type="space",
    )
    for name in SHARED_ENTRYPOINTS:
        source = WORKER_DIR / name
        if not source.exists():
            print(f"Missing shared entrypoint: {source}", file=sys.stderr)
            return 2
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="space",
        )
    print(f"Pushed lab files to https://huggingface.co/spaces/{repo_id}")
    print(f"Job image reference: hf.co/spaces/{repo_id}")
    runtime = api.get_space_runtime(repo_id)
    print(f"Space stage: {runtime.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
