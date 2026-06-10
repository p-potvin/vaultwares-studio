"""Generic stage bootstrap baked into the vw-studio-worker image.

Mirrors runners/hf_jobs.py BOOTSTRAP_SOURCE: the runner passes the stage
definition via the VW_STAGE_CONFIG env var (JSON: repo, prefix, command) and
HF_TOKEN as a job secret. Inputs land in $VW_IN, the stage command writes its
results to $VW_OUT, and everything in $VW_OUT is uploaded back to the
artifact dataset under <prefix>/out/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> int:
    if "VW_STAGE_CONFIG" not in os.environ:
        # Image-build mode: this container also runs as a (paused) HF Space
        # whose only purpose is to have HF build the image server-side.
        # Idle instead of crashing so the Space build registers as healthy.
        import time

        print("[vw-stage] no VW_STAGE_CONFIG — idling (image-build Space mode)", flush=True)
        while True:
            time.sleep(3600)
    cfg = json.loads(os.environ["VW_STAGE_CONFIG"])
    work = Path("/tmp/vw_stage")
    in_dir = work / "in"
    out_dir = work / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=cfg["repo"],
        repo_type="dataset",
        allow_patterns=[cfg["prefix"] + "/in/*"],
        local_dir=str(work / "dl"),
    )
    staged = work / "dl" / cfg["prefix"] / "in"
    if staged.exists():
        for item in staged.iterdir():
            item.replace(in_dir / item.name)

    env = dict(os.environ)
    env["VW_IN"] = str(in_dir)
    env["VW_OUT"] = str(out_dir)
    print(f"[vw-stage] running: {cfg['command']}", flush=True)
    result = subprocess.run(cfg["command"], env=env)
    print(f"[vw-stage] exit code: {result.returncode}", flush=True)

    api = HfApi()
    if any(out_dir.iterdir()):
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=cfg["prefix"] + "/out",
            repo_id=cfg["repo"],
            repo_type="dataset",
        )
        print("[vw-stage] outputs uploaded", flush=True)
    else:
        print("[vw-stage] no outputs produced", flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
