"""Wait for the vw-studio-worker Space image build, then pause the Space.

The Space exists only so HF builds the worker image; once built (stage
RUNNING) it is paused to keep things tidy. Polls every 30 s, gives up after
90 minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi  # noqa: E402

from vaultwares_studio.runners import get_hf_token  # noqa: E402

TERMINAL_OK = {"RUNNING", "APP_STARTING", "RUNNING_APP_STARTING"}
TERMINAL_FAIL = {"BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR", "DELETING", "STOPPED"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="vw-studio-worker")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    args = parser.parse_args()

    api = HfApi(token=get_hf_token())
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{args.space}"
    deadline = time.monotonic() + args.max_minutes * 60
    last_stage = ""

    while time.monotonic() < deadline:
        stage = api.get_space_runtime(repo_id).stage
        if stage != last_stage:
            print(f"[{time.strftime('%H:%M:%S')}] stage: {stage}", flush=True)
            last_stage = stage
        if stage in TERMINAL_FAIL:
            print(f"BUILD FAILED: {stage} — check https://huggingface.co/spaces/{repo_id}?logs=build")
            return 1
        if stage in TERMINAL_OK:
            # Leave the Space RUNNING: the Jobs backend resolves the image from
            # a public running Space (paused/private both produced 500s), and
            # the stub server sits on the free CPU tier at no cost.
            print(f"Image built and Space running. Job image: hf.co/spaces/{repo_id}")
            return 0
        time.sleep(args.poll_seconds)

    print(f"Timed out after {args.max_minutes} min; last stage: {last_stage}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
