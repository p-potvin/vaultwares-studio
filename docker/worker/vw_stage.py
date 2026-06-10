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
        # Image-build mode: this container also runs as an HF Space whose only
        # purpose is to have HF build (and host) the image server-side. Docker
        # Spaces health-check an HTTP port, so serve a stub on 7860 — without
        # it the Space hangs in APP_STARTING and Jobs refuses the image ref.
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class _Stub(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = b"vw-studio-worker image host; real work runs in HF Jobs."
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        print("[vw-stage] no VW_STAGE_CONFIG — serving image-host stub on :7860", flush=True)
        HTTPServer(("0.0.0.0", 7860), _Stub).serve_forever()
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
