"""Run a train-only DA3/Splatfacto stage and retain its complete provenance.

This runs inside an HF Job after the standard bootstrap has populated VW_IN
and VW_OUT.  It intentionally keeps the stable input archives and a streamed
log alongside the worker's usual splat/checkpoint outputs; temporary training
directories remain owned by the worker and are not copied to VW_OUT.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_train_only_with_retention.py <worker command...>")

    in_dir = Path(os.environ["VW_IN"])
    out_dir = Path(os.environ["VW_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    command = sys.argv[1:]
    started = time.monotonic()

    log_path = out_dir / "stage.log"
    with log_path.open("w", encoding="utf-8") as log:
        # Nerfstudio/Rich otherwise sees a pipe and may hold its progress output
        # until process exit, which makes a healthy long training look stalled.
        child_env = dict(os.environ, PYTHONUNBUFFERED="1")
        process = subprocess.Popen(
            command,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()

    # These are the exact inputs used for this run.  The worker's post-training
    # processed_min.zip is deliberately compact, so preserve the complete
    # image-bearing handoff separately as training_input.zip.
    retained = {
        "frames.zip": "frames.zip",
        "processed_min.zip": "training_input.zip",
    }
    for source_name, target_name in retained.items():
        source = in_dir / source_name
        if source.is_file():
            shutil.copyfile(source, out_dir / target_name)

    (out_dir / "run_result.json").write_text(
        json.dumps(
            {
                "returncode": returncode,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "command": command,
                "retained_inputs": [target for source, target in retained.items() if (in_dir / source).is_file()],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
