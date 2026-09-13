"""Container-only runner for a prepared pose-only DA3 comparison variant.

Run only inside an approved HF job, after unpacking worker.zip into /opt/vw.
The user controls cancellation. There is no local execution timeout; the
worker runs to completion and then archives its outputs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def validate_args(args: list[str]) -> None:
    expected = ["--stream-sfm", "--da3-model", "depth-anything/DA3-LARGE-1.1",
                "--stream-resolution", "504x280", "--stream-chunk-size", "60",
                "--stream-overlap", "30"]
    if args not in (expected, expected + ["--stream-loop-closure"]):
        raise ValueError("Arguments differ from the prepared pose-only comparison.")


def main() -> int:
    args = sys.argv[1:]
    validate_args(args)
    output = Path(os.environ["VW_OUT"])
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="vw-da3-comparison-"))
    child_output = work / "results"
    child_output.mkdir()
    env = dict(os.environ, VW_OUT=str(child_output))
    started = time.monotonic()
    code = 1
    error = None
    with (output / "stage.log").open("w", encoding="utf-8") as log:
        try:
            with subprocess.Popen([sys.executable, "/opt/vw/da3_entrypoint.py", *args],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", bufsize=1,
                                  start_new_session=True) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
        except OSError as exc:
            error = str(exc)
    # The retained per-frame NPZ files are already compressed. Store them as-is
    # to make archiving proportional to disk I/O, not repeated CPU compression.
    # DA3 cleans raw chunk scratch after producing its stable outputs.
    print("Archiving retained streaming outputs...", flush=True)
    # Retain the exact input archive with the result so the remote run is
    # self-contained and the frames can be recovered alongside its outputs.
    input_frames = Path(os.environ["VW_IN"]) / "frames.zip"
    if input_frames.exists():
        shutil.copyfile(input_frames, output / "frames.zip")
    with zipfile.ZipFile(output / "streaming_artifacts.zip", "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(child_output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(child_output).as_posix())
    processed = child_output / "processed_min.zip"
    if processed.exists():
        shutil.copyfile(processed, output / processed.name)
    (output / "run_result.json").write_text(json.dumps({
        "returncode": code, "error": error, "worker_args": args,
        "elapsed_seconds_including_archive": time.monotonic() - started,
    }, indent=2), encoding="utf-8")
    print(f"Pose comparison finished with code {code}; retained artifacts in {output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
