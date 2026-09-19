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
import threading
import time
from pathlib import Path

TRAIN_ROOT = Path("/tmp/da3_recon/train")


def _latest_step(root: Path = TRAIN_ROOT) -> int | None:
    """Highest checkpoint step written so far, or None before the first save."""
    steps = []
    for path in root.glob("**/nerfstudio_models/step-*.ckpt"):
        digits = path.stem.split("-")[-1]
        if digits.isdigit():
            steps.append(int(digits))
    return max(steps) if steps else None


def _vram() -> str:
    """Peak VRAM the training process has asked CUDA for.

    Host RAM is what the container gets killed over, so that is logged first,
    but VRAM is what decides whether a smaller GPU would do. Nothing in the
    stack reported it before, so after the 18 Sep run the honest answer to
    "how much VRAM did that need?" was that we had never measured it.

    Read out of the child's process, not this one: the wrapper never touches
    CUDA. ``nvidia-smi`` reports the whole device, which for a single-tenant
    job is the number we want.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            used, total, util = (x.strip() for x in result.stdout.strip().splitlines()[0].split(","))
            return f"vram {int(used)/1024:.1f}/{int(total)/1024:.1f} GB, gpu {util}%"
    except Exception:  # noqa: BLE001 - diagnostics are best-effort
        pass
    return "vram unavailable"


def _memory() -> str:
    """Container memory, from the cgroup where available and /proc as a fallback.

    Worth logging every heartbeat: on 17 Sep a run was OOMKilled after 271
    minutes with no warning in the log at all, because the only thing being
    printed was the child's stdout and nerfstudio's Rich progress never reaches
    a pipe. A climbing number here is the signal that was missing.
    """
    parts = []
    for current, limit in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                           ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
                            "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            used = int(Path(current).read_text().strip())
            cap_raw = Path(limit).read_text().strip()
            cap = None if cap_raw == "max" else int(cap_raw)
            if cap and cap < (1 << 62):
                parts.append(f"cgroup {used/1e9:.1f}/{cap/1e9:.1f} GB ({used/cap:.0%})")
            else:
                parts.append(f"cgroup {used/1e9:.1f} GB")
            break
        except Exception:  # noqa: BLE001 - diagnostics are best-effort
            continue
    try:
        info = dict(
            (line.split(":")[0], int(line.split()[1]))
            for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line
        )
        parts.append(f"host available {info['MemAvailable']/1e6:.1f} GB of {info['MemTotal']/1e6:.1f}")
    except Exception:  # noqa: BLE001
        pass
    return " | ".join(parts) or "memory unavailable"


def _heartbeat(process: subprocess.Popen, log, interval: float = 60.0) -> None:
    """Print progress and memory while the child is quiet.

    nerfstudio renders its progress through Rich, which withholds output when
    stdout is a pipe, so a healthy multi-hour training is indistinguishable
    from a hung one. Checkpoint filenames give the step count instead.
    """
    started = time.monotonic()
    while process.poll() is None:
        time.sleep(interval)
        if process.poll() is not None:
            break
        step = _latest_step()
        elapsed = time.monotonic() - started
        rate = f", {step/elapsed:.1f} it/s" if step else ""
        line = (f"[heartbeat] {elapsed/60:.0f} min | "
                f"{'step ' + str(step) if step else 'no checkpoint yet'}{rate} | "
                f"{_memory()} | {_vram()}\n")
        print(line, end="", flush=True)
        try:
            log.write(line)
            log.flush()
        except Exception:  # noqa: BLE001
            pass


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
        threading.Thread(target=_heartbeat, args=(process, log), daemon=True).start()
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
