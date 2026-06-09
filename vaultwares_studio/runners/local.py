"""LocalStageRunner: subprocess execution with streamed output and cancel.

Replaces the old blocking ``subprocess.run`` in pipeline._run_command so the
GUI gets live log lines and a working cancel button. Heavy local runs are
opt-in; this runner also executes the cheap local stages (ffprobe, ffmpeg,
USD authoring).
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from typing import Callable

from .base import (
    CancelToken,
    StageCancelledError,
    StageContext,
    StageResult,
    StageRunner,
)

# Matches nerfstudio "... 1234 (12.34%) ..." iteration lines and COLMAP
# "Registering image #42" lines well enough for a coarse progress signal.
_PROGRESS_HINTS = ("it/s", "%", "Registering image", "Iteration")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        import psutil

        root = psutil.Process(proc.pid)
        children = root.children(recursive=True)
        for child in children:
            child.kill()
        root.kill()
        return
    except Exception:  # noqa: BLE001 - psutil missing or process already gone
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            check=False,
            capture_output=True,
        )
    else:
        proc.kill()


class LocalStageRunner(StageRunner):
    name = "local"

    def run(self, ctx: StageContext) -> StageResult:
        argv = ctx.params.get("argv")
        if not argv:
            raise ValueError("LocalStageRunner requires params['argv'].")
        self.run_command(
            argv,
            error_message=ctx.params.get("error_message", f"Stage {ctx.stage_key} failed."),
            timeout_seconds=int(ctx.params.get("timeout_seconds", 3600)),
            log=ctx.log,
            env=ctx.params.get("env"),
            cancel=ctx.cancel,
        )
        missing = [path for path in ctx.expected_outputs if not path.exists()]
        if missing:
            raise RuntimeError(
                f"Stage {ctx.stage_key} finished but expected outputs are missing: "
                + ", ".join(str(path) for path in missing)
            )
        return StageResult(status="complete", artifacts=list(ctx.expected_outputs))

    def run_command(
        self,
        cmd: list[str],
        *,
        error_message: str,
        timeout_seconds: int,
        log: Callable[[str], None],
        env: dict[str, str] | None = None,
        progress: Callable[[float, str], None] | None = None,
        cancel: CancelToken | None = None,
    ) -> None:
        """Run a command, streaming each output line to ``log`` as it arrives."""
        log(f"Running: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.put(line.rstrip())
            lines.put(None)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        deadline = time.monotonic() + timeout_seconds
        stream_open = True
        while True:
            if cancel is not None and cancel.cancelled:
                _kill_process_tree(proc)
                raise StageCancelledError(f"Cancelled: {error_message}")
            if time.monotonic() > deadline:
                _kill_process_tree(proc)
                raise RuntimeError(f"{error_message} (timed out after {timeout_seconds}s)")
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                if not stream_open and proc.poll() is not None:
                    break
                continue
            if line is None:
                stream_open = False
                continue
            log(line)
            if progress is not None and any(hint in line for hint in _PROGRESS_HINTS):
                progress(-1.0, line)  # -1 = indeterminate; parsers refine in M1

        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"{error_message} (exit code={returncode})")
