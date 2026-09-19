"""Follow a running HF Job's logs locally, reconnecting when the stream dies.

``fetch_job_logs`` delivers a burst and then goes quiet. The launcher opens one
stream at submit time and keeps it for the life of the job, so when that stream
stalls the local log simply stops — which on 17 Sep left a training with no
visible output for over two hours, and made a healthy run indistinguishable
from a hung one right up until it was OOMKilled.

A fresh stream works fine. So this reconnects instead of trusting one stream to
last, keeps a record of what it has already written, and only appends lines it
has not seen. It stops on its own when the job reaches a terminal stage.

    python tools/tail_job_logs.py --job-id 6aad6b...        # follow one job
    python tools/tail_job_logs.py --latest                  # newest running job
    python tools/tail_job_logs.py --latest --out train.log  # append to a file
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "TIMEOUT"}


def latest_job(api, namespace: str, running_only: bool = True):
    for job in api.list_jobs(namespace=namespace):
        if not running_only or job.status.stage not in TERMINAL:
            return job
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--latest", action="store_true", help="follow the newest non-terminal job")
    parser.add_argument("--out", type=Path, help="append here as well as printing")
    parser.add_argument("--reconnect-seconds", type=float, default=5.0)
    parser.add_argument("--max-minutes", type=float, default=240.0)
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi, fetch_job_logs

    from vaultwares_studio.runners import get_hf_token

    token = get_hf_token()
    api = HfApi(token=token)
    namespace = api.whoami()["name"]

    job_id = args.job_id
    if not job_id:
        if not args.latest:
            parser.error("pass --job-id or --latest")
        job = latest_job(api, namespace)
        if job is None:
            print("[tail] no running job", file=sys.stderr)
            return 1
        job_id = job.id
    print(f"[tail] following {job_id}", flush=True)

    sink = args.out.open("a", encoding="utf-8") if args.out else None
    # Lines can repeat across reconnects; the log has no cursor, so identity is
    # the line itself plus how many times it has been seen.
    seen: dict[str, int] = {}
    started = time.monotonic()
    last_stage = ""

    def emit(text: str) -> None:
        print(text, flush=True)
        if sink:
            sink.write(text + "\n")
            sink.flush()

    while time.monotonic() - started < args.max_minutes * 60:
        counts: dict[str, int] = {}
        try:
            for line in fetch_job_logs(job_id=job_id, token=token):
                clean = line.rstrip("\n").encode("ascii", "replace").decode("ascii")
                counts[clean] = counts.get(clean, 0) + 1
                if counts[clean] > seen.get(clean, 0):
                    seen[clean] = counts[clean]
                    emit(f"[remote] {clean}")
        except Exception as exc:  # noqa: BLE001 - reconnecting is the whole point
            emit(f"[tail] stream ended: {type(exc).__name__}: {exc}")

        info = api.inspect_job(job_id=job_id)
        stage = info.status.stage
        if stage != last_stage:
            emit(f"[tail] status: {stage}")
            last_stage = stage
        if stage in TERMINAL:
            emit(f"[tail] {job_id} finished as {stage}")
            if sink:
                sink.close()
            return 0
        time.sleep(args.reconnect_seconds)

    emit(f"[tail] gave up after {args.max_minutes:.0f} min; job still {last_stage}")
    if sink:
        sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
